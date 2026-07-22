"""Pi0FAST variant that predicts four frequency-RVQ action codes."""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_fast
from openpi.models.utils.frequency_rvq_tokens import DEFAULT_FREQUENCY_RVQ_LAYOUT
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class Pi0FrequencyRVQConfig(pi0_fast.Pi0FASTConfig):
    """Architecture-compatible with pi0_fast_base; the tokenizer remains external and frozen."""

    @override
    def create(self, rng: at.KeyArrayLike) -> Pi0FrequencyRVQ:
        return Pi0FrequencyRVQ(self, rngs=nnx.Rngs(rng))


class Pi0FrequencyRVQ(pi0_fast.Pi0FAST):
    _stage_weights = (2.0, 1.0, 0.7, 0.5)

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b"]:
        del actions
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )
        input_embeddings, input_mask, ar_mask = self.embed_inputs(observation)
        attention_mask = pi0_fast.make_attn_mask(input_mask, ar_mask)
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=input_embeddings[:, :-1],
            mask=attention_mask[:, :-1, :-1],
            return_prelogits=True,
        )
        assert observation.tokenized_prompt is not None
        assert observation.token_loss_mask is not None
        targets = observation.tokenized_prompt[:, 1:]
        loss_mask = observation.token_loss_mask[:, 1:]
        # Image tokens precede the text sequence but do not have language-token targets.
        pre_logits = pre_logits[:, -targets.shape[1] :]
        logits, _ = self.PaliGemma.llm(pre_logits=pre_logits)
        stage_indices = jnp.cumsum(loss_mask.astype(jnp.int32), axis=-1) - 1
        vocab_size = self.PaliGemma.llm.module.vocab_size
        weighted_loss = jnp.zeros(targets.shape, dtype=jnp.float32)
        total_weight = jnp.zeros(targets.shape, dtype=jnp.float32)

        for stage, weight in enumerate(self._stage_weights):
            low, high = DEFAULT_FREQUENCY_RVQ_LAYOUT.paligemma_range(stage, vocab_size)
            stage_logits = logits[..., low : high + 1]
            target_classes = jnp.clip(targets - low, 0, high - low)
            target_log_probs = jnp.take_along_axis(
                jax.nn.log_softmax(stage_logits, axis=-1), target_classes[..., None], axis=-1
            )[..., 0]
            stage_mask = loss_mask & (stage_indices == stage)
            weighted_loss = weighted_loss - target_log_probs * stage_mask * weight
            total_weight = total_weight + stage_mask * weight

        return jnp.sum(weighted_loss, axis=-1) / jnp.clip(jnp.sum(total_weight, axis=-1), 1.0)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int = 4,
        temperature: float = 0.0,
    ) -> _model.Actions:
        del rng, temperature
        if max_decoding_steps != DEFAULT_FREQUENCY_RVQ_LAYOUT.num_stages:
            raise ValueError(
                f"Frequency-RVQ always decodes {DEFAULT_FREQUENCY_RVQ_LAYOUT.num_stages} tokens, "
                f"got max_decoding_steps={max_decoding_steps}"
            )
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )
        prefix_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs(observation)
        prefix_attention_mask = pi0_fast.make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_embeddings, prefix_mask, prefix_attention_mask = pi0_fast.left_to_right_align(
            prefix_embeddings, prefix_mask, prefix_attention_mask
        )
        prefill_size = prefix_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len
        prefix_attention_mask = jnp.pad(
            prefix_attention_mask,
            ((0, 0), (0, 0), (0, DEFAULT_FREQUENCY_RVQ_LAYOUT.num_stages)),
        )
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        prefix_logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=prefix_embeddings,
            mask=prefix_attention_mask,
            positions=prefix_positions,
            decode=True,
        )
        last_logit = prefix_logits[:, -1:]
        output_tokens = jnp.zeros((last_logit.shape[0], DEFAULT_FREQUENCY_RVQ_LAYOUT.num_stages), dtype=jnp.int32)
        vocab_size = self.PaliGemma.llm.module.vocab_size

        for stage in range(DEFAULT_FREQUENCY_RVQ_LAYOUT.num_stages):
            low, high = DEFAULT_FREQUENCY_RVQ_LAYOUT.paligemma_range(stage, vocab_size)
            token = jnp.argmax(last_logit[..., low : high + 1], axis=-1).astype(jnp.int32) + low
            output_tokens = output_tokens.at[:, stage].set(token[:, 0])
            token_embedding = self.PaliGemma.llm(token, embed_only=True)
            positions = prefill_len[:, None] + stage + 1
            cache_positions = jnp.arange(prefill_size + DEFAULT_FREQUENCY_RVQ_LAYOUT.num_stages)[None, None, :]
            mask = jnp.logical_and(
                cache_positions >= prefix_start[:, None, None],
                cache_positions < jnp.broadcast_to(prefill_size + stage + 1, (prefix_start.shape[0], 1, 1)),
            )
            last_logit, kv_cache, _ = self.PaliGemma.llm(
                embedded_prefix=token_embedding,
                mask=mask,
                positions=positions,
                decode=True,
                kv_cache=kv_cache,
            )
        return output_tokens
