import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.predict_future_states_enabled = config.predict_future_states
        self.future_state_loss_weight = config.future_state_loss_weight
        self.detach_future_state_condition = config.detach_future_state_condition
        self.state_dim = config.state_dim
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_expert_width = action_expert_config.width
        if config.predict_future_states:
            self.future_state_mlp_in = nnx.Linear(
                paligemma_config.width + config.action_dim, action_expert_config.width, rngs=rngs
            )
            self.future_state_mlp_out = nnx.Linear(
                action_expert_config.width, config.action_horizon * config.action_dim, rngs=rngs
            )
            self.future_state_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.aligned_future_state_in_proj = nnx.Linear(
                config.action_dim, action_expert_config.width, rngs=rngs
            )
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        future_state_condition: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        if future_state_condition is not None:
            # Preserve the exact q_{t+k+1} <-> a_{t+k} correspondence in addition to
            # allowing every action query to attend to the full state-trajectory KV block.
            action_tokens += self.aligned_future_state_in_proj(future_state_condition)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def predict_future_state_trajectory(self, prefix_out, prefix_mask, current_state):
        """Predict q_{t+1:t+H} through a numerical, explicitly supervised bottleneck."""
        weights = prefix_mask[..., None].astype(prefix_out.dtype)
        pooled = jnp.sum(prefix_out * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1)
        hidden = self.future_state_mlp_in(jnp.concatenate([pooled, current_state], axis=-1))
        hidden = nnx.swish(hidden)
        deltas = self.future_state_mlp_out(hidden).reshape(
            current_state.shape[0], self.action_horizon, self.action_dim
        )
        predicted = current_state[:, None, :] + jnp.cumsum(deltas, axis=1)
        state_dim_mask = jnp.arange(self.action_dim) < self.state_dim
        return jnp.where(state_dim_mask[None, None, :], predicted, 0.0)

    def embed_state_trajectory(self, state_trajectory):
        tokens = self.future_state_in_proj(state_trajectory)
        trajectory_length = state_trajectory.shape[1]
        step_ids = jnp.arange(trajectory_length, dtype=jnp.float32)
        step_emb = posemb_sincos(
            step_ids,
            self.action_expert_width,
            min_period=1.0,
            max_period=max(float(trajectory_length), 2.0),
        )
        return tokens + step_emb[None, :, :]

    def _append_state_cache(self, prefix_kv, prefix_mask, current_state, future_states):
        """Re-encode numerical predicted states and append them to the transformer KV cache."""
        batch_size = future_states.shape[0]
        state_trajectory = jnp.concatenate([current_state[:, None, :], future_states], axis=1)
        state_tokens = self.embed_state_trajectory(state_trajectory)
        trajectory_length = self.action_horizon + 1
        state_mask = jnp.ones((batch_size, trajectory_length), dtype=jnp.bool_)
        state_ar = jnp.array([True] + [False] * (trajectory_length - 1))
        state_self_mask = make_attn_mask(state_mask, state_ar)
        prefix_to_state = einops.repeat(prefix_mask, "b p -> b s p", s=trajectory_length)
        full_mask = jnp.concatenate([prefix_to_state, state_self_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.arange(trajectory_length)[None, :]
        zero_cond = jnp.zeros((batch_size, self.action_expert_width), dtype=state_tokens.dtype)
        (_, state_out), state_kv = self.PaliGemma.llm(
            [None, state_tokens],
            mask=full_mask,
            positions=positions,
            kv_cache=prefix_kv,
            adarms_cond=[None, zero_cond],
        )
        assert state_out is not None
        return state_kv, state_mask

    def _future_state_loss(self, observation, predicted_states):
        if observation.future_states is None:
            raise ValueError("predict_future_states=True requires future_states in the training batch")
        error = predicted_states - observation.future_states
        abs_error = jnp.abs(error)
        huber = jnp.where(abs_error < 1.0, 0.5 * jnp.square(error), abs_error - 0.5)
        if observation.future_state_dim_mask is not None:
            dim_mask = observation.future_state_dim_mask[:, None, :].astype(huber.dtype)
            state_loss = jnp.sum(huber * dim_mask, axis=-1) / jnp.maximum(jnp.sum(dim_mask, axis=-1), 1)
        else:
            state_loss = jnp.mean(huber, axis=-1)
        if observation.future_state_mask is not None:
            state_loss *= observation.future_state_mask.astype(state_loss.dtype)
        return state_loss

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        if self.predict_future_states_enabled:
            # Stage 1: encode the observation and predict a numerical state trajectory.
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
            (prefix_out, _), prefix_kv = self.PaliGemma.llm(
                [prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions
            )
            assert prefix_out is not None
            predicted_states = self.predict_future_state_trajectory(prefix_out, prefix_mask, observation.state)
            state_condition = (
                jax.lax.stop_gradient(predicted_states)
                if self.detach_future_state_condition
                else predicted_states
            )

            # Stage 2: the numerical prediction is the only state-predictor output exposed to the
            # action model. Re-encode it once into a time-independent state KV block.
            state_kv, state_mask = self._append_state_cache(
                prefix_kv, prefix_mask, observation.state, state_condition
            )

            # Stage 3: action flow matching attends to observation KV + predicted-state KV.
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, time, state_condition
            )
            suffix_self_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            cached_mask = jnp.concatenate([prefix_mask, state_mask], axis=1)
            cached_to_action = einops.repeat(cached_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_mask = jnp.concatenate([cached_to_action, suffix_self_mask], axis=-1)
            positions = jnp.sum(cached_mask, axis=-1)[:, None] + jnp.arange(suffix_tokens.shape[1])[None, :]
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_mask,
                positions=positions,
                kv_cache=state_kv,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
            state_loss = self._future_state_loss(observation, predicted_states)
            return action_loss + self.future_state_loss_weight * state_loss

        # Original pi0/pi0.5 path: one big forward pass of prefix + suffix at once.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # First fill KV cache with a forward pass of the prefix.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        cached_mask = prefix_mask
        if self.predict_future_states_enabled:
            assert prefix_out is not None
            predicted_states = self.predict_future_state_trajectory(prefix_out, prefix_mask, observation.state)
            kv_cache, state_mask = self._append_state_cache(
                kv_cache, prefix_mask, observation.state, predicted_states
            )
            cached_mask = jnp.concatenate([prefix_mask, state_mask], axis=1)
        else:
            predicted_states = None

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size), predicted_states
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(cached_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                cached_mask.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(cached_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
