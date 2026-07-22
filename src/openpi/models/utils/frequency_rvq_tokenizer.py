"""DCT frequency-decomposed action tokenizer with a low-band VQ and high-band RVQ."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as nnf

from openpi.models.utils.residual_vq import ResidualVQ
from openpi.models.utils.vector_quantize_pytorch import VectorQuantize


@dataclasses.dataclass(frozen=True)
class FrequencyRVQConfig:
    action_horizon: int = 16
    action_dim: int = 7
    low_frequency_bins: int = 4
    latent_dim: int = 128
    hidden_dim: int = 256
    low_codebook_size: int = 512
    high_codebook_size: int = 256
    num_residual_stages: int = 3
    ema_decay: float = 0.99
    kmeans_init: bool = True
    kmeans_iters: int = 20
    dead_code_threshold: float = 2.0

    def __post_init__(self) -> None:
        if not 0 < self.low_frequency_bins < self.action_horizon:
            raise ValueError("low_frequency_bins must be between zero and action_horizon")
        if self.num_residual_stages < 1:
            raise ValueError("num_residual_stages must be positive")


def make_orthonormal_dct_matrix(horizon: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Construct the orthonormal DCT-II matrix; its transpose is the inverse transform."""
    n = torch.arange(horizon, dtype=dtype)
    k = torch.arange(horizon, dtype=dtype)[:, None]
    matrix = torch.cos(math.pi / horizon * (n + 0.5) * k)
    matrix[0] *= math.sqrt(1.0 / horizon)
    matrix[1:] *= math.sqrt(2.0 / horizon)
    return matrix


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class FrequencyRVQTokenizerModel(nn.Module):
    """Maps one normalized action chunk to four coarse-to-fine discrete codes."""

    def __init__(self, config: FrequencyRVQConfig):
        super().__init__()
        self.config = config
        self.register_buffer("dct_matrix", make_orthonormal_dct_matrix(config.action_horizon))

        low_dim = config.low_frequency_bins * config.action_dim
        high_dim = (config.action_horizon - config.low_frequency_bins) * config.action_dim
        self.low_encoder = MLP(low_dim, config.hidden_dim, config.latent_dim)
        self.high_encoder = MLP(high_dim, config.hidden_dim, config.latent_dim)
        self.low_decoder = MLP(config.latent_dim, config.hidden_dim, low_dim)
        self.high_decoder = MLP(config.latent_dim, config.hidden_dim, high_dim)

        quantizer_kwargs = {
            "decay": config.ema_decay,
            "kmeans_init": config.kmeans_init,
            "kmeans_iters": config.kmeans_iters,
            "threshold_ema_dead_code": config.dead_code_threshold,
            "commitment_weight": 1.0,
            "ema_update": True,
            "learnable_codebook": False,
        }
        self.low_vq = VectorQuantize(
            dim=config.latent_dim,
            codebook_size=config.low_codebook_size,
            **quantizer_kwargs,
        )
        self.high_rvq = ResidualVQ(
            dim=config.latent_dim,
            num_quantizers=config.num_residual_stages,
            codebook_size=config.high_codebook_size,
            shared_codebook=False,
            quantize_dropout=False,
            **quantizer_kwargs,
        )

    def dct(self, actions: torch.Tensor) -> torch.Tensor:
        self._validate_actions(actions)
        matrix = self.dct_matrix.to(dtype=actions.dtype)
        return torch.einsum("kn,bnd->bkd", matrix, actions)

    def idct(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.ndim != 3 or coefficients.shape[1:] != (
            self.config.action_horizon,
            self.config.action_dim,
        ):
            raise ValueError(
                f"Expected coefficients [B,{self.config.action_horizon},{self.config.action_dim}], "
                f"got {tuple(coefficients.shape)}"
            )
        matrix = self.dct_matrix.to(dtype=coefficients.dtype)
        return torch.einsum("kn,bkd->bnd", matrix, coefficients)

    def encode_latents(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coefficients = self.dct(actions)
        split = self.config.low_frequency_bins
        low = coefficients[:, :split]
        high = coefficients[:, split:]
        h_low = self.low_encoder(low.flatten(start_dim=1))
        h_high = self.high_encoder(high.flatten(start_dim=1))
        return coefficients, h_low, h_high

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode actions without modifying EMA codebooks."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                _, h_low, h_high = self.encode_latents(actions)
                _, low_ids, _ = self.low_vq(h_low, freeze_codebook=True)
                _, residual_ids, _ = self.high_rvq(h_high[:, None], return_all_codes=False)
                return torch.cat((low_ids[:, None], residual_ids[:, 0]), dim=-1)
        finally:
            self.train(was_training)

    def decode(self, code_ids: torch.Tensor, *, num_residual_stages: int | None = None) -> torch.Tensor:
        if code_ids.ndim != 2 or code_ids.shape[1] != 1 + self.config.num_residual_stages:
            raise ValueError(
                f"Expected code IDs [B,{1 + self.config.num_residual_stages}], got {tuple(code_ids.shape)}"
            )
        stages = self.config.num_residual_stages if num_residual_stages is None else num_residual_stages
        if not 0 <= stages <= self.config.num_residual_stages:
            raise ValueError(f"num_residual_stages must be in [0,{self.config.num_residual_stages}]")

        low_ids = code_ids[:, 0]
        if torch.any((low_ids < 0) | (low_ids >= self.config.low_codebook_size)):
            raise ValueError("Low-band code ID is outside its codebook")
        low_codebook = self.low_vq.codebook
        q_low = low_codebook[low_ids]

        q_high = torch.zeros((code_ids.shape[0], self.config.latent_dim), device=code_ids.device, dtype=q_low.dtype)
        high_codebooks = self.high_rvq.codebooks
        for stage in range(stages):
            stage_ids = code_ids[:, stage + 1]
            if torch.any((stage_ids < 0) | (stage_ids >= self.config.high_codebook_size)):
                raise ValueError(f"Residual stage {stage} code ID is outside its codebook")
            q_high = q_high + high_codebooks[stage, stage_ids]

        batch = code_ids.shape[0]
        split = self.config.low_frequency_bins
        low = self.low_decoder(q_low).reshape(batch, split, self.config.action_dim)
        high = self.high_decoder(q_high).reshape(batch, self.config.action_horizon - split, self.config.action_dim)
        return self.idct(torch.cat((low, high), dim=1))

    def forward(self, actions: torch.Tensor, *, quantize: bool = True) -> dict[str, torch.Tensor]:
        coefficients, h_low, h_high = self.encode_latents(actions)
        if quantize:
            q_low, low_ids, low_commit = self.low_vq(h_low)
            q_high_seq, residual_ids, residual_commit = self.high_rvq(h_high[:, None])
            q_high = q_high_seq[:, 0]
            code_ids = torch.cat((low_ids[:, None], residual_ids[:, 0]), dim=-1)
            commitment = low_commit.sum() + residual_commit.sum()
        else:
            q_low, q_high = h_low, h_high
            code_ids = torch.full(
                (actions.shape[0], 1 + self.config.num_residual_stages),
                -1,
                dtype=torch.long,
                device=actions.device,
            )
            commitment = torch.zeros((), dtype=actions.dtype, device=actions.device)

        batch = actions.shape[0]
        split = self.config.low_frequency_bins
        predicted_low = self.low_decoder(q_low).reshape(batch, split, self.config.action_dim)
        predicted_high = self.high_decoder(q_high).reshape(
            batch, self.config.action_horizon - split, self.config.action_dim
        )
        reconstructed = self.idct(torch.cat((predicted_low, predicted_high), dim=1))
        return {
            "code_ids": code_ids,
            "coefficients": coefficients,
            "predicted_low": predicted_low,
            "predicted_high": predicted_high,
            "reconstructed": reconstructed,
            "commitment": commitment,
        }

    def compute_loss(
        self,
        actions: torch.Tensor,
        *,
        quantize: bool = True,
        frequency_weight: float = 0.5,
        velocity_weight: float = 0.2,
        commitment_weight: float = 0.25,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self(actions, quantize=quantize)
        split = self.config.low_frequency_bins
        action_l1 = nnf.l1_loss(output["reconstructed"], actions)
        low_frequency_l1 = nnf.l1_loss(output["predicted_low"], output["coefficients"][:, :split])
        high_frequency_l1 = nnf.l1_loss(output["predicted_high"], output["coefficients"][:, split:])
        frequency_l1 = low_frequency_l1 + 0.5 * high_frequency_l1
        velocity_l1 = nnf.l1_loss(
            torch.diff(output["reconstructed"], dim=1),
            torch.diff(actions, dim=1),
        )
        loss = (
            action_l1
            + frequency_weight * frequency_l1
            + velocity_weight * velocity_l1
            + commitment_weight * output["commitment"]
        )
        metrics = {
            "loss": loss.detach(),
            "action_l1": action_l1.detach(),
            "frequency_l1": frequency_l1.detach(),
            "low_frequency_l1": low_frequency_l1.detach(),
            "high_frequency_l1": high_frequency_l1.detach(),
            "velocity_l1": velocity_l1.detach(),
            "commitment": output["commitment"].detach(),
        }
        if quantize:
            metrics.update(self.codebook_metrics(output["code_ids"]))
        return loss, metrics

    def codebook_metrics(self, code_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        sizes = [self.config.low_codebook_size] + [self.config.high_codebook_size] * self.config.num_residual_stages
        names = ["low"] + [f"res{stage + 1}" for stage in range(self.config.num_residual_stages)]
        metrics: dict[str, torch.Tensor] = {}
        for column, (name, size) in enumerate(zip(names, sizes, strict=True)):
            counts = torch.bincount(code_ids[:, column], minlength=size).float()
            probabilities = counts / counts.sum().clamp_min(1.0)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            metrics[f"{name}_perplexity"] = entropy.exp().detach()
            metrics[f"{name}_batch_unused_ratio"] = (counts == 0).float().mean().detach()
        return metrics

    def save_pretrained(self, directory: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(
            json.dumps(dataclasses.asdict(self.config), indent=2, sort_keys=True), encoding="utf-8"
        )
        if metadata is not None:
            (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        torch.save(self.state_dict(), directory / "model.pt")

    @classmethod
    def from_pretrained(
        cls, directory: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> FrequencyRVQTokenizerModel:
        directory = Path(directory)
        config = FrequencyRVQConfig(**json.loads((directory / "config.json").read_text(encoding="utf-8")))
        model = cls(config)
        state_dict = torch.load(directory / "model.pt", map_location=map_location, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _validate_actions(self, actions: torch.Tensor) -> None:
        if actions.ndim != 3 or actions.shape[1:] != (
            self.config.action_horizon,
            self.config.action_dim,
        ):
            raise ValueError(
                f"Expected actions [B,{self.config.action_horizon},{self.config.action_dim}], "
                f"got {tuple(actions.shape)}"
            )
