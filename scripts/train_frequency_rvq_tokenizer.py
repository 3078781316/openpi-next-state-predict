"""Train and evaluate the standalone PyTorch frequency-RVQ action tokenizer."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tyro

from openpi.models.utils.frequency_rvq_tokenizer import FrequencyRVQConfig
from openpi.models.utils.frequency_rvq_tokenizer import FrequencyRVQTokenizerModel


@dataclasses.dataclass(frozen=True)
class Args:
    data_dir: str
    output_dir: str
    epochs: int = 100
    pretrain_epochs: int = 5
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"
    amp: bool = True
    gradient_clip: float = 1.0
    latent_dim: int = 128
    hidden_dim: int = 256
    low_frequency_bins: int = 4
    low_codebook_size: int = 512
    high_codebook_size: int = 256
    num_residual_stages: int = 3
    ema_decay: float = 0.99
    kmeans_iters: int = 20
    dead_code_threshold: float = 2.0
    save_every: int = 10


class ActionChunkDataset(Dataset[torch.Tensor]):
    def __init__(self, path: Path):
        with np.load(path) as data:
            self.actions = torch.from_numpy(data["actions"].astype(np.float32))

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.actions[index]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset: Dataset, args: Args, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        drop_last=shuffle,
    )


def reduce_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = set().union(*(metric.keys() for metric in metrics))
    return {key: float(np.mean([metric[key] for metric in metrics if key in metric])) for key in sorted(keys)}


@torch.no_grad()
def validate(model: FrequencyRVQTokenizerModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    collected: list[dict[str, float]] = []
    stage_errors = [0.0] * (model.config.num_residual_stages + 1)
    batches = 0
    for action_batch in loader:
        actions = action_batch.to(device, non_blocking=True)
        _, metrics = model.compute_loss(actions, quantize=True)
        code_ids = model.encode(actions)
        for stages in range(model.config.num_residual_stages + 1):
            reconstruction = model.decode(code_ids, num_residual_stages=stages)
            stage_errors[stages] += torch.mean(torch.abs(reconstruction - actions)).item()
        collected.append({key: value.item() for key, value in metrics.items()})
        batches += 1
    result = reduce_metrics(collected)
    for stages, error in enumerate(stage_errors):
        result[f"reconstruction_residual_stages_{stages}"] = error / max(batches, 1)
    return result


def main(args: Args) -> None:
    seed_everything(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = ActionChunkDataset(data_dir / "train.npz")
    validation_dataset = ActionChunkDataset(data_dir / "validation.npz")
    if len(validation_dataset) == 0:
        raise ValueError("Validation split is empty")
    action_horizon, action_dim = train_dataset.actions.shape[1:]
    config = FrequencyRVQConfig(
        action_horizon=action_horizon,
        action_dim=action_dim,
        low_frequency_bins=args.low_frequency_bins,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        low_codebook_size=args.low_codebook_size,
        high_codebook_size=args.high_codebook_size,
        num_residual_stages=args.num_residual_stages,
        ema_decay=args.ema_decay,
        kmeans_init=True,
        kmeans_iters=args.kmeans_iters,
        dead_code_threshold=args.dead_code_threshold,
    )
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    model = FrequencyRVQTokenizerModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    train_loader = make_loader(train_dataset, args, shuffle=True)
    validation_loader = make_loader(validation_dataset, args, shuffle=False)
    best_validation = float("inf")
    history_path = output_dir / "history.jsonl"

    for epoch in range(args.epochs):
        model.train()
        quantize = epoch >= args.pretrain_epochs
        train_metrics: list[dict[str, float]] = []
        for action_batch in train_loader:
            actions = action_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
                loss, metrics = model.compute_loss(actions, quantize=quantize)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            train_metrics.append({key: value.item() for key, value in metrics.items()})

        validation_metrics = validate(model, validation_loader, device) if quantize else {}
        record = {
            "epoch": epoch,
            "quantize": quantize,
            "train": reduce_metrics(train_metrics),
            "validation": validation_metrics,
        }
        with history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, indent=2, sort_keys=True))

        metadata = {"epoch": epoch, "data_dir": str(data_dir), "validation": validation_metrics}
        model.save_pretrained(output_dir / "last", metadata=metadata)
        if quantize and validation_metrics["action_l1"] < best_validation:
            best_validation = validation_metrics["action_l1"]
            model.save_pretrained(output_dir / "best", metadata=metadata)
        if (epoch + 1) % args.save_every == 0:
            model.save_pretrained(output_dir / f"epoch_{epoch + 1:04d}", metadata=metadata)

    norm_stats = data_dir / "norm_stats.npz"
    if norm_stats.exists():
        import shutil

        shutil.copy2(norm_stats, output_dir / "best" / "norm_stats.npz")
        shutil.copy2(norm_stats, output_dir / "last" / "norm_stats.npz")


if __name__ == "__main__":
    main(tyro.cli(Args))
