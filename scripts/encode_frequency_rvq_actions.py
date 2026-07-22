"""Offline-encode prepared action chunks with a frozen frequency-RVQ tokenizer."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch
import tyro

from openpi.models.utils.frequency_rvq_tokenizer import FrequencyRVQTokenizerModel


@dataclasses.dataclass(frozen=True)
class Args:
    data_dir: str
    tokenizer_path: str
    output_dir: str
    batch_size: int = 2048
    device: str = "cuda"


def main(args: Args) -> None:
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    model = FrequencyRVQTokenizerModel.from_pretrained(args.tokenizer_path, map_location=device).to(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        input_path = Path(args.data_dir) / f"{split}.npz"
        with np.load(input_path) as data:
            actions = data["actions"].astype(np.float32)
            episode_ids = data["episode_ids"]
            starts = data["starts"]
        encoded: list[np.ndarray] = []
        for start in range(0, len(actions), args.batch_size):
            batch = torch.from_numpy(actions[start : start + args.batch_size]).to(device)
            encoded.append(model.encode(batch).cpu().numpy().astype(np.int16))
        code_ids = np.concatenate(encoded) if encoded else np.empty((0, 4), dtype=np.int16)
        np.savez_compressed(
            output_dir / f"{split}.npz",
            code_ids=code_ids,
            episode_ids=episode_ids,
            starts=starts,
        )
        print(f"{split}: {len(code_ids)} chunks")


if __name__ == "__main__":
    main(tyro.cli(Args))
