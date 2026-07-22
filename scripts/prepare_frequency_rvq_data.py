"""Prepare episode-safe normalized action chunks for the frequency-RVQ tokenizer.

Example:
    uv run scripts/prepare_frequency_rvq_data.py \
        --repo-id physical-intelligence/libero \
        --output-dir data/frequency_rvq_libero

The script splits complete episodes before extracting overlapping windows. This prevents
near-identical windows from the same trajectory leaking across train/validation/test.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    repo_id: str
    output_dir: str
    action_key: str = "actions"
    state_key: str = "state"
    episode_key: str = "episode_index"
    action_horizon: int = 16
    stride: int = 2
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    seed: int = 42
    q_low: float = 0.01
    q_high: float = 0.99
    max_episodes: int | None = None


def _scalar(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def load_episodes(args: Args) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Load one action at a time and group it by episode without terminal padding."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError("LeRobot is required; run this script through `uv run` in the openpi repository") from exc

    dataset = LeRobotDataset(args.repo_id)
    source = getattr(dataset, "hf_dataset", dataset)
    action_episodes: dict[int, list[np.ndarray]] = {}
    state_episodes: dict[int, list[np.ndarray]] = {}
    for index in range(len(source)):
        sample = source[index]
        if args.action_key not in sample:
            available = ", ".join(sorted(sample))
            raise KeyError(f"Action key {args.action_key!r} not found. Available keys: {available}")
        if args.episode_key not in sample:
            available = ", ".join(sorted(sample))
            raise KeyError(f"Episode key {args.episode_key!r} not found. Available keys: {available}")
        episode_id = _scalar(sample[args.episode_key])
        if episode_id not in action_episodes:
            if args.max_episodes is not None and len(action_episodes) >= args.max_episodes:
                continue
            action_episodes[episode_id] = []
            state_episodes[episode_id] = []
        if args.state_key not in sample:
            available = ", ".join(sorted(sample))
            raise KeyError(f"State key {args.state_key!r} not found. Available keys: {available}")
        action_episodes[episode_id].append(np.asarray(sample[args.action_key], dtype=np.float32))
        state_episodes[episode_id].append(np.asarray(sample[args.state_key], dtype=np.float32))

    actions_result = {episode_id: np.stack(actions) for episode_id, actions in action_episodes.items()}
    states_result = {episode_id: np.stack(states) for episode_id, states in state_episodes.items()}
    if not actions_result:
        raise ValueError("No episodes were loaded")
    action_dims = {actions.shape[-1] for actions in actions_result.values()}
    if len(action_dims) != 1:
        raise ValueError(f"All episodes must use the same action dimension, got {sorted(action_dims)}")
    return actions_result, states_result


def split_episode_ids(args: Args, episode_ids: list[int]) -> dict[str, list[int]]:
    if not 0 < args.train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0 <= args.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if args.train_fraction + args.validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1")
    shuffled = episode_ids.copy()
    random.Random(args.seed).shuffle(shuffled)
    train_end = max(1, round(len(shuffled) * args.train_fraction))
    validation_end = train_end + round(len(shuffled) * args.validation_fraction)
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def extract_chunks(
    episodes: dict[int, np.ndarray], episode_ids: list[int], *, horizon: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunks: list[np.ndarray] = []
    chunk_episode_ids: list[int] = []
    starts: list[int] = []
    for episode_id in episode_ids:
        actions = episodes[episode_id]
        for start in range(0, len(actions) - horizon + 1, stride):
            chunks.append(actions[start : start + horizon])
            chunk_episode_ids.append(episode_id)
            starts.append(start)
    if not chunks:
        action_dim = next(iter(episodes.values())).shape[-1]
        return (
            np.empty((0, horizon, action_dim), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.stack(chunks).astype(np.float32),
        np.asarray(chunk_episode_ids, dtype=np.int64),
        np.asarray(starts, dtype=np.int64),
    )


def normalize(actions: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    scale = np.maximum(q99 - q01, 1e-6)
    return np.clip(2.0 * (actions - q01) / scale - 1.0, -1.0, 1.0).astype(np.float32)


def main(args: Args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes, state_episodes = load_episodes(args)
    split_ids = split_episode_ids(args, sorted(episodes))

    raw_splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split, episode_ids in split_ids.items():
        raw_splits[split] = extract_chunks(episodes, episode_ids, horizon=args.action_horizon, stride=args.stride)
    train_actions = raw_splits["train"][0]
    if len(train_actions) == 0:
        raise ValueError("The training split produced no complete action chunks")
    q01 = np.quantile(train_actions, args.q_low, axis=(0, 1)).astype(np.float32)
    q99 = np.quantile(train_actions, args.q_high, axis=(0, 1)).astype(np.float32)
    if np.any(q99 - q01 < 1e-6):
        constant_dims = np.flatnonzero(q99 - q01 < 1e-6).tolist()
        raise ValueError(f"Quantile range is zero for action dimensions {constant_dims}")

    np.savez(output_dir / "norm_stats.npz", q01=q01, q99=q99)
    train_states = np.concatenate([state_episodes[episode_id] for episode_id in split_ids["train"]], axis=0)
    state_q01 = np.quantile(train_states, args.q_low, axis=0).astype(np.float32)
    state_q99 = np.quantile(train_states, args.q_high, axis=0).astype(np.float32)
    norm_stats = {
        "norm_stats": {
            "actions": {
                "mean": np.mean(train_actions, axis=(0, 1)).tolist(),
                "std": np.std(train_actions, axis=(0, 1)).tolist(),
                "q01": q01.tolist(),
                "q99": q99.tolist(),
            },
            "state": {
                "mean": np.mean(train_states, axis=0).tolist(),
                "std": np.std(train_states, axis=0).tolist(),
                "q01": state_q01.tolist(),
                "q99": state_q99.tolist(),
            },
        }
    }
    (output_dir / "norm_stats.json").write_text(json.dumps(norm_stats, indent=2, sort_keys=True), encoding="utf-8")
    split_counts: dict[str, int] = {}
    for split, (actions, episode_ids, starts) in raw_splits.items():
        normalized = normalize(actions, q01, q99)
        np.savez_compressed(
            output_dir / f"{split}.npz",
            actions=normalized,
            episode_ids=episode_ids,
            starts=starts,
        )
        split_counts[split] = len(actions)

    metadata = {
        **dataclasses.asdict(args),
        "num_episodes": len(episodes),
        "action_dim": int(train_actions.shape[-1]),
        "split_episode_ids": split_ids,
        "split_chunk_counts": split_counts,
        "normalization": "per-dimension clipped train-set quantiles mapped to [-1, 1]",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **split_counts}, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
