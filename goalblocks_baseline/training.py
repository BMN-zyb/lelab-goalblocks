"""Training helpers used by the GoalBlocks experiment notebook."""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import ACTION_KEY, STATE_KEY, GoalConditionedMultiRepoDataset, goalblocks_collate


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stats_vector(dataset: GoalConditionedMultiRepoDataset, key: str, stat: str) -> torch.Tensor:
    value = dataset.stats[key][stat]
    return torch.as_tensor(value, dtype=torch.float32).reshape(-1)


def make_dataloader(
    dataset: GoalConditionedMultiRepoDataset,
    batch_size: int,
    *,
    balanced: bool,
    num_workers: int = 2,
    seed: int = 42,
) -> DataLoader:
    sampler = dataset.balanced_sampler(seed=seed) if balanced else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=not balanced,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=goalblocks_collate,
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def train_steps(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    steps: int,
    grad_clip_norm: float = 1.0,
) -> list[float]:
    if steps < 1:
        return []
    model.train()
    iterator: Iterable = iter(dataloader)
    losses: list[float] = []
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, max_batches: int = 20):
    model.eval()
    losses = []
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= max_batches:
            break
        losses.append(float(model(move_batch(batch, device))["loss"].cpu()))
    return sum(losses) / len(losses) if losses else float("nan")


def default_normalization(dataset: GoalConditionedMultiRepoDataset) -> dict[str, torch.Tensor]:
    return {
        "state_mean": stats_vector(dataset, STATE_KEY, "mean"),
        "state_std": stats_vector(dataset, STATE_KEY, "std"),
        "action_mean": stats_vector(dataset, ACTION_KEY, "mean"),
        "action_std": stats_vector(dataset, ACTION_KEY, "std"),
    }


def save_training_state(
    model,
    optimizer: torch.optim.Optimizer,
    output_dir: str | Path,
    losses: list[float],
) -> Path:
    output = Path(output_dir)
    checkpoint = model.save_checkpoint(output)
    torch.save({"optimizer": optimizer.state_dict(), "losses": losses}, output / "training_state.pt")
    return checkpoint
