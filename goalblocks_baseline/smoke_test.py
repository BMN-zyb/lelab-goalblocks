"""Run one real-data optimizer step across both GoalBlocks repositories."""

from __future__ import annotations

import argparse

import torch

from .dataset import GoalConditionedMultiRepoDataset, goalblocks_collate
from .model import GoalConditionedActionChunkPolicy, GoalPolicyConfig
from .training import default_normalization, move_batch, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    repo_ids = [
        "bmnzyb/GoalBloakcs_1_20260906_155729",
        "bmnzyb/GoalBloakcs_1_20260906_162409",
    ]
    set_seed(42)
    dataset = GoalConditionedMultiRepoDataset(
        repo_ids,
        root=args.root,
        action_horizon=8,
        image_size=(96, 96),
    )
    samples = [dataset[0], dataset[dataset.cumulative_lengths[0]]]
    batch = goalblocks_collate(samples)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = GoalConditionedActionChunkPolicy(
        GoalPolicyConfig(action_horizon=8, hidden_dim=96, text_dim=48),
        **default_normalization(dataset),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    output = model(move_batch(batch, device))
    output["loss"].backward()
    optimizer.step()
    print(f"frames={len(dataset)} lengths={dataset.lengths} fps={dataset.fps}")
    print(f"top={tuple(batch['observation.images.top'].shape)}")
    print(f"wrist={tuple(batch['observation.images.wrist'].shape)}")
    print(f"goal={tuple(batch['goal.image'].shape)} state={tuple(batch['observation.state'].shape)}")
    print(f"action={tuple(batch['action'].shape)} loss={float(output['loss'].detach().cpu()):.6f}")
    for goal in dataset.goals:
        print(f"goal repo={goal.repo_id} source={goal.source} images={list(goal.image_paths)} task={goal.task}")


if __name__ == "__main__":
    main()
