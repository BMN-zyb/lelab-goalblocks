"""Production-style training entry point for the GoalBlocks baseline."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

from .dataset import ACTION_KEY, STATE_KEY, GoalConditionedMultiRepoDataset
from .model import GoalConditionedActionChunkPolicy, GoalPolicyConfig
from .training import default_normalization, evaluate, make_dataloader, move_batch, set_seed

DEFAULT_REPOS = (
    "bmnzyb/GoalBloakcs_1_20260906_155729",
    "bmnzyb/GoalBloakcs_1_20260906_162409",
)
ACTION_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


@dataclass
class TrainConfig:
    repo_ids: tuple[str, ...] = DEFAULT_REPOS
    data_root: str = "/workspace/goalblocks_data_cache"
    output_dir: str = "/workspace/outputs/goalblocks_baseline_formal"
    train_episodes: str = "0,1;0"
    val_episodes: str = "2;1"
    steps: int = 10_000
    batch_size: int = 16
    num_workers: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 250
    eval_every: int = 250
    save_every: int = 500
    log_every: int = 20
    eval_batches: int = 50
    action_horizon: int = 16
    image_size: int = 128
    hidden_dim: int = 192
    text_dim: int = 96
    dropout: float = 0.1
    grad_clip_norm: float = 1.0
    seed: int = 42
    device: str = "cuda"
    amp: str = "bf16"
    resume: str | None = None


def parse_episode_groups(spec: str, repo_ids: tuple[str, ...]) -> dict[str, list[int]]:
    groups = spec.split(";")
    if len(groups) != len(repo_ids):
        raise ValueError(f"Expected {len(repo_ids)} episode groups in {spec!r}")
    result = {}
    for repo_id, group in zip(repo_ids, groups, strict=True):
        values = [int(value) for value in group.split(",") if value.strip()]
        if not values:
            raise ValueError(f"No episodes selected for {repo_id}")
        result[repo_id] = values
    return result


def make_scheduler(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, schedule)


def checkpoint_metadata(config: TrainConfig, train_dataset) -> dict:
    return {
        "repo_ids": list(config.repo_ids),
        "image_size": [config.image_size, config.image_size],
        "state_key": STATE_KEY,
        "action_key": ACTION_KEY,
        "camera_keys": ["observation.images.top", "observation.images.wrist"],
        "state_names": list(ACTION_NAMES),
        "action_names": list(ACTION_NAMES),
        "fps": train_dataset.fps,
    }


def save_state(path: Path, model, optimizer, scheduler, scaler, step, best_val, config, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "best_val": best_val,
            "model": model.checkpoint_payload(metadata),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "train_config": asdict(config),
        },
        path,
    )


def load_state(path: Path, model, optimizer, scheduler, scaler) -> tuple[int, float]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"]["state_dict"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload.get("scaler", {}))
    return int(payload["step"]), float(payload.get("best_val", float("inf")))


def train(config: TrainConfig) -> Path:
    set_seed(config.seed)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "train_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    train_episodes = parse_episode_groups(config.train_episodes, config.repo_ids)
    val_episodes = parse_episode_groups(config.val_episodes, config.repo_ids)
    shared = dict(
        repo_ids=config.repo_ids,
        root=config.data_root,
        action_horizon=config.action_horizon,
        image_size=(config.image_size, config.image_size),
    )
    train_dataset = GoalConditionedMultiRepoDataset(episodes=train_episodes, **shared)
    val_dataset = GoalConditionedMultiRepoDataset(episodes=val_episodes, **shared)
    train_loader = make_dataloader(
        train_dataset, config.batch_size, balanced=True, num_workers=config.num_workers, seed=config.seed
    )
    val_loader = make_dataloader(
        val_dataset, config.batch_size, balanced=False, num_workers=config.num_workers, seed=config.seed
    )
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    policy_config = GoalPolicyConfig(
        action_horizon=config.action_horizon,
        hidden_dim=config.hidden_dim,
        text_dim=config.text_dim,
        dropout=config.dropout,
    )
    model = GoalConditionedActionChunkPolicy(
        policy_config, **default_normalization(train_dataset)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = make_scheduler(optimizer, config.warmup_steps, config.steps)
    amp_enabled = device.type == "cuda" and config.amp != "off"
    amp_dtype = torch.bfloat16 if config.amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    metadata = checkpoint_metadata(config, train_dataset)
    step, best_val = 0, float("inf")
    resume_path = Path(config.resume) if config.resume else output / "last_training.pt"
    if resume_path.exists():
        step, best_val = load_state(resume_path, model, optimizer, scheduler, scaler)
        print(f"Resumed from {resume_path} at step {step}")

    iterator = iter(train_loader)
    metrics_path = output / "metrics.jsonl"
    running_loss = 0.0
    running_count = 0
    started = time.perf_counter()
    while step < config.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            loss = model(batch)["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        running_loss += float(loss.detach())
        running_count += 1

        if step % config.log_every == 0 or step == 1:
            record = {
                "step": step,
                "train_loss": running_loss / running_count,
                "learning_rate": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "steps_per_second": step / max(time.perf_counter() - started, 1e-6),
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record))
            running_loss = 0.0
            running_count = 0

        if step % config.eval_every == 0 or step == config.steps:
            val_loss = evaluate(model, val_loader, device, max_batches=config.eval_batches)
            record = {"step": step, "val_loss": val_loss}
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record))
            if val_loss < best_val:
                best_val = val_loss
                model.save_checkpoint(output, metadata=metadata, filename="best_model.pt")

        if step % config.save_every == 0 or step == config.steps:
            model.save_checkpoint(output, metadata=metadata, filename="last_model.pt")
            save_state(
                output / "last_training.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                step,
                best_val,
                config,
                metadata,
            )
            snapshot = output / "checkpoints" / f"step_{step:06d}.pt"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output / "last_model.pt", snapshot)
    return output / "best_model.pt"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in TrainConfig.__dataclass_fields__.values():
        name = f"--{field.name.replace('_', '-')}"
        default = field.default
        if field.name == "repo_ids":
            parser.add_argument(name, nargs="+", default=list(default))
        else:
            parser.add_argument(name, type=type(default) if default is not None else str, default=default)
    values = vars(parser.parse_args())
    values["repo_ids"] = tuple(values["repo_ids"])
    return TrainConfig(**values)


if __name__ == "__main__":
    print(f"Best checkpoint: {train(parse_args())}")
