"""Goal-conditioned multi-repository loading for LeRobot v3 datasets."""

from __future__ import annotations

import bisect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

TOP_KEY = "observation.images.top"
WRIST_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


@dataclass(frozen=True)
class RepositoryGoal:
    repo_id: str
    image_paths: tuple[str, ...]
    task: str
    source: str


def encode_task(task: str, max_bytes: int = 256) -> torch.Tensor:
    """Encode UTF-8 text without requiring a pretrained tokenizer."""
    payload = task.encode("utf-8")[:max_bytes]
    encoded = torch.zeros(max_bytes, dtype=torch.long)
    if payload:
        encoded[: len(payload)] = torch.tensor([value + 1 for value in payload], dtype=torch.long)
    return encoded


def prepare_rgb_image(image: torch.Tensor | Image.Image, size: tuple[int, int]) -> torch.Tensor:
    """Convert image-like input to float CHW RGB and resize it."""
    if isinstance(image, Image.Image):
        tensor = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1)
    else:
        tensor = torch.as_tensor(image)
        if tensor.ndim != 3:
            raise ValueError(f"Expected a 3D image tensor, got shape {tuple(tensor.shape)}")
        if tensor.shape[0] not in (1, 3, 4) and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        tensor = tensor[:3]
    tensor = tensor.to(dtype=torch.float32)
    if tensor.max().item() > 1.5:
        tensor = tensor / 255.0
    tensor = F.interpolate(tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0]
    return tensor.clamp_(0.0, 1.0)


def _discover_goal_paths(
    repo_id: str,
    local_root: Path,
    revision: str | None,
    token: str | bool | None,
) -> list[str]:
    local_goals = sorted(
        path.relative_to(local_root).as_posix()
        for path in (local_root / "goals").glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if local_goals:
        return local_goals
    files = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision, token=token)
    return sorted(
        path
        for path in files
        if path.startswith("goals/") and Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def _task_from_metadata(dataset: LeRobotDataset) -> str:
    tasks = dataset.meta.tasks
    if tasks is None or len(tasks.index) != 1:
        raise ValueError(
            f"{dataset.repo_id} must contain exactly one task when goal.task is missing; "
            f"found {0 if tasks is None else len(tasks.index)}"
        )
    return str(tasks.index[0])


def _load_repository_goal(
    dataset: LeRobotDataset,
    revision: str | None,
    token: str | bool | None,
    goal_image_overrides: Mapping[str, Sequence[str]] | None,
    task_overrides: Mapping[str, str] | None,
    image_size: tuple[int, int],
) -> tuple[RepositoryGoal, torch.Tensor]:
    raw_info = json.loads((dataset.root / "meta" / "info.json").read_text(encoding="utf-8"))
    goal_info = raw_info.get("goal") or {}
    override_paths = (goal_image_overrides or {}).get(dataset.repo_id)
    image_paths = list(override_paths or goal_info.get("goal_images") or [])
    source = "info.json"
    if not image_paths:
        image_paths = _discover_goal_paths(dataset.repo_id, dataset.root, revision, token)
        source = "auto-discovered"
    if not image_paths:
        raise ValueError(f"No goal images found for {dataset.repo_id}")
    if source == "auto-discovered" and len(image_paths) != 1:
        raise ValueError(
            f"{dataset.repo_id} has no goal metadata and {len(image_paths)} candidate images. "
            "Pass goal_image_overrides to choose explicitly."
        )

    task = (task_overrides or {}).get(dataset.repo_id) or goal_info.get("task") or _task_from_metadata(dataset)
    tensors = []
    for image_path in image_paths:
        local_candidate = dataset.root / image_path
        local_path = local_candidate if local_candidate.exists() else hf_hub_download(
            dataset.repo_id, image_path, repo_type="dataset", revision=revision, token=token
        )
        with Image.open(local_path) as image:
            tensors.append(prepare_rgb_image(image, image_size))
    goal_tensor = torch.stack(tensors).mean(dim=0)
    return RepositoryGoal(dataset.repo_id, tuple(image_paths), task, source), goal_tensor


class GoalConditionedMultiRepoDataset(Dataset):
    """Concatenate LeRobot repositories while preserving per-repository goals."""

    required_features = {TOP_KEY, WRIST_KEY, STATE_KEY, ACTION_KEY}

    def __init__(
        self,
        repo_ids: Sequence[str],
        *,
        episodes: Mapping[str, Sequence[int]] | None = None,
        root: str | Path | None = None,
        action_horizon: int = 16,
        image_size: tuple[int, int] = (128, 128),
        revision: str | None = None,
        video_backend: str | None = "pyav",
        token: str | bool | None = None,
        goal_image_overrides: Mapping[str, Sequence[str]] | None = None,
        task_overrides: Mapping[str, str] | None = None,
    ) -> None:
        if not repo_ids:
            raise ValueError("repo_ids cannot be empty")
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")

        self.repo_ids = list(repo_ids)
        self.image_size = image_size
        self.action_horizon = action_horizon
        self.datasets: list[LeRobotDataset] = []
        self.goals: list[RepositoryGoal] = []
        self.goal_tensors: list[torch.Tensor] = []

        root_path = Path(root) if root is not None else None
        for repo_id in self.repo_ids:
            selected_episodes = list(episodes[repo_id]) if episodes and repo_id in episodes else None
            dataset_root = root_path / repo_id if root_path is not None else None
            metadata_probe = LeRobotDataset(
                repo_id,
                root=dataset_root,
                episodes=selected_episodes,
                revision=revision,
                download_videos=False,
                token=token,
            )
            missing = self.required_features.difference(metadata_probe.features)
            if missing:
                raise ValueError(f"{repo_id} is missing required features: {sorted(missing)}")
            if metadata_probe.fps <= 0:
                raise ValueError(f"{repo_id} has invalid fps={metadata_probe.fps}")
            delta_timestamps = {
                ACTION_KEY: [step / metadata_probe.fps for step in range(action_horizon)]
            }
            dataset = LeRobotDataset(
                repo_id,
                root=dataset_root,
                episodes=selected_episodes,
                delta_timestamps=delta_timestamps,
                revision=revision,
                download_videos=True,
                video_backend=video_backend,
                token=token,
            )
            goal, goal_tensor = _load_repository_goal(
                dataset,
                revision,
                token,
                goal_image_overrides,
                task_overrides,
                image_size,
            )
            self.datasets.append(dataset)
            self.goals.append(goal)
            self.goal_tensors.append(goal_tensor)

        first_fps = self.datasets[0].fps
        if any(dataset.fps != first_fps for dataset in self.datasets):
            raise ValueError("All repositories must use the same fps")
        self.fps = first_fps
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.cumulative_lengths: list[int] = []
        running_total = 0
        for length in self.lengths:
            running_total += length
            self.cumulative_lengths.append(running_total)
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self.datasets])

    def __len__(self) -> int:
        return self.cumulative_lengths[-1]

    def _resolve_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_index = bisect.bisect_right(self.cumulative_lengths, index)
        previous = 0 if dataset_index == 0 else self.cumulative_lengths[dataset_index - 1]
        return dataset_index, index - previous

    def __getitem__(self, index: int) -> dict:
        dataset_index, local_index = self._resolve_index(index)
        item = self.datasets[dataset_index][local_index]
        action = torch.as_tensor(item[ACTION_KEY], dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        action_is_pad = torch.as_tensor(
            item.get(f"{ACTION_KEY}_is_pad", torch.zeros(action.shape[0], dtype=torch.bool)),
            dtype=torch.bool,
        )
        task = str(item.get("task") or self.goals[dataset_index].task)
        return {
            TOP_KEY: prepare_rgb_image(item[TOP_KEY], self.image_size),
            WRIST_KEY: prepare_rgb_image(item[WRIST_KEY], self.image_size),
            STATE_KEY: torch.as_tensor(item[STATE_KEY], dtype=torch.float32),
            ACTION_KEY: action,
            "action_is_pad": action_is_pad,
            "goal.image": self.goal_tensors[dataset_index].clone(),
            "task": task,
            "task_tokens": encode_task(task),
            "dataset_index": torch.tensor(dataset_index, dtype=torch.long),
            "episode_index": torch.as_tensor(item["episode_index"], dtype=torch.long),
            "repo_id": self.repo_ids[dataset_index],
        }

    def balanced_sampler(self, num_samples: int | None = None, seed: int = 0) -> WeightedRandomSampler:
        weights = []
        for length in self.lengths:
            weights.extend([1.0 / length] * length)
        generator = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.double),
            num_samples=num_samples or len(self),
            replacement=True,
            generator=generator,
        )


def goalblocks_collate(samples: Sequence[dict]) -> dict:
    batch: dict = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        batch[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
    return batch
