"""Inspect GoalBlocks repositories without merging them."""

from __future__ import annotations

import argparse
import json

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def inspect_repository(repo_id: str, download_videos: bool) -> dict:
    dataset = LeRobotDataset(repo_id, download_videos=download_videos)
    info_path = dataset.root / "meta" / "info.json"
    raw_info = json.loads(info_path.read_text(encoding="utf-8"))
    summary = {
        "repo_id": repo_id,
        "root": str(dataset.root),
        "frames": len(dataset),
        "episodes": dataset.num_episodes,
        "fps": dataset.fps,
        "features": sorted(dataset.features),
        "camera_keys": dataset.meta.camera_keys,
        "goal": raw_info.get("goal"),
    }
    if download_videos and len(dataset):
        sample = dataset[0]
        summary["sample"] = {
            key: list(value.shape) if hasattr(value, "shape") else type(value).__name__
            for key, value in sample.items()
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_ids", nargs="+")
    parser.add_argument("--download-videos", action="store_true")
    args = parser.parse_args()
    for repo_id in args.repo_ids:
        print(json.dumps(inspect_repository(repo_id, args.download_videos), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
