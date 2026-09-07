"""A small goal-conditioned action chunk policy for pipeline validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .dataset import ACTION_KEY, STATE_KEY, TOP_KEY, WRIST_KEY


@dataclass
class GoalPolicyConfig:
    state_dim: int = 6
    action_dim: int = 6
    action_horizon: int = 16
    hidden_dim: int = 192
    text_dim: int = 96
    dropout: float = 0.1


class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)


class ByteTextEncoder(nn.Module):
    def __init__(self, text_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(257, text_dim, padding_idx=0)
        self.projection = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(0).unsqueeze(-1)
        embedded = self.embedding(tokens)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.projection(pooled)


class GoalConditionedActionChunkPolicy(nn.Module):
    """Fuse top, wrist, goal, task and state to predict future joint targets."""

    def __init__(
        self,
        config: GoalPolicyConfig | None = None,
        *,
        state_mean: torch.Tensor | None = None,
        state_std: torch.Tensor | None = None,
        action_mean: torch.Tensor | None = None,
        action_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or GoalPolicyConfig()
        hidden = self.config.hidden_dim
        self.image_encoder = ImageEncoder(hidden)
        self.top_role = nn.Linear(hidden, hidden)
        self.wrist_role = nn.Linear(hidden, hidden)
        self.goal_role = nn.Linear(hidden, hidden)
        self.text_encoder = ByteTextEncoder(self.config.text_dim, hidden)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.config.state_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.config.action_horizon * self.config.action_dim),
        )
        self.register_buffer("state_mean", self._stat(state_mean, self.config.state_dim, 0.0))
        self.register_buffer("state_std", self._stat(state_std, self.config.state_dim, 1.0).clamp_min(1e-6))
        self.register_buffer("action_mean", self._stat(action_mean, self.config.action_dim, 0.0))
        self.register_buffer("action_std", self._stat(action_std, self.config.action_dim, 1.0).clamp_min(1e-6))

    @staticmethod
    def _stat(value: torch.Tensor | None, size: int, default: float) -> torch.Tensor:
        if value is None:
            return torch.full((size,), default, dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32).reshape(-1)[:size]

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        top = self.top_role(self.image_encoder(batch[TOP_KEY]))
        wrist = self.wrist_role(self.image_encoder(batch[WRIST_KEY]))
        goal = self.goal_role(self.image_encoder(batch["goal.image"]))
        text = self.text_encoder(batch["task_tokens"])
        state = (batch[STATE_KEY] - self.state_mean) / self.state_std
        state = self.state_encoder(state)
        fused = torch.cat([top, wrist, goal, text, state], dim=-1)
        normalized_actions = self.fusion(fused).view(
            -1, self.config.action_horizon, self.config.action_dim
        )
        result = {"normalized_actions": normalized_actions}
        if ACTION_KEY in batch:
            target = (batch[ACTION_KEY] - self.action_mean) / self.action_std
            valid = ~batch.get(
                "action_is_pad",
                torch.zeros(target.shape[:2], dtype=torch.bool, device=target.device),
            )
            squared_error = (normalized_actions - target).square()
            denominator = valid.sum().clamp_min(1) * self.config.action_dim
            result["loss"] = (squared_error * valid.unsqueeze(-1)).sum() / denominator
        return result

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        self.eval()
        normalized = self(batch)["normalized_actions"]
        return normalized * self.action_std + self.action_mean

    def checkpoint_payload(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "format_version": 2,
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "metadata": metadata or {},
        }

    def save_checkpoint(
        self,
        directory: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        filename: str = "goalblocks_baseline.pt",
    ) -> Path:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / filename
        torch.save(self.checkpoint_payload(metadata), checkpoint)
        return checkpoint

    @classmethod
    def load_checkpoint(cls, checkpoint: str | Path, map_location: str = "cpu"):
        payload = torch.load(checkpoint, map_location=map_location, weights_only=True)
        model = cls(GoalPolicyConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        return model

    @staticmethod
    def load_metadata(checkpoint: str | Path) -> dict[str, Any]:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        return dict(payload.get("metadata") or {})
