"""Goal-image annotation and persistence for GoalBlocks datasets."""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, SecretStr, field_validator


QWEN_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
GOAL_PROMPT = """You are a robot-dataset annotation assistant.
Use the supplied goal images and optional notes to produce one concise, explicit,
and executable task description for an SO-101 robot arm. State which blocks are
used, the desired arrangement, and any color-matching requirement. Do not describe
irrelevant background details or infer facts unsupported by the inputs. Return one
natural-language task description."""


class GoalImage(BaseModel):
    name: str
    mime_type: str
    data: str

    @field_validator("mime_type")
    @classmethod
    def supported_image(cls, value: str) -> str:
        if value not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("goal image must be JPEG, PNG, or WebP")
        return value


class GoalDescriptionRequest(BaseModel):
    api_key: SecretStr
    images: list[GoalImage] = Field(min_length=1, max_length=4)
    notes: str = Field(default="", max_length=1000)


def generate_goal_description(request: GoalDescriptionRequest) -> str:
    """Call Qwen through DashScope's OpenAI-compatible endpoint.

    The key only lives in this request and is never persisted or logged.
    """
    content: list[dict[str, Any]] = [
        {"type": "text", "text": f"{GOAL_PROMPT}\nOptional notes: {request.notes or 'None'}"}
    ]
    for image in request.images:
        # Validate before forwarding, and cap each decoded image at 10 MiB.
        raw = base64.b64decode(image.data, validate=True)
        if len(raw) > 10 * 1024 * 1024:
            raise ValueError("each goal image must be 10 MiB or smaller")
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:{image.mime_type};base64,{image.data}"}}
        )

    body = json.dumps(
        {
            "model": "qwen-vl-plus",
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.2,
        }
    ).encode()
    http_request = urllib.request.Request(
        QWEN_ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {request.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=60) as response:
            result = json.load(response)
        description = result["choices"][0]["message"]["content"].strip()
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError, IndexError, TypeError) as exc:
        # Avoid returning provider response bodies, which can contain request details.
        raise RuntimeError("Qwen task-description generation failed; check the API key and network") from exc
    if not description:
        raise RuntimeError("Qwen returned an empty task description")
    return description


def save_goal_assets(dataset_root: Path, images: list[GoalImage], task: str) -> list[str]:
    """Save goal images and add their mapping to LeRobot's info.json."""
    goals_dir = dataset_root / "goals"
    goals_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    suffixes = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
    for index, image in enumerate(images, start=1):
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(image.name).stem).strip("._") or f"goal_{index}"
        filename = f"{index:02d}_{stem}{suffixes[image.mime_type]}"
        raw = base64.b64decode(image.data, validate=True)
        (goals_dir / filename).write_bytes(raw)
        paths.append(f"goals/{filename}")

    write_goal_metadata(dataset_root, paths, task)
    return paths


def write_goal_metadata(dataset_root: Path, paths: list[str], task: str) -> None:
    """Add goal metadata after LeRobot has finished updating info.json."""
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    task_language = "zh" if re.search(r"[\u3400-\u9fff]", task) else "en"
    info["goal"] = {
        "goal_images": paths,
        "task_prompt_source": "qwen-vl-generated-from-goal-image",
        "task_language": task_language,
        "task": task,
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
