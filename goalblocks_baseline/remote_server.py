"""GPU inference server for the GoalBlocks baseline."""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .dataset import STATE_KEY, TOP_KEY, WRIST_KEY, encode_task, prepare_rgb_image
from .model import GoalConditionedActionChunkPolicy
from .remote_protocol import decode_arrays, encode_arrays, receive_message, send_message

DEFAULT_TASK = (
    "Use green, red, blue, and orange blocks to place the red block on top of the blue block, "
    "forming a step-like structure."
)


class InferenceEngine:
    def __init__(self, checkpoint: str, goal_image: str, task: str, device: str) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = GoalConditionedActionChunkPolicy.load_checkpoint(checkpoint, str(self.device)).to(self.device)
        self.model.eval()
        metadata = self.model.load_metadata(checkpoint)
        self.image_size = tuple(metadata.get("image_size", (128, 128)))
        with Image.open(goal_image) as image:
            self.goal = prepare_rgb_image(image, self.image_size).unsqueeze(0).to(self.device)
        self.tokens = encode_task(task).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict(self, request: dict[str, np.ndarray]) -> np.ndarray:
        batch = {
            TOP_KEY: prepare_rgb_image(torch.from_numpy(request["top"]), self.image_size)
            .unsqueeze(0)
            .to(self.device),
            WRIST_KEY: prepare_rgb_image(torch.from_numpy(request["wrist"]), self.image_size)
            .unsqueeze(0)
            .to(self.device),
            STATE_KEY: torch.from_numpy(request["state"].astype(np.float32, copy=False))
            .reshape(1, -1)
            .to(self.device),
            "goal.image": self.goal,
            "task_tokens": self.tokens,
        }
        return self.model.predict_action_chunk(batch)[0].float().cpu().numpy()


def serve(args) -> None:
    engine = InferenceEngine(args.checkpoint, args.goal_image, args.task, args.device)
    with socket.create_server((args.host, args.port), reuse_port=False) as server:
        print(f"GoalBlocks inference server listening on {args.host}:{args.port}", flush=True)
        while True:
            connection, address = server.accept()
            print(f"Client connected: {address}", flush=True)
            with connection:
                connection.settimeout(args.timeout)
                try:
                    while True:
                        request = decode_arrays(receive_message(connection))
                        started = time.perf_counter()
                        actions = engine.predict(request)
                        latency_ms = (time.perf_counter() - started) * 1000
                        send_message(
                            connection,
                            encode_arrays(actions=actions, latency_ms=np.array(latency_ms, np.float32)),
                        )
                except (ConnectionError, TimeoutError, socket.timeout):
                    print(f"Client disconnected: {address}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--goal-image", required=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    serve(parse_args())
