"""WSL client that sends SO-101 observations to a remote GoalBlocks server."""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from .remote_protocol import decode_arrays, encode_arrays, receive_message, send_message

ACTION_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def request_actions(connection: socket.socket, observation: dict) -> tuple[np.ndarray, float]:
    state = np.asarray([observation[name] for name in ACTION_NAMES], dtype=np.float32)
    send_message(
        connection,
        encode_arrays(
            top=np.asarray(observation["top"], dtype=np.uint8),
            wrist=np.asarray(observation["wrist"], dtype=np.uint8),
            state=state,
        ),
    )
    response = decode_arrays(receive_message(connection))
    return response["actions"].astype(np.float32), float(response["latency_ms"])


def run(args) -> None:
    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=Path(args.top_camera), fps=args.camera_fps, width=640, height=480, fourcc=args.fourcc
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=Path(args.wrist_camera), fps=args.camera_fps, width=640, height=480, fourcc=args.fourcc
        ),
    }
    config = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=cameras,
        use_degrees=True,
        num_read_retries=2,
        max_relative_target=args.max_relative_target,
        disable_torque_on_disconnect=True,
    )
    robot = SO101Follower(config)
    mode = "EXECUTE" if args.execute else "DRY RUN (actions are not sent)"
    print(f"Mode: {mode}")
    connected = False
    try:
        robot.connect()
        connected = True
        with socket.create_connection((args.server_host, args.server_port), timeout=args.timeout) as connection:
            connection.settimeout(args.timeout)
            period = 1.0 / args.control_fps
            chunks_completed = 0
            while args.max_chunks == 0 or chunks_completed < args.max_chunks:
                observation = robot.get_observation()
                actions, latency_ms = request_actions(connection, observation)
                count = min(args.actions_per_chunk, len(actions))
                print(f"Received {len(actions)} actions in {latency_ms:.1f} ms; using {count}")
                for values in actions[:count]:
                    started = time.perf_counter()
                    if args.execute:
                        robot.send_action(dict(zip(ACTION_NAMES, values.tolist(), strict=True)))
                    remaining = period - (time.perf_counter() - started)
                    if remaining > 0:
                        time.sleep(remaining)
                chunks_completed += 1
    except KeyboardInterrupt:
        print("Stopping safely after Ctrl+C")
    finally:
        if connected or robot.bus.is_connected:
            robot.disconnect()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8080)
    parser.add_argument("--robot-port", required=True)
    parser.add_argument("--robot-id", default="my_follower")
    parser.add_argument("--top-camera", default="/dev/video0")
    parser.add_argument("--wrist-camera", default="/dev/video2")
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--control-fps", type=float, default=15.0)
    parser.add_argument("--actions-per-chunk", type=int, default=4)
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument(
        "--max-chunks", type=int, default=0, help="Stop after this many requests; 0 runs until Ctrl+C."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send predicted actions. Without this flag the client is observation-only.",
    )
    args = parser.parse_args()
    if args.actions_per_chunk < 1:
        parser.error("--actions-per-chunk must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
