"""Probe a GoalBlocks inference server without opening robot hardware."""

from __future__ import annotations

import argparse
import socket

import numpy as np

from .remote_protocol import decode_arrays, encode_arrays, receive_message, send_message


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="172.17.176.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    with socket.create_connection((args.host, args.port), timeout=args.timeout) as connection:
        connection.settimeout(args.timeout)
        send_message(
            connection,
            encode_arrays(
                top=np.zeros((480, 640, 3), dtype=np.uint8),
                wrist=np.zeros((480, 640, 3), dtype=np.uint8),
                state=np.zeros(6, dtype=np.float32),
            ),
        )
        response = decode_arrays(receive_message(connection))
    print("actions shape:", response["actions"].shape)
    print("server latency ms:", float(response["latency_ms"]))
    print("first action:", response["actions"][0].tolist())


if __name__ == "__main__":
    main()
