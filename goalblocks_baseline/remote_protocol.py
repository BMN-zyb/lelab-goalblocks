"""Small, dependency-free protocol shared by the inference server and robot client."""

from __future__ import annotations

import io
import socket
import struct

import numpy as np

MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def encode_arrays(**arrays) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def decode_arrays(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("Connection closed while receiving a message")
        chunks.extend(chunk)
    return bytes(chunks)


def send_message(connection: socket.socket, payload: bytes) -> None:
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"Message is too large: {len(payload)} bytes")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def receive_message(connection: socket.socket) -> bytes:
    size = struct.unpack("!I", _receive_exact(connection, 4))[0]
    if size > MAX_MESSAGE_BYTES:
        raise ValueError(f"Refusing oversized message: {size} bytes")
    return _receive_exact(connection, size)
