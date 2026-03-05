"""Wire protocol for ScholarAgent socket communication.

Protocol: 4-byte big-endian length prefix + UTF-8 JSON payload.
Used for communication between LMHandler and environment subprocesses.
"""

import json
import socket
import struct


def socket_send(conn: socket.socket, data: dict) -> None:
    """Send JSON data with 4-byte length prefix."""
    payload = json.dumps(data).encode("utf-8")
    header = struct.pack("!I", len(payload))
    conn.sendall(header + payload)


def socket_recv(conn: socket.socket) -> dict:
    """Receive JSON data with 4-byte length prefix.

    Returns empty dict if the connection is cleanly closed before any
    header bytes arrive.  Raises ConnectionError if it closes mid-message.
    """
    header = _recv_exact(conn, 4)
    length = struct.unpack("!I", header)[0]
    payload = _recv_exact(conn, length)
    return json.loads(payload.decode("utf-8"))


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    """Read exactly *n* bytes from *conn*."""
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed")
        data += chunk
    return data


def socket_request(address: tuple[str, int], data: dict, timeout: int = 300) -> dict:
    """Open a connection, send *data*, wait for a response, then close."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(address)
        socket_send(s, data)
        return socket_recv(s)
