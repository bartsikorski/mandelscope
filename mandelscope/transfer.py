"""Worker results: small pickle metadata, numerical buffers sent in bounded chunks.

Even on a receiving thread, Queue.get() unpickles large NumPy arrays while holding
the GIL. Protocol 5 keeps those arrays out of the pickle; bounded pipe reads keep
the remaining GIL-held allocations/copies short. The receiver owns ordinary NumPy
memory, so results can be retained, discarded or used after the worker closes.
"""
from __future__ import annotations

import pickle
import queue

import numpy as np


CHUNK_BYTES = 1 << 20


class ResultSender:
    """Single producer, synchronous put(): snapshots finish before computation resumes."""

    def __init__(self, connection):
        self.connection = connection

    def put(self, message):
        buffers = []
        metadata = pickle.dumps(message, protocol=5, buffer_callback=buffers.append)
        views = [buffer.raw() for buffer in buffers]
        self.connection.send((metadata, [view.nbytes for view in views]))
        for view in views:
            for offset in range(0, view.nbytes, CHUNK_BYTES):
                self.connection.send_bytes(view[offset:offset + CHUNK_BYTES])

    def close(self):
        self.connection.close()


class ResultReceiver:
    """Only used by the receiving thread; get() may wait for the whole message."""

    def __init__(self, connection):
        self.connection = connection

    def get(self, timeout=None):
        if timeout is not None and not self.connection.poll(timeout):
            raise queue.Empty
        metadata, lengths = self.connection.recv()
        buffers = []
        for length in lengths:
            # empty() avoids a large GIL-held zero-fill (bytearray), and gives each
            # reconstructed array aligned, independently owned storage.
            data = np.empty(length, dtype=np.uint8)
            view = memoryview(data)
            for offset in range(0, length, CHUNK_BYTES):
                target = view[offset:offset + CHUNK_BYTES]
                if self.connection.recv_bytes_into(target) != len(target):
                    raise ValueError("incomplete worker result buffer")
            buffers.append(data)
        return pickle.loads(metadata, buffers=buffers)

    def close(self):
        self.connection.close()
