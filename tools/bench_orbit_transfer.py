"""Separate result IPC, GIL stalls, and GPU packing/upload costs.

Run from a checkout: uv run python tools/bench_orbit_transfer.py --gpu --shared-memory \
    --output orbit-transfer.json
No rendering runs during IPC timings: a 1 ms main-thread heartbeat measures Python
responsiveness, not actual GUI frames. Queue mode reproduces PR #7's transport;
pipe mode uses the production replacement. Shared memory is a deliberately simple
prototype with a preallocated, reusable arena and a NumPy copy on receipt.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
import multiprocessing as mp
import multiprocessing.queues as mp_queues
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import pickle
import platform
import shutil
import statistics
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mandelscope.bla import build_bla, structure_for
from mandelscope.reference import compute_reference
from mandelscope.transfer import CHUNK_BYTES, ResultReceiver, ResultSender


def fingerprint(value):
    """All dataclass fields, array shapes/dtypes/raw bits; independent of pickle layout."""
    h = hashlib.sha256()

    def visit(v):
        if isinstance(v, np.ndarray):
            h.update(str((v.dtype, v.shape)).encode())
            h.update(v.tobytes())
        elif is_dataclass(v):
            for f in fields(v):
                h.update(f.name.encode())
                visit(getattr(v, f.name))
        elif isinstance(v, dict):
            for k in sorted(v):
                h.update(k.encode())
                visit(v[k])
        elif isinstance(v, (list, tuple)):
            for item in v:
                visit(item)
        else:
            h.update(repr(v).encode())

    visit(value)
    return h.hexdigest()


def _producer(q, writer, control, iterations):
    ref = compute_reference("0", "1", 128, iterations)
    bla = build_bla(ref.zm, ref.ze, -60., -24., structure=structure_for(ref))
    ref.state = None
    message = dict(id=1, kind="reference", ref=ref, bla=bla, resumable=True)
    buffers = []
    metadata = pickle.dumps(message, protocol=5, buffer_callback=buffers.append)
    size = sum(b.raw().nbytes for b in buffers)
    control.send(dict(array_bytes=size, metadata_bytes=len(metadata),
                      queue_pickle_bytes=len(mp_queues._ForkingPickler.dumps(message))))
    name = control.recv()
    arena = SharedMemory(name=name) if name else None
    sender = ResultSender(writer)
    try:
        while (mode := control.recv()) is not None:
            if mode == "queue":
                q.put(message)
            elif mode == "pipe":
                sender.put(message)
            elif mode == "shared":
                buffers = []
                metadata = pickle.dumps(message, protocol=5, buffer_callback=buffers.append)
                offset, lengths = 0, []
                for b in buffers:
                    view = b.raw()
                    arena.buf[offset:offset + view.nbytes] = view
                    lengths.append(view.nbytes)
                    offset += view.nbytes
                q.put((metadata, lengths))
    finally:
        if arena is not None:
            arena.close()
        sender.close()


def sample(action):
    """Start the timing before releasing the receiver, so a GIL stall cannot be missed."""
    start, done = threading.Event(), threading.Event()
    result, errors, elapsed = [], [], []

    def run():
        start.wait()
        t0 = time.perf_counter()
        try:
            result.append(action())
        except BaseException as exc:
            errors.append(exc)
        finally:
            elapsed.append(1000 * (time.perf_counter() - t0))
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    gaps = []
    previous = time.perf_counter()
    start.set()
    while not done.is_set():
        time.sleep(.001)
        now = time.perf_counter()
        gaps.append(1000 * (now - previous))
        previous = now
    thread.join()
    if errors:
        raise errors[0]
    return result[0], dict(total_ms=elapsed[0], heartbeat_max_ms=max(gaps, default=0.),
                           heartbeat_p99_ms=float(np.percentile(gaps, 99)) if gaps else 0.)


class TimedConnection:
    def __init__(self, connection, stages):
        self.connection, self.stages = connection, stages

    def poll(self, timeout):
        return self.connection.poll(timeout)

    def _call(self, method, *args):
        start = time.perf_counter()
        result = getattr(self.connection, method)(*args)
        self.stages["pipe_io_ms"] += 1000 * (time.perf_counter() - start)
        return result

    def recv(self):
        return self._call("recv")

    def recv_bytes_into(self, target):
        return self._call("recv_bytes_into", target)


def gpu_measure(message, repeats):
    from mandelscope.cli import make_headless_context
    from mandelscope.renderer import GPURef, GPUBla
    ctx = make_headless_context()
    rows = []

    class Uploads:
        upload_ms = 0.

        def buffer(self, data):
            start = time.perf_counter()
            buffer = ctx.buffer(data)
            self.upload_ms += 1000 * (time.perf_counter() - start)
            return buffer

    try:
        for precision in (False, True, 2):
            for repeat in range(repeats + 1):
                proxy = Uploads()
                ctx.finish()
                start = time.perf_counter()
                ref = GPURef(proxy, message["ref"], precision)
                bla = GPUBla(proxy, message["bla"], precision)
                create_ms = 1000 * (time.perf_counter() - start)
                start = time.perf_counter()
                ctx.finish()
                finish_ms = 1000 * (time.perf_counter() - start)
                if repeat:
                    rows.append(dict(precision=["f32", "f64", "ff"][int(precision)],
                                     create_ms=create_ms, buffer_calls_ms=proxy.upload_ms,
                                     packing_ms=create_ms - proxy.upload_ms, finish_ms=finish_ms,
                                     gpu_bytes=ref.nbytes + bla.nbytes))
                ref.release()
                bla.release()
        return dict(renderer=ctx.info["GL_RENDERER"], version=ctx.info["GL_VERSION"], trials=rows)
    finally:
        ctx.release()


def benchmark(iterations, repeats, shared, gpu):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    reader, writer = ctx.Pipe(duplex=False)
    control, child_control = ctx.Pipe()
    proc = ctx.Process(target=_producer, args=(q, writer, child_control, iterations))
    proc.start()
    writer.close()
    child_control.close()
    arena = None
    output = dict(iterations=iterations, trials=[])
    try:
        output.update(control.recv())
        enough = not Path("/dev/shm").exists() or shutil.disk_usage("/dev/shm").free > output["array_bytes"] * 1.2
        if shared and enough:
            try:
                arena = SharedMemory(create=True, size=output["array_bytes"])
            except OSError as exc:
                output["shared_memory_skipped"] = str(exc)
        elif shared:
            output["shared_memory_skipped"] = "insufficient /dev/shm capacity"
        control.send(arena.name if arena is not None else None)
        modes = ["queue", "pipe"] + (["shared"] if arena is not None else [])
        expected, payload, last = None, None, None
        for repeat in range(repeats + 1):
            # First round warms imports/allocators. Rotate order of all measured rounds.
            order = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
            for mode in order + (["pickle"] if payload is not None else []):
                stages = dict(queue_read_ms=0., queue_decode_ms=0., pipe_io_ms=0., shared_copy_ms=0.)
                original_read, original_loads = q._recv_bytes, mp_queues._ForkingPickler.loads

                def timed_read():
                    start = time.perf_counter()
                    data = original_read()
                    stages["queue_read_ms"] += 1000 * (time.perf_counter() - start)
                    return data

                def timed_loads(data):
                    start = time.perf_counter()
                    obj = original_loads(data)
                    stages["queue_decode_ms"] += 1000 * (time.perf_counter() - start)
                    return obj

                def receive():
                    if mode == "pickle":
                        return pickle.loads(payload)
                    control.send(mode)
                    if mode == "queue":
                        return q.get()
                    if mode == "pipe":
                        return ResultReceiver(TimedConnection(reader, stages)).get()
                    metadata, lengths = q.get()
                    start = time.perf_counter()
                    data = np.frombuffer(arena.buf, dtype=np.uint8, count=sum(lengths)).copy()
                    stages["shared_copy_ms"] = 1000 * (time.perf_counter() - start)
                    offset, views = 0, []
                    for length in lengths:
                        views.append(memoryview(data)[offset:offset + length])
                        offset += length
                    return pickle.loads(metadata, buffers=views)

                q._recv_bytes = timed_read
                mp_queues._ForkingPickler.loads = staticmethod(timed_loads)
                try:
                    last, row = sample(receive)
                finally:
                    q._recv_bytes = original_read
                    mp_queues._ForkingPickler.loads = staticmethod(original_loads)
                digest = fingerprint(last)               # outside all timing windows
                if expected is None:
                    expected = digest
                    payload = mp_queues._ForkingPickler.dumps(last)
                assert digest == expected, (mode, "payload differs")
                if repeat:
                    output["trials"].append(dict(mode=mode, repeat=repeat, **row, **stages))
                print(f"{iterations:,} {mode:6} {'warmup' if not repeat else repeat}: "
                      f"total {row['total_ms']:.1f} ms, heartbeat max {row['heartbeat_max_ms']:.2f} ms",
                      flush=True)
        output["payload_sha256"] = expected
        output["all_payloads_equal"] = True
        if gpu:
            output["gpu"] = gpu_measure(last, min(repeats, 3))
    finally:
        if proc.is_alive():
            control.send(None)
            proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
            proc.join()
        if arena is not None:
            arena.close()
            arena.unlink()
        reader.close()
        control.close()
        q.close()
        q.join_thread()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", nargs="+", type=int, default=[300_000, 3_000_000])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--shared-memory", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _, idle = sample(lambda: time.sleep(.1))
    result = dict(python=sys.version, numpy=np.__version__, platform=platform.platform(),
                  cpu_count=os.cpu_count(), chunk_bytes=CHUNK_BYTES, idle=idle,
                  note="1 ms heartbeat gaps are responsiveness proxies, not rendered UI frame times",
                  cases=[benchmark(n, args.repeats, args.shared_memory, args.gpu) for n in args.iterations])
    for case in result["cases"]:
        case["summary"] = {}
        for mode in sorted({row["mode"] for row in case["trials"]}):
            rows = [r for r in case["trials"] if r["mode"] == mode]
            case["summary"][mode] = {
                "median_total_ms": statistics.median(r["total_ms"] for r in rows),
                "median_heartbeat_max_ms": statistics.median(r["heartbeat_max_ms"] for r in rows),
                "worst_heartbeat_ms": max(r["heartbeat_max_ms"] for r in rows),
            }
        print(case["iterations"], json.dumps(case["summary"], indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
