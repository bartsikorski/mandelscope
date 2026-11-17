# Large reference-orbit transfers

Measured on 2026-11-17 against PR #7, merge commit
`f6f8420ac65fe06141d607e8fc7a2b3d2c5afc1d`.

## Result and scope

Replace the result `multiprocessing.Queue` with a one-way pipe using pickle
protocol 5 out-of-band NumPy buffers, transferred in **1 MiB chunks**. Keep the
receiving thread and the UI's local inbox. No renderer or numerical algorithms
change. The production transport is 62 lines and needs no new dependency.

For a 3,000,000-iteration reference and its BLA table, the median worst
main-thread heartbeat gap fell from **114.5 ms to 3.0 ms**. Across seven measured
transfers the worst gaps were **281.1 ms and 9.0 ms**. This reduces the transfer
GIL stall; it does **not** remove the separate pause when the UI packs and uploads
a new reference to OpenGL.

These are measurements on this session's **Linux cloud host**, with 9 visible
CPUs (AMD EPYC 9V74), Python 3.12.14 and NumPy 2.5.4. OpenGL is **Mesa llvmpipe
(LLVM 19.1.7, 256 bits)**. No RTX 5060 Ti, Windows driver, or physical GPU was
available. PR #7's reported 74 ms frame was on a different VM; it is not a
reproduced baseline here. Heartbeat gaps below are not rendered GUI frames.

## Reproduce

```sh
uv sync --all-extras
uv run python tools/bench_orbit_transfer.py --gpu --shared-memory \
    --repeats 7 --output orbit-transfer.json
uv run pytest -q
uv run ruff check .
```

`--gpu` measures OpenGL packing/upload independently. `--shared-memory` adds an
experimental comparison; it does not change application behavior. Omit either
option when unavailable. On an RTX machine, check the emitted renderer string
before treating GPU results as hardware measurements. Use `--iterations 3000000`
to run only the large case.

The [raw measurements](benchmarks/orbit-transfer-linux.json) contain every measured
trial and the environment. One warmup per transport is excluded. Trial order
rotates between queue, pipe and shared memory. The host is noisy; medians and
observed worst cases are reported, without a hard real-time claim.

## Separate IPC, deserialization, and GIL contention

The spawned producer computes the same nonescaping `c = i` reference at 128 bits
and default second-order BLA table (`log2_cmax=-60`, `log2_eps=-24`). Computation
finishes **before** transfer timing begins. For 3M iterations the queue pickle is
143,439,215 bytes (143.4 MB / 136.8 MiB); protocol 5 separates 143,437,415 bytes of
numerical buffers from 1,289 bytes of object metadata.

The parent receives on a background thread. Its main thread requests a wakeup
every 1 ms and records elapsed gaps. An event releases the receiver only after
main-thread timing starts, so immediate GIL holds cannot escape measurement.
The idle heartbeat maximum was 1.28 ms.

* **Queue:** the exact `multiprocessing.Queue.put/get` transport used by PR #7.
  Instrumentation times `_recv_bytes` and `ForkingPickler.loads` separately in
  the receiving process. Total time includes producer serialization and pipe
  transfer; the read timer includes waiting for those stages.
* **Pickle only:** deserialize an already resident copy of the queue payload on
  the receiving thread. Removes pipe IO but retains GIL contention.
* **Pipe:** the actual new `ResultSender`/`ResultReceiver`. The JSON records time
  inside connection reads separately; the remainder covers allocation, framing
  and rebuilding the small metadata pickle.
* **Shared memory:** a preallocated reusable arena owned by the parent, copies
  into it in the producer, a metadata message, then a NumPy copy into ordinary
  receiver-owned memory. Allocation/ownership setup is outside timing, making
  this a favorable prototype comparison rather than a production implementation.

All dataclass fields and array dtypes, shapes and raw bytes are fingerprinted
outside timing. Every result from all four modes matched, for both sizes.

| Iterations | Transport | Median complete transfer | Median of per-trial worst heartbeat gaps | Worst gap in seven trials |
| ---: | --- | ---: | ---: | ---: |
| 300,000 | PR #7 queue | 62.6 ms | 7.21 ms | 58.96 ms |
| 300,000 | New pipe | 50.4 ms | 1.83 ms | 8.91 ms |
| 300,000 | Shared-memory prototype | 8.2 ms | 1.21 ms | 3.09 ms |
| 3,000,000 | PR #7 queue | 666.3 ms | 114.48 ms | 281.15 ms |
| 3,000,000 | New pipe | 146.7 ms | 2.98 ms | 9.01 ms |
| 3,000,000 | Shared-memory prototype | 60.2 ms | 1.66 ms | 11.97 ms |

The 3M queue unpickle stage alone had a median of **67.9 ms**. Isolated
`pickle.loads` had a median of **18.7 ms**, with matching heartbeat gaps and an
observed worst of **154.9 ms**. Allocation/cache conditions affect these numbers
substantially. Queue receive gaps can also exceed deserialization time: the
large pipe message must be accumulated in a Python buffer. Moving only
`Queue.get()` onto another thread does not bound these GIL-held operations.
Medians of separate stages need not sum to the median total.

The replacement bounds pipe buffer allocation/copy operations to 1 MiB and uses
`np.empty` for destinations, avoiding a large `bytearray` zero-fill. The small
pickle reconstructs views onto independently owned NumPy allocations. A future
result containing large noncontiguous or object-dtype arrays could fall back to
an in-band NumPy pickle; current reference/BLA buffers are contiguous numeric
arrays. Reporting metadata size makes this assumption visible.

## Shared-memory decision

Shared memory **materially improves total delivery time** in this prototype:
60.2 ms versus 146.7 ms for the new pipe at 3M iterations. It lowers the median
worst heartbeat by another 1.3 ms, close to idle. For the transfer hitch, the
bounded pipe already removes the large GIL pause.

A production shared-memory implementation also needs arena sizing and fallback,
acknowledgment before reuse, and ownership/cleanup across partial results, stale
messages, cancellation and worker death. Keeping views in the arena changes
array lifetimes; copying them out avoids that but retains a full memory copy.
Linux shared-memory capacity and Windows handle lifetime differ. The pipe keeps
ordinary-array lifetime without a memory pool, manager process, unlink protocol
or platform branches. Revisit shared memory if complete transfer latency becomes
a measured bottleneck after GPU upload work is addressed.

## Separate GPU cost, unchanged by this patch

The benchmark calls the actual `GPURef` and `GPUBla` constructors on the main
thread, times `ctx.buffer` calls separately from packing, and calls `ctx.finish`
afterwards. These are llvmpipe host/driver costs, **not PCIe bandwidth measurements**.
No rendering or worker transfer runs concurrently. Each precision has one warmup
and three measured trials.

| 3M reference | Median total construction | Median host packing | Median buffer calls |
| --- | ---: | ---: | ---: |
| float32 | 75.4 ms | 33.5 ms | 41.9 ms |
| float64 | 116.0 ms | 46.8 ms | 69.9 ms |
| float-float | 298.7 ms | 211.7 ms | 87.1 ms |

`ctx.finish` added 0.01–0.02 ms. These costs remain; no claim of stall-free
rendering or sub-16-ms end-to-end frames follows from this transfer improvement.
Avoiding large parent-side pickle copies should also reduce GIL contention on
the RTX, but neither the gap nor upload speed can be predicted reliably here.
Run the benchmark there and pair it with a motion study for displayed frame timing.

## Correctness and lifecycle

* The single producer's `put` completes snapshots synchronously. Partial, final,
  progress, error and auxiliary messages preserve order. Request submission and
  job-id cancellation filtering are unchanged.
* The sender can wait for pipe backpressure while delivering a result.
  Cancellation is observed at the next existing computation check after that
  transfer. The dedicated receiver drains independently of UI polling.
* Cached MPFR state stays in the worker via `dataclasses.replace`, instead of
  temporarily clearing/restoring the cached object's state.
* The parent closes its writer after spawning. Worker death halfway through a
  result produces EOF, and only complete results reach the UI inbox.
* Shutdown cancels, requests exit and terminates if needed, now also joining a
  terminated child and closing IPC handles. Repeated `close` is harmless.
* Real spawn/IPC tests cover cancellation, partial prefixes, resumed cached state,
  BLA indices, Julia pairs and cached critical orbits, Burning Ship XY arrays,
  errors, ordered polling and shutdown. Low-level tests cover chunks, aliases,
  empty/Fortran/strided arrays, signed zeros/NaNs, snapshot independence and use
  after close.
* GPU tests compare every reference, mantissa, exponent, XY, BLA and BLAQ buffer
  with the old pickle transport for Mandelbrot, Burning Ship and Julia cases in
  all three precision modes.

The pre-change non-GUI suite passed **171 tests, 1 skipped**. Patched suite and
lint results are recorded in the pull request. Display/Xvfb was unavailable
locally, so the scripted GUI session must run in CI or on a desktop.
