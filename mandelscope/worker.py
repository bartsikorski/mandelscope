"""Background worker process for reference orbits, BLA tables and minibrot search.

gmpy2 loops hold the GIL, so they run in a separate (spawned) process; the UI
thread only polls a queue.  Jobs can be cancelled; stale results are dropped
by the caller using job ids.
"""
from __future__ import annotations

import itertools
import multiprocessing as mp
import queue
import time
import traceback


def _worker_main(req_q, res_q, cancel_val):
    # imports inside the child process
    from .reference import compute_reference
    from .bla import build_bla
    from .nucleus import find_minibrot
    from .mp import to_str

    cache = {}   # ref_id -> ReferenceOrbit (only the latest few)
    stopped = [None]   # the orbit of the last cancelled reference job (resumable)
    julia_b = {}       # (c re, c im) -> critical orbit of the current Julia set

    def resumable_for(job):
        """A cached or cancelled orbit of the same point to continue instead of starting over
        (zooming deeper or raising the iteration limit asks for the same point again)."""
        from gmpy2 import mpfr
        best = None
        for r in list(cache.values()) + ([stopped[0]] if stopped[0] is not None else []):
            if r.state is None or r.prec < job["prec"] or r.length > job["max_iter"]:
                continue
            if getattr(r, "formula", "mandelbrot") != job.get("formula", "mandelbrot"):
                continue
            p = r.prec
            try:
                same = mpfr(job["re"], p) == mpfr(r.re, p) and mpfr(job["im"], p) == mpfr(r.im, p)
            except Exception:
                same = False
            if same and (best is None or r.length > best.length):
                best = r
        return best
    while True:
        job = req_q.get()
        if job is None:
            break
        jid = job["id"]
        kind = job["kind"]

        def cancelled():
            return cancel_val.value >= jid

        last = [0.0]

        def progress(f):
            now = time.perf_counter()
            if now - last[0] > 0.1:
                last[0] = now
                res_q.put({"id": jid, "kind": "progress", "value": float(f)})

        try:
            if kind == "reference":
                resume = cache.get(job.get("resume_id")) if job.get("resume_id") else None
                if resume is None:
                    resume = resumable_for(job)
                prec = max(job["prec"], resume.prec) if resume is not None else job["prec"]

                def partial(r, job=job, jid=jid):
                    # the orbit so far: the explorer draws with it (iterations capped at its
                    # length) while the rest is computed
                    if cancelled():
                        return
                    r.ref_id = jid
                    b = None
                    if job.get("log2_cmax") is not None:
                        b = build_bla(r.zm, r.ze, job["log2_cmax"], job["log2_eps"], lev0=job.get("lev0"),
                                      max_iter=r.max_iter)
                        b.ref_id = jid
                    res_q.put({"id": jid, "kind": "partial", "ref": r, "bla": b})

                ref = compute_reference(job["re"], job["im"], prec, job["max_iter"],
                                        progress=progress, cancelled=cancelled, resume=resume,
                                        partial=partial if job.get("stream") else None,
                                        partial_after=float(job.get("partial_after", 0.5)),
                                        keep_cancelled=True, formula=job.get("formula", "mandelbrot"))
                if ref is None or ref.cancelled:
                    if ref is not None:
                        stopped[0] = ref
                    res_q.put({"id": jid, "kind": "cancelled"})
                    continue
                ref.resumed_from = resume.length if resume is not None else 0
                ref.ref_id = jid
                cache[jid] = ref
                while len(cache) > 3:
                    cache.pop(next(iter(cache)))
                bla = None
                if job.get("log2_cmax") is not None:
                    bla = build_bla(ref.zm, ref.ze, job["log2_cmax"], job["log2_eps"],
                                    lev0=job.get("lev0"), max_iter=ref.max_iter)
                    bla.ref_id = jid
                state = ref.state
                ref.state = None          # mpc objects stay in the worker
                res_q.put({"id": jid, "kind": "reference", "ref": ref, "bla": bla,
                           "resumable": state is not None})
                ref.state = state
            elif kind == "julia_reference":
                from .offline import compute_julia_references
                key = (job["c_re"], job["c_im"])
                rb = julia_b.get(key)
                if rb is not None and (rb.prec < job["prec"] or not rb.complete_for(job["max_iter"])):
                    rb = None
                out = compute_julia_references(job, job["log2_eps"], ref_b=rb, progress=progress,
                                               cancelled=cancelled)
                if out is None:
                    res_q.put({"id": jid, "kind": "cancelled"})
                    continue
                ref_a, ref_b, bla_a, bla_b = out
                julia_b.clear()
                julia_b[key] = ref_b
                for r in (ref_a, ref_b):
                    r.ref_id = jid
                from dataclasses import replace as _rep
                res_q.put({"id": jid, "kind": "julia_reference", "ref_a": _rep(ref_a, state=None),
                           "ref_b": _rep(ref_b, state=None),
                           "bla_a": bla_a, "bla_b": bla_b, "log2_eps": job["log2_eps"]})
            elif kind == "bla":
                ref = cache.get(job["ref_id"])
                if ref is None:
                    res_q.put({"id": jid, "kind": "error", "error": "reference not cached"})
                    continue
                bla = build_bla(ref.zm, ref.ze, job["log2_cmax"], job["log2_eps"], lev0=job.get("lev0"),
                                max_iter=ref.max_iter)
                bla.ref_id = job["ref_id"]
                res_q.put({"id": jid, "kind": "bla", "bla": bla})
            elif kind == "minibrot":
                r = find_minibrot(job["re"], job["im"], job["log2_radius"],
                                  max_period=job.get("max_period", 1 << 20),
                                  cancelled=cancelled, progress=progress,
                                  formula=job.get("formula", "mandelbrot"))
                if r is None:
                    res_q.put({"id": jid, "kind": "minibrot", "result": None})
                else:
                    out = dict(r)
                    out["re"] = to_str(r["re"], r["prec"])
                    out["im"] = to_str(r["im"], r["prec"])
                    res_q.put({"id": jid, "kind": "minibrot", "result": out})
            else:
                res_q.put({"id": jid, "kind": "error", "error": f"unknown job {kind}"})
        except Exception:
            res_q.put({"id": jid, "kind": "error", "error": traceback.format_exc()})


class Worker:
    """Owns one background process; submit() returns a job id, poll() results."""

    def __init__(self):
        ctx = mp.get_context("spawn")
        self.req_q = ctx.Queue()
        self.res_q = ctx.Queue()
        self.cancel_val = ctx.Value("q", 0)
        self.proc = ctx.Process(target=_worker_main, args=(self.req_q, self.res_q, self.cancel_val),
                                daemon=True, name="mandelscope-worker")
        self.proc.start()
        self._ids = itertools.count(1)
        self.busy_job = None

    def submit(self, kind: str, cancel_previous: bool = True, **kw) -> int:
        jid = next(self._ids)
        if cancel_previous:
            with self.cancel_val.get_lock():
                self.cancel_val.value = jid - 1
        self.req_q.put({"id": jid, "kind": kind, **kw})
        return jid

    def cancel_all(self):
        with self.cancel_val.get_lock():
            self.cancel_val.value = 1 << 62

    def poll(self) -> list:
        out = []
        while True:
            try:
                out.append(self.res_q.get_nowait())
            except queue.Empty:
                break
        return out

    def wait(self, jid: int, timeout: float = 600.0):
        """Blocking wait for a given job (for scripts/tests)."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            try:
                msg = self.res_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if msg["id"] == jid and msg["kind"] != "progress":
                return msg
        raise TimeoutError(f"job {jid} timed out")

    def close(self):
        try:
            self.cancel_all()
            self.req_q.put(None)
            self.proc.join(timeout=2)
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.terminate()
