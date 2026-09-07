"""
A background thread that heartbeats a run's lease while a slow operation
(a subprocess, typically) runs on the main thread, and kills that
subprocess the moment the heartbeat reports the lease is no longer ours.

Why this needs to exist at all: the main thread is blocked inside
subprocess.run(), doing nothing but waiting -- it cannot notice "I've
been superseded" until the subprocess finishes on its own, which might
be minutes away. This thread is the only thing actively polling
heartbeat() in real time, so it's the only thing positioned to react
the instant ownership is lost. If it only stopped looping quietly
instead of killing the process, the subprocess would become an orphan:
still running, still possibly writing to a shared build cache, no
longer authorized by the system's own bookkeeping to be doing so.
"""
from __future__ import annotations

import subprocess
import threading
import time

import psycopg

from core.claim import heartbeat

HEARTBEAT_INTERVAL_SECONDS = 30


class LeaseLostError(Exception):
    """
    Raised when a subprocess step discovers, after the fact, that its
    lease was lost mid-run. This is deliberately NOT a RuntimeError or
    any other generic exception: the worker loop must treat this
    completely differently from an ordinary failure. An ordinary
    failure means "this run needs to be retried, bump attempt, release
    it back" -- but this run no longer belongs to us, so calling
    release() on it would mean writing to a row another worker now
    owns. On this exception the worker loop must do nothing to the row
    at all and simply move on.
    """
    pass


class HeartbeatGuard:
    """
    Usage:
        proc = subprocess.Popen([...])
        with HeartbeatGuard(conn, run_id, worker_id, proc):
            proc.wait(timeout=600)

    While the `with` block is open, a background thread heartbeats every
    HEARTBEAT_INTERVAL_SECONDS. If heartbeat() ever returns False (lease
    lost), the guard kills `proc` immediately and stops.
    """

    def __init__(
        self,
        conn: psycopg.Connection,
        run_id: int,
        worker_id: str,
        proc: subprocess.Popen,
    ):
        self._conn = conn
        self._run_id = run_id
        self._worker_id = worker_id
        self._proc = proc
        self._stop_event = threading.Event()
        self._lease_lost = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop_event.wait(timeout=HEARTBEAT_INTERVAL_SECONDS):
            # NOTE: heartbeat() runs its own UPDATE + commit on this
            # connection. Sharing one connection between this thread and
            # the main thread's subprocess-waiting is safe ONLY because
            # the main thread isn't touching the connection while it
            # waits on the subprocess -- if it were issuing queries
            # concurrently, we'd need a second connection here, since a
            # single psycopg connection isn't safe for concurrent use
            # from two threads at once.
            still_owned = heartbeat(self._conn, self._run_id, self._worker_id)
            if not still_owned:
                self._lease_lost.set()
                self._proc.kill()
                return

    def __enter__(self) -> "HeartbeatGuard":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)

    @property
    def lease_lost(self) -> bool:
        return self._lease_lost.is_set()
