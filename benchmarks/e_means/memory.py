"""Sample a macOS child's current physical footprint and RSS during a fit.

No third-party packages, GPU APIs, or dataset imports are needed. The ctypes
layouts follow the public macOS SDK ``sys/resource.h`` rusage_info_v0/v4 and
``libproc.h`` proc_pid_rusage declaration.

The controller owns phase boundaries: receive FIT_READY from the child, call
``MemorySampler(pid).start()``, acknowledge GO, receive FIT_DONE, call ``stop()``,
then acknowledge STOP_ACK. Keeping the child blocked until STOP_ACK prevents
scoring or teardown from entering the measured interval.

Physical footprint and RSS are different kernel accounting metrics, not
additive components. Neither is a dedicated GPU-allocation counter. Physical
footprint is the kernel's process ledger and may include charged unified-memory
resources; RSS is currently resident process memory. These are process totals,
including data and imports already present at FIT_READY. Sampled phase peaks
can miss transients shorter than the polling interval. The lifetime maximum is
reported only as a diagnostic because it includes preparation and warm-up.
"""

from __future__ import annotations

import ctypes
import errno
import math
import os
import sys
import threading
import time
from functools import lru_cache
from numbers import Integral, Real

__all__ = ["snapshot", "MemorySampler"]

_V0_FIELDS = [
    ("ri_uuid", ctypes.c_uint8 * 16),
    *[(name, ctypes.c_uint64) for name in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
        "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
        "ri_proc_start_abstime", "ri_proc_exit_abstime",
    )],
]


class _RusageInfoV0(ctypes.Structure):
    _fields_ = _V0_FIELDS


class _RusageInfoV4(ctypes.Structure):
    _fields_ = _V0_FIELDS + [
        (name, ctypes.c_uint64) for name in (
            "ri_child_user_time", "ri_child_system_time", "ri_child_pkg_idle_wkups",
            "ri_child_interrupt_wkups", "ri_child_pageins", "ri_child_elapsed_abstime",
            "ri_diskio_bytesread", "ri_diskio_byteswritten", "ri_cpu_time_qos_default",
            "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background",
            "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
            "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
            "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
            "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
            "ri_runnable_time",
        )
    ]


@lru_cache(maxsize=1)
def _rusage_function():
    if sys.platform != "darwin":
        raise NotImplementedError("process footprint sampling requires macOS")
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    function = library.proc_pid_rusage
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    function.restype = ctypes.c_int
    return function


def snapshot(pid: int) -> dict:
    """Return one process-memory observation; all memory quantities are bytes.

    ``lifetime_peak_physical_footprint_bytes`` is None if only rusage V0 is
    supported. ``process_start_abstime`` identifies the process incarnation;
    its opaque units are not seconds. ``monotonic_ns`` is the midpoint of the
    system call in the sampling process's monotonic clock. The syscall's wall
    duration is included so callers can inspect collection overhead.

    Raise ValueError for an invalid PID, NotImplementedError outside macOS,
    and OSError if the process exits or the kernel denies inspection. Failure
    is never converted to a zero-byte observation.
    """
    if isinstance(pid, bool) or not isinstance(pid, Integral) or not 0 < pid <= 2**31 - 1:
        raise ValueError("pid must be a positive signed 32-bit integer")
    pid = int(pid)
    function = _rusage_function()
    start = time.monotonic_ns()
    info = _RusageInfoV4()
    flavor = 4
    ctypes.set_errno(0)
    result = function(pid, flavor, ctypes.byref(info))
    if result != 0:
        error = ctypes.get_errno() or errno.EIO
        if error not in {errno.EINVAL, errno.ENOTSUP}:
            raise OSError(error, f"proc_pid_rusage({pid}): {os.strerror(error)}")
        info, flavor = _RusageInfoV0(), 0
        ctypes.set_errno(0)
        if function(pid, flavor, ctypes.byref(info)) != 0:
            error = ctypes.get_errno() or errno.EIO
            raise OSError(error, f"proc_pid_rusage({pid}): {os.strerror(error)}")
    finish = time.monotonic_ns()
    return {
        "supported": True,
        "source": "proc_pid_rusage",
        "pid": pid,
        "rusage_flavor": flavor,
        "monotonic_ns": (start + finish) // 2,
        "collection_duration_ns": finish - start,
        "rss_bytes": int(info.ri_resident_size),
        "physical_footprint_bytes": int(info.ri_phys_footprint),
        "lifetime_peak_physical_footprint_bytes": (
            int(info.ri_lifetime_max_phys_footprint) if flavor >= 4 else None
        ),
        "process_start_abstime": int(info.ri_proc_start_abstime),
        "process_exit_abstime": int(info.ri_proc_exit_abstime),
    }


class MemorySampler:
    """Poll a child from the parent, retaining only phase aggregates by default.

    ``start`` records a synchronous baseline, launches a background polling
    thread, and returns self. ``stop`` stops/join the thread, records a final
    snapshot while the child remains blocked, and returns a JSON-safe summary.
    Each object measures exactly one phase; repeated ``stop`` calls return the
    same summary. No subprocess is started, signalled, or stopped by this class.

    Errors after the initial snapshot are recorded in ``sampling_errors`` and
    make ``complete`` false. A PID-reuse check protects against accidentally
    sampling a replacement process. ``keep_samples=True`` includes the raw
    observations in the summary; the default uses constant memory over time.
    """

    def __init__(self, pid: int, interval_seconds: float = 0.01, *, keep_samples=False):
        if isinstance(pid, bool) or not isinstance(pid, Integral) or not 0 < pid <= 2**31 - 1:
            raise ValueError("pid must be a positive signed 32-bit integer")
        if (
            isinstance(interval_seconds, bool) or not isinstance(interval_seconds, Real)
            or not math.isfinite(interval_seconds) or interval_seconds <= 0
        ):
            raise ValueError("interval_seconds must be finite and positive")
        self.pid = int(pid)
        self.interval_seconds = float(interval_seconds)
        self.keep_samples = bool(keep_samples)
        self._stop_event = threading.Event()
        self._thread = None
        self._summary = None
        self._sample_count = 0
        self._baseline = self._end = None
        self._maximum_gap_ns = 0
        self._peak_rss = self._peak_footprint = 0
        self._lifetime_peak = None
        self._errors = []
        self._observations = [] if self.keep_samples else None

    def _record(self, observation):
        if self._baseline is None:
            self._baseline = observation
        elif observation["process_start_abstime"] != self._baseline["process_start_abstime"]:
            raise ProcessLookupError(errno.ESRCH, "PID was reused by a different process")
        if self._end is not None:
            gap = observation["monotonic_ns"] - self._end["monotonic_ns"]
            self._maximum_gap_ns = max(self._maximum_gap_ns, gap)
        self._end = observation
        self._sample_count += 1
        self._peak_rss = max(self._peak_rss, observation["rss_bytes"])
        self._peak_footprint = max(self._peak_footprint, observation["physical_footprint_bytes"])
        lifetime = observation["lifetime_peak_physical_footprint_bytes"]
        if lifetime is not None:
            self._lifetime_peak = max(self._lifetime_peak or 0, lifetime)
        if self._observations is not None:
            self._observations.append(observation)

    def _observe(self):
        try:
            self._record(snapshot(self.pid))
            return True
        except OSError as error:
            self._errors.append({
                "monotonic_ns": time.monotonic_ns(),
                "errno": error.errno,
                "message": str(error),
            })
            return False

    def _poll(self):
        deadline = time.monotonic() + self.interval_seconds
        while not self._stop_event.wait(max(0.0, deadline - time.monotonic())):
            if not self._observe():
                # A failed observation leaves the last valid snapshot intact.
                # End this polling loop; stop() still attempts its final read.
                return
            deadline += self.interval_seconds
            if deadline < time.monotonic():
                # Skip missed deadlines instead of generating a busy catch-up.
                deadline = time.monotonic() + self.interval_seconds

    def start(self):
        """Record baseline and begin polling; call before acknowledging GO."""
        if self._thread is not None:
            raise RuntimeError("a MemorySampler can only be started once")
        self._record(snapshot(self.pid))
        self._thread = threading.Thread(target=self._poll, name=f"memory-pid-{self.pid}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        """Finish polling and return a summary; call before STOP_ACK."""
        if self._thread is None:
            raise RuntimeError("start() must be called before stop()")
        if self._summary is not None:
            return self._summary
        self._stop_event.set()
        self._thread.join()
        self._observe()
        self._summary = {
            "supported": True,
            "source": "proc_pid_rusage",
            "pid": self.pid,
            "sample_interval_seconds": self.interval_seconds,
            "sample_count": self._sample_count,
            "duration_seconds": (self._end["monotonic_ns"] - self._baseline["monotonic_ns"]) / 1e9,
            "maximum_sample_gap_seconds": self._maximum_gap_ns / 1e9,
            "baseline": self._baseline,
            "end": self._end,
            "peak_rss_bytes": self._peak_rss,
            "peak_physical_footprint_bytes": self._peak_footprint,
            "rss_peak_increase_bytes": max(0, self._peak_rss - self._baseline["rss_bytes"]),
            "physical_footprint_peak_increase_bytes": max(0, self._peak_footprint - self._baseline["physical_footprint_bytes"]),
            "lifetime_peak_physical_footprint_bytes": self._lifetime_peak,
            "sampling_errors": self._errors,
            "complete": not self._errors,
            "notes": [
                "Phase peaks are maxima of current-memory samples, not lifetime maxima.",
                "Sampling can miss transients shorter than the observed sampling gaps.",
                "Phase totals include memory allocated before FIT_READY; increments subtract that baseline.",
                "RSS and physical footprint are different process metrics and must not be added.",
                "Process footprint is not an exclusive GPU-memory measurement.",
                "Lifetime peak is diagnostic only and includes preparation/warm-up.",
            ],
        }
        if self._observations is not None:
            self._summary["samples"] = self._observations
        return self._summary

    def __enter__(self):
        return self.start()

    def __exit__(self, exception_type, exception, traceback):
        self.stop()
