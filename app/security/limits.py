"""OS-level resource limits for the sandbox process.

Windows: a Job Object caps committed memory, allows exactly one process (no children) and kills the
process when the job handle closes. Linux/macOS: an RLIMIT_DATA memory cap applied in the child
before exec; process creation is blocked by the runtime audit hook in sandbox_runner.py.
"""

from __future__ import annotations

import sys
from typing import Callable


def posix_preexec(memory_mb: int) -> Callable[[], None] | None:
    if sys.platform == "win32":
        return None

    def apply() -> None:
        import resource

        # RLIMIT_DATA caps real heap/anonymous allocations. RLIMIT_AS would also count the large
        # *virtual* reservations numpy/scipy make at import, breaking ordinary code.
        # (Process creation is blocked by the runtime audit hook; RLIMIT_NPROC is not used
        # because it also counts threads and every process of the user.)
        limit = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))

    return apply


class WindowsJob:
    """Wraps a Win32 Job Object; a no-op elsewhere."""

    def __init__(self, memory_mb: int):
        self.handle = None
        if sys.platform != "win32":
            return
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._k32 = k32

        class IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]

        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        limit_process_memory, limit_active_process, kill_on_close = 0x100, 0x8, 0x2000
        info = ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = limit_process_memory | limit_active_process | kill_on_close
        # A venv's python.exe is a launcher that starts the real interpreter as a child process,
        # so the job must allow launcher + interpreter; the interpreter still cannot spawn anything.
        is_launcher = getattr(sys, "_base_executable", sys.executable) != sys.executable
        info.BasicLimitInformation.ActiveProcessLimit = 2 if is_launcher else 1
        info.ProcessMemoryLimit = memory_mb * 1024 * 1024
        handle = k32.CreateJobObjectW(None, None)
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        if not k32.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):  # 9 = extended limits
            k32.CloseHandle(handle)
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
        self.handle = handle

    def assign(self, process_handle: int) -> None:
        if self.handle and not self._k32.AssignProcessToJobObject(self.handle, process_handle):
            import ctypes

            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")

    def close(self) -> None:
        if self.handle:
            self._k32.CloseHandle(self.handle)  # kills the process if it is still running
            self.handle = None
