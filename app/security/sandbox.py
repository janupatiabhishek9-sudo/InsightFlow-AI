"""Run guarded analysis code in a separate, isolated Python process.

Layers: AST code guard -> fresh interpreter (-I isolated mode) with an empty environment
(no secrets) and a private temporary working directory -> OS limits (memory cap, no child
processes; see limits.py) -> wall-clock timeout and output-size limit -> runtime audit hook
inside the child (see sandbox_runner.py).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from app.security.code_guard import check_code
from app.security.limits import WindowsJob, posix_preexec

RUNNER = Path(__file__).with_name("sandbox_runner.py")
MAX_OUTPUT_BYTES = 200_000


class SandboxResult(BaseModel):
    ok: bool
    result: Any = None
    error: str | None = None


class Sandbox:
    def __init__(self, base_dir: Path, timeout: float = 20, memory_mb: int = 1024):
        self.base_dir = Path(base_dir)
        self.timeout = timeout
        self.memory_mb = memory_mb

    @staticmethod
    def child_env() -> dict[str, str]:
        """A minimal environment: nothing from the parent (API keys, tokens) is inherited."""
        env = {"PYTHONIOENCODING": "utf-8", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        if sys.platform == "win32":  # Windows needs SYSTEMROOT to start Python at all
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
        return env

    def run(self, code: str, inputs: dict[str, pd.DataFrame] | None = None) -> SandboxResult:
        verdict = check_code(code)
        if not verdict.allowed:
            return SandboxResult(ok=False, error="code guard rejected: " + "; ".join(verdict.violations))

        self.base_dir.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="run_", dir=self.base_dir))
        try:
            (workdir / "inputs").mkdir()
            for name, df in (inputs or {}).items():
                df.to_csv(workdir / "inputs" / f"{name}.csv", index=False)
            (workdir / "code.py").write_text(code, encoding="utf-8")
            stdout, stderr, error = self._execute(workdir)
            if error:
                return SandboxResult(ok=False, error=error)
            if len(stdout) > MAX_OUTPUT_BYTES:
                return SandboxResult(ok=False, error="sandbox output exceeded size limit")
            try:
                payload = json.loads(stdout.decode("utf-8"))
            except json.JSONDecodeError:
                detail = stderr.decode(errors="replace")[-500:] or "no output (process killed, possibly by the memory limit)"
                return SandboxResult(ok=False, error="sandbox produced invalid output: " + detail)
            return SandboxResult(ok=bool(payload.get("ok")), result=payload.get("result"), error=payload.get("error"))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _execute(self, workdir: Path) -> tuple[bytes, bytes, str | None]:
        job = WindowsJob(self.memory_mb)
        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", str(RUNNER), str(workdir)],
                cwd=workdir, env=self.child_env(), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=posix_preexec(self.memory_mb),
            )
            try:
                if job.handle:
                    job.assign(int(proc._handle))  # noqa: SLF001 - the Win32 process handle
            except OSError:
                proc.kill()
                proc.communicate()
                return b"", b"", "sandbox could not apply resource limits; refusing to run"
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                return b"", b"", f"sandbox timed out after {self.timeout}s"
            return stdout[: MAX_OUTPUT_BYTES + 1], stderr, None
        finally:
            job.close()
