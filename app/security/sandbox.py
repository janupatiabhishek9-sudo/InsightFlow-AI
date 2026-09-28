"""Run guarded analysis code in a separate, isolated Python process.

Layers: AST code guard -> fresh interpreter (-I isolated mode) with an empty environment
(no secrets), a private temporary working directory, a wall-clock timeout, an output-size
limit and a runtime audit hook. Memory limits are not enforced on Windows; on Linux the
process could additionally be wrapped with resource limits or a container.
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

RUNNER = Path(__file__).with_name("sandbox_runner.py")
MAX_OUTPUT_BYTES = 200_000


class SandboxResult(BaseModel):
    ok: bool
    result: Any = None
    error: str | None = None


class Sandbox:
    def __init__(self, base_dir: Path, timeout: float = 20):
        self.base_dir = Path(base_dir)
        self.timeout = timeout

    @staticmethod
    def child_env() -> dict[str, str]:
        """A minimal environment: nothing from the parent (API keys, tokens) is inherited."""
        env = {"PYTHONIOENCODING": "utf-8"}
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
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(RUNNER), str(workdir)],
                    cwd=workdir,
                    env=self.child_env(),
                    capture_output=True,
                    timeout=self.timeout,
                    stdin=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired:
                return SandboxResult(ok=False, error=f"sandbox timed out after {self.timeout}s")
            out = proc.stdout[: MAX_OUTPUT_BYTES + 1]
            if len(out) > MAX_OUTPUT_BYTES:
                return SandboxResult(ok=False, error="sandbox output exceeded size limit")
            try:
                payload = json.loads(out.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                return SandboxResult(ok=False, error="sandbox produced invalid output: " + proc.stderr.decode(errors="replace")[-500:])
            return SandboxResult(ok=bool(payload.get("ok")), result=payload.get("result"), error=payload.get("error"))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
