"""Child-process entry point for sandboxed analysis code. Never import this in the main app.

Protocol: argv[1] is a work directory containing `code.py` and `inputs/*.csv`.
The code sees `data` (dict of DataFrames) and must assign a JSON-serialisable `result`.
A runtime audit hook blocks network, process creation and file access outside the work
directory and the Python installation, as a second layer behind the AST code guard.
"""

import json
import os
import sys


def main() -> None:
    workdir = os.path.realpath(sys.argv[1])
    import math  # noqa: F401  - pre-import allowed libraries before locking down
    import statistics  # noqa: F401

    import numpy as np  # noqa: F401
    import pandas as pd
    import scipy.stats  # noqa: F401

    data = {}
    inputs = os.path.join(workdir, "inputs")
    for name in sorted(os.listdir(inputs)) if os.path.isdir(inputs) else []:
        if name.endswith(".csv"):
            data[name[:-4]] = pd.read_csv(os.path.join(inputs, name))
    with open(os.path.join(workdir, "code.py"), encoding="utf-8") as fh:
        code = fh.read()

    readable_roots = tuple(os.path.realpath(p) for p in {sys.prefix, sys.base_prefix, sys.exec_prefix, workdir})
    blocked_prefixes = ("socket.", "subprocess.", "os.system", "os.exec", "os.spawn", "os.posix_spawn",
                        "os.fork", "os.kill", "os.putenv", "os.unsetenv", "ctypes.", "winreg.", "urllib.",
                        "http.", "ftplib.", "smtplib.", "webbrowser.", "shutil.", "os.remove", "os.rename",
                        "os.rmdir", "os.mkdir", "os.chmod", "os.symlink", "os.link", "sys.addaudithook")

    def hook(event: str, args: tuple) -> None:
        if event.startswith(blocked_prefixes) or event in {"os.startfile", "import.subprocess"}:
            raise PermissionError(f"sandbox: '{event}' is blocked")
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.path.realpath(os.fsdecode(args[0]))
            mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
            writing = any(c in mode for c in "wax+")
            if writing and not path.startswith(workdir):
                raise PermissionError("sandbox: writing outside the work directory is blocked")
            if not path.startswith(readable_roots):
                raise PermissionError("sandbox: reading outside the sandbox is blocked")

    sys.addaudithook(hook)
    namespace = {"data": data, "result": None}
    exec(compile(code, "<analysis>", "exec"), namespace)  # noqa: S102 - guarded + isolated process
    result = namespace.get("result")
    if hasattr(result, "to_dict"):
        result = result.to_dict(orient="records") if hasattr(result, "columns") else result.to_dict()
    sys.stdout.write(json.dumps({"ok": True, "result": result}, default=str))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # report every failure as structured output
        sys.stdout.write(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
