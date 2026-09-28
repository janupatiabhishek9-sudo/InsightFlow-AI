"""Static (AST) validation of model-generated Python before it reaches the sandbox."""

from __future__ import annotations

import ast

from pydantic import BaseModel

ALLOWED_IMPORTS = {"pandas", "numpy", "scipy", "scipy.stats", "statistics", "math", "json", "plotly",
                   "plotly.express", "plotly.graph_objects", "collections", "itertools", "datetime"}
FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "input", "breakpoint", "exit", "quit", "help", "memoryview", "__builtins__", "__loader__",
    "__spec__", "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib", "ctypes", "builtins",
}
FORBIDDEN_ATTRS = {
    "system", "popen", "spawn", "fork", "remove", "unlink", "rmdir", "environ", "getenv", "putenv",
    "to_csv", "to_excel", "to_parquet", "to_pickle", "to_sql", "to_hdf", "to_feather", "to_stata",
    "to_clipboard", "to_html", "write_html", "write_image", "write_json", "savetxt", "save", "savez",
    "load", "loadtxt", "fromfile", "tofile", "genfromtxt", "memmap", "read_pickle",
}
MAX_CODE_CHARS = 8000


class CodeGuardResult(BaseModel):
    allowed: bool
    violations: list[str] = []


def check_code(code: str) -> CodeGuardResult:
    if len(code) > MAX_CODE_CHARS:
        return CodeGuardResult(allowed=False, violations=[f"code longer than {MAX_CODE_CHARS} characters"])
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return CodeGuardResult(allowed=False, violations=[f"syntax error: {e.msg} (line {e.lineno})"])

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    violations.append(f"import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "") not in ALLOWED_IMPORTS:
                violations.append(f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            violations.append(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                violations.append(f"access to private/dunder attribute '{node.attr}' is not allowed")
            elif node.attr in FORBIDDEN_ATTRS or node.attr.startswith("read_"):
                violations.append(f"attribute '{node.attr}' (file/process/system access) is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            violations.append("global/nonlocal statements are not allowed")
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncWith, ast.AsyncFor)):
            violations.append("async code is not allowed")
    return CodeGuardResult(allowed=not violations, violations=sorted(set(violations)))
