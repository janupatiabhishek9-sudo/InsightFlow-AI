import subprocess
import sys

import pandas as pd
import pytest

from app.security.code_guard import check_code
from app.security.sandbox import RUNNER, Sandbox


@pytest.mark.parametrize("code", [
    "import os\nos.system('dir')",
    "import subprocess\nsubprocess.run(['whoami'])",
    "import socket\nsocket.create_connection(('example.com', 80))",
    "open('../../etc/passwd').read()",
    "__import__('os').system('x')",
    "import pandas as pd\npd.read_csv('C:/secret.csv')",
    "import pandas as pd\nresult = pd.DataFrame().__class__.__bases__",
    "eval('1+1')",
    "import requests\nrequests.get('http://evil.example')",
    "import numpy as np\nnp.load('x.npy')",
    "import pandas as pd\ndata['x'].to_csv('out.csv')",
    "getattr(object, 'x')",
])
def test_code_guard_rejects_dangerous_code(code):
    assert not check_code(code).allowed


def test_code_guard_allows_analysis():
    code = "import pandas as pd\nimport numpy as np\nfrom scipy import stats\nresult = float(np.mean([1, 2, 3]))"
    assert check_code(code).allowed


def test_sandbox_runs_analysis(tmp_path):
    out = Sandbox(tmp_path).run("result = float(data['t']['x'].sum())", {"t": pd.DataFrame({"x": [1, 2, 3]})})
    assert out.ok and out.result == 6.0


def test_sandbox_rejects_before_execution(tmp_path):
    out = Sandbox(tmp_path).run("import os\nresult = os.environ")
    assert not out.ok and "code guard" in out.error


def test_sandbox_timeout(tmp_path):
    out = Sandbox(tmp_path, timeout=3).run("x = 0\nwhile True:\n    x += 1")
    assert not out.ok and "timed out" in out.error


def _run_raw(tmp_path, code: str, name: str = "raw") -> str:
    """Bypass the AST guard to test the runtime layers (audit hook, clean environment) on their own."""
    work = tmp_path / name
    work.mkdir()
    (work / "code.py").write_text(code, encoding="utf-8")
    proc = subprocess.run([sys.executable, "-I", str(RUNNER), str(work)], capture_output=True,
                          env=Sandbox.child_env(), timeout=60)
    return proc.stdout.decode()


def test_runtime_hook_blocks_file_read_outside_sandbox(tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret")
    out = _run_raw(tmp_path, f"result = open(r'{outside}').read()")
    assert '"ok": false' in out and "blocked" in out


def test_runtime_hook_blocks_subprocess_and_network(tmp_path):
    assert "blocked" in _run_raw(tmp_path, "import subprocess\nsubprocess.run(['whoami'])", "a")
    assert "blocked" in _run_raw(tmp_path, "import socket\nsocket.socket().connect(('127.0.0.1', 9))", "b")


def test_sandbox_cannot_see_parent_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    assert "OPENAI_API_KEY" not in Sandbox.child_env()
    out = _run_raw(tmp_path, "import os\nresult = os.environ.get('OPENAI_API_KEY')")
    assert '"result": null' in out and "sk-should-not-leak" not in out
