"""One-command launcher.

  python -m app.run            # the web app (one process, lowest RAM)
  python -m app.run --api      # the web app + REST API (UI talks to the API, so the service loads once)

Generates the sample dataset on first run and opens the browser. Ctrl+C stops everything.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import webbrowser

from app.config import PROJECT_ROOT

UI_SCRIPT = PROJECT_ROOT / "frontend" / "streamlit_app.py"
SAMPLE = PROJECT_ROOT / "data" / "examples" / "sales.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Start InsightFlow AI")
    parser.add_argument("--api", action="store_true", help="also start the REST API (UI then uses it over HTTP)")
    parser.add_argument("--ui-port", type=int, default=8501)
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not SAMPLE.exists():
        print("Generating the sample dataset ...")
        subprocess.run([sys.executable, "-m", "app.datagen"], cwd=PROJECT_ROOT, check=True)

    env = dict(os.environ)
    processes: list[subprocess.Popen] = []
    if args.api:
        processes.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(args.api_port)], cwd=PROJECT_ROOT, env=env))
        env.update(UI_BACKEND="http", API_URL=f"http://localhost:{args.api_port}")
    processes.append(subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(UI_SCRIPT), "--server.port", str(args.ui_port),
         "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=PROJECT_ROOT, env=env))

    url = f"http://localhost:{args.ui_port}"
    print(f"\nInsightFlow AI is starting: {url}")
    if args.api:
        print(f"REST API: http://localhost:{args.api_port}/docs")
    print("Press Ctrl+C to stop.\n")
    if not args.no_browser:
        time.sleep(4)
        webbrowser.open(url)
    try:
        while all(p.poll() is None for p in processes):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
