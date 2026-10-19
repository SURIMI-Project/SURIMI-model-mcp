"""python -m surimi_model_mcp starts the server (2026-10-13).

The image used to run `python server.py`. After the package restructure it runs the package. This starts
it on a free port with no storage configured (so the prefetch threads skip themselves) and checks that
the MCP endpoint answers.
"""
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_package_starts_and_answers_on_mcp():
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("AWS_")}
    env.update(PYTHONPATH=str(ROOT), MCP_PORT=str(port), SURIMI_RECEIPT_SECRET="x")
    proc = subprocess.Popen([sys.executable, "-m", "surimi_model_mcp"], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline, status = time.time() + 60, None
        while time.time() < deadline and status is None:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/mcp", timeout=2)
                status = 200
            except urllib.error.HTTPError as e:
                status = e.code          # any HTTP answer means the app is serving
            except OSError:
                time.sleep(0.5)
        assert status is not None, "the server did not start"
    finally:
        proc.terminate()
        out = proc.communicate(timeout=20)[0]
    assert "drift check OK (26 tools" in out, out[-2000:]
