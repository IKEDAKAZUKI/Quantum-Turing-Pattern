from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import check_qtp_environment as environment_check

NOTEBOOK_NAME = "qtp_exhibit.ipynb"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the Quantum Turing Patterns presentation viewer as a "
            "localhost-only Voilà application. Notebook code and the Jupyter "
            "file browser are hidden."
        )
    )
    parser.add_argument("--port", type=int, default=8866)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--idle-reset-seconds", type=int, default=300)
    args, extra = parser.parse_known_args(argv)

    root = Path(__file__).resolve().parent
    notebook = root / NOTEBOOK_NAME
    if not notebook.exists():
        print(f"Notebook not found: {notebook}", file=sys.stderr)
        return 2
    if importlib.util.find_spec("voila") is None:
        print("Voilà is not installed in this Python environment.", file=sys.stderr)
        print("Install the package environment again, or run:", file=sys.stderr)
        print('  python -m pip install "voila>=0.5.8,<1"', file=sys.stderr)
        return 2

    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            "qtp-display",
            "--display-name",
            "QTP Display",
        ],
        cwd=root,
        check=False,
    )
    if install.returncode != 0:
        return install.returncode

    kernel_ok, executable_matches, kernel_message = environment_check.kernelspec_status()
    if not (kernel_ok and executable_matches):
        print(f"QTP kernelspec refresh failed: {kernel_message}", file=sys.stderr)
        return 2

    if args.check_only:
        print("Voilà, the presentation assets, and the QTP kernel are ready.")
        print(f"Python: {sys.executable}")
        print(f"Kernel: {kernel_message}")
        return 0

    environment = os.environ.copy()
    environment.update(
        {
            "QTP_EXHIBIT_IDLE_SECONDS": str(max(0, int(args.idle_reset_seconds))),
            "QTP_EXHIBIT_EMBED_MEDIA": "1",
        }
    )
    command = [
        sys.executable,
        "-m",
        "voila",
        NOTEBOOK_NAME,
        "--no-browser",
        "--Voila.ip=127.0.0.1",
        f"--port={args.port}",
        "--VoilaConfiguration.strip_sources=True",
        "--VoilaConfiguration.show_tracebacks=False",
    ]
    if not args.no_browser:
        command.remove("--no-browser")
    command.extend(extra)
    print("Launching the QTP presentation viewer on localhost:")
    print("  " + " ".join(json.dumps(part) for part in command))
    return subprocess.call(command, cwd=root, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
