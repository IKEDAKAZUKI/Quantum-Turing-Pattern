from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import check_qtp_environment as environment_check


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch the Quantum Turing Patterns research explorer in Jupyter."
    )
    parser.add_argument("--frontend", choices=["lab", "notebook"], default="lab")
    parser.add_argument("--notebook", default="qtp_explorer.ipynb")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args, extra = parser.parse_known_args(argv)

    root = Path(__file__).resolve().parent
    status = environment_check.collect_status(root)
    interactive_ok = bool(status["widget_packages_ok"] and status["labextension_ok"])
    if not interactive_ok:
        print("Interactive controls are not available in this Python environment.", file=sys.stderr)
        for message in status["widget_package_messages"]:
            print(f"  {message}", file=sys.stderr)
        print(f"  JupyterLab manager: {status['labextension_message']}", file=sys.stderr)
        print("\nSuggested installation command:", file=sys.stderr)
        print(f"  {environment_check.repair_command()}", file=sys.stderr)
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
        print("Interactive controls and the QTP kernel are ready.")
        print(f"Python: {sys.executable}")
        print(f"Kernel: {kernel_message}")
        return 0

    notebook = root / args.notebook
    if not notebook.exists():
        print(f"Notebook not found: {notebook}", file=sys.stderr)
        return 2

    environment = os.environ.copy()
    environment["QTP_INTERACTIVE_CONTROLS_READY"] = "1"
    environment["QTP_DISPLAY_ROOT"] = str(root)
    command = [sys.executable, "-m", "jupyter", args.frontend, notebook.name]
    if args.no_browser:
        command.append("--no-browser")
    command.extend(extra)
    print("Launching the research explorer:")
    print("  " + " ".join(json.dumps(part) for part in command))
    return subprocess.call(command, cwd=root, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
