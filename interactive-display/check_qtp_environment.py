from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

CORE_PKGS = [
    "numpy",
    "matplotlib",
    "pillow",
    "imageio",
    "imageio-ffmpeg",
    "ipywidgets",
    "jupyterlab_widgets",
    "widgetsnbextension",
    "ipykernel",
    "jupyterlab",
    "notebook",
    "nbconvert",
    "playwright",
    "voila",
]
WIDGET_RULES = {
    "ipywidgets": 8,
    "jupyterlab_widgets": 3,
    "widgetsnbextension": 4,
}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "NOT INSTALLED"


def major_version(value: str) -> int | None:
    match = re.match(r"^(\d+)", value or "")
    return int(match.group(1)) if match else None


def kernelspec_status() -> tuple[bool, bool | None, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "jupyter", "kernelspec", "list", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if proc.returncode != 0:
            return (
                False,
                None,
                f"jupyter kernelspec list failed: {proc.stderr.strip() or proc.stdout.strip()}",
            )
        data = json.loads(proc.stdout)
        specs = data.get("kernelspecs", {})
        if "qtp-display" not in specs:
            available = ", ".join(sorted(specs)) or "none"
            return False, None, f"qtp-display not registered; available: {available}"
        spec = specs["qtp-display"].get("spec", {})
        argv = spec.get("argv", [])
        exe = argv[0] if argv else "unknown"
        try:
            exe_matches = Path(exe).resolve() == Path(sys.executable).resolve()
        except Exception:
            exe_matches = False
        if exe_matches:
            return True, True, f"qtp-display registered; executable matches current Python: {exe}"
        return (
            True,
            False,
            f"qtp-display registered, but executable differs: {exe}; current Python: {sys.executable}",
        )
    except Exception as exc:
        return False, None, f"could not inspect kernelspecs ({exc.__class__.__name__}: {exc})"


def labextension_status() -> tuple[bool, str]:
    """Check the prebuilt widget manager in the current Jupyter server environment."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "jupyter", "labextension", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
        )
        output = ANSI_RE.sub("", (proc.stdout or "") + "\n" + (proc.stderr or ""))
        manager_lines = [
            line.strip()
            for line in output.splitlines()
            if "@jupyter-widgets/jupyterlab-manager" in line
        ]
        if proc.returncode != 0:
            return False, f"jupyter labextension list failed: {output.strip()}"
        if not manager_lines:
            return False, "prebuilt @jupyter-widgets/jupyterlab-manager extension not found"
        line = manager_lines[-1]
        ok = "enabled" in line.lower() and "ok" in line.lower()
        return ok, line
    except Exception as exc:
        return False, f"could not inspect JupyterLab extensions ({exc.__class__.__name__}: {exc})"


def import_status(mod: str) -> str:
    try:
        importlib.import_module(mod)
        return "OK"
    except Exception as exc:
        return f"FAIL ({exc.__class__.__name__}: {exc})"


def widget_package_status(pkg_versions: dict[str, str]) -> tuple[bool, list[str]]:
    messages = []
    ok = True
    for pkg, expected_major in WIDGET_RULES.items():
        value = pkg_versions.get(pkg, "NOT INSTALLED")
        major = major_version(value)
        compatible = major == expected_major
        ok = ok and compatible
        if value == "NOT INSTALLED":
            messages.append(f"{pkg}: missing (expected major {expected_major})")
        elif not compatible:
            messages.append(f"{pkg}: {value} (expected major {expected_major})")
        else:
            messages.append(f"{pkg}: {value} (compatible)")
    return ok, messages


def repair_command() -> str:
    exe = json.dumps(sys.executable)
    return (
        f"{exe} -m pip install --upgrade "
        '"ipywidgets>=8.1,<9" "jupyterlab_widgets>=3,<4" '
        '"widgetsnbextension>=4,<5" "jupyterlab>=4.2,<5" "notebook>=7.2,<8"'
    )


def collect_status(root: Path) -> dict:
    pkg_versions = {pkg: version(pkg) for pkg in CORE_PKGS}
    kernel_ok, exe_matches, kernel_msg = kernelspec_status()
    lab_ok, lab_msg = labextension_status()
    widget_pkgs_ok, widget_messages = widget_package_status(pkg_versions)
    return {
        "package_root": str(root),
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "packages": pkg_versions,
        "widget_packages_ok": widget_pkgs_ok,
        "widget_package_messages": widget_messages,
        "kernelspec_ok": kernel_ok,
        "kernelspec_executable_matches": exe_matches,
        "kernelspec_message": kernel_msg,
        "labextension_ok": lab_ok,
        "labextension_message": lab_msg,
        "session_controls_enabled": os.environ.get("QTP_INTERACTIVE_CONTROLS_READY") == "1",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Check the QTP display environment, interactive controls, and bundled results"
    )
    ap.add_argument(
        "--require-controls",
        action="store_true",
        help="return nonzero unless compatible widget packages, kernel, and JupyterLab manager are present",
    )
    ap.add_argument(
        "--reference-only",
        action="store_true",
        help="check bundled reference results without requiring interactive controls",
    )
    ap.add_argument("--json", action="store_true", help="also write environment_check_report.json")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent
    report = collect_status(root)
    print("QTP environment check")
    print(f"Package root: {root}")
    print(f"Python executable: {sys.executable}")
    print(f"Python: {platform.python_version()}")

    print("\nPackage versions")
    for pkg in CORE_PKGS:
        print(f"  {pkg:22s} {report['packages'][pkg]}")
    failed = False
    print("\nImports")
    required_imports = {
        "qtp_kernels": "qtp_kernels",
        "qtp_display": "qtp_display",
        "qtp_movie_renderer": "qtp_movie_renderer",
        "verify_display": "verify_display",
        "imageio_ffmpeg": "imageio_ffmpeg",
    }
    for label, mod in required_imports.items():
        status = import_status(mod)
        print(f"  {label:22s} {status}")
        failed = failed or status.startswith("FAIL")

    print("\nQTP Display kernelspec")
    kernel_level = (
        "OK"
        if report["kernelspec_ok"] and report["kernelspec_executable_matches"]
        else (
            "WARN"
            if report["kernelspec_ok"]
            else ("FAIL" if args.require_controls and not args.reference_only else "WARN")
        )
    )
    print(f"  {kernel_level}: {report['kernelspec_message']}")

    optional_controls = bool(args.reference_only and not args.require_controls)
    print("\nInteractive-control packages")
    for message in report["widget_package_messages"]:
        level = "OK" if "compatible" in message else ("WARN" if optional_controls else "FAIL")
        print(f"  {level}: {message}")

    print("\nJupyterLab control manager")
    manager_level = "OK" if report["labextension_ok"] else ("WARN" if optional_controls else "FAIL")
    print(f"  {manager_level}: {report['labextension_message']}")
    print(f"  controls enabled for this session: {report['session_controls_enabled']}")

    interactive_ok = bool(
        report["widget_packages_ok"]
        and report["kernelspec_ok"]
        and report["kernelspec_executable_matches"]
        and report["labextension_ok"]
    )
    report["interactive_controls_ready"] = interactive_ok

    if not interactive_ok:
        if optional_controls:
            print("\nInteractive-control readiness: NOT REQUIRED FOR REFERENCE CHECKS")
        else:
            print("\nInteractive-control readiness: FAIL")
        print("  Suggested installation command for the Jupyter environment:")
        print(f"  {repair_command()}")
        print(
            f'  {json.dumps(sys.executable)} -m ipykernel install --user --name qtp-display --display-name "QTP Display"'
        )
        print("  Then launch with: python launch_qtp_explorer.py")
        if args.require_controls and not args.reference_only:
            failed = True
    else:
        print("\nInteractive-control readiness: PASS")
        print(
            "  Launch with `python launch_qtp_explorer.py` so the notebook can safely enable interactive controls."
        )

    try:
        import verify_display

        ok, rows = verify_display.verify_display(root)
        print("\nBundled reference verification")
        print(f"  status: {'PASS' if ok else 'FAIL'}")
        print(
            f"  report rows passing in selected mode: {sum(bool(r['ok']) for r in rows)} / {len(rows)}"
        )
        report["bundled_reference_verification"] = bool(ok)
        failed = failed or not ok
    except Exception as exc:
        print("\nBundled reference verification")
        print(f"  status: FAIL ({exc.__class__.__name__}: {exc})")
        report["bundled_reference_verification"] = False
        report["bundled_reference_error"] = repr(exc)
        failed = True

    if args.json:
        out = root / "verification_runtime" / "environment_check_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {out.name}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
