#!/usr/bin/env python3
"""Validate the notebooks and the standard notebook mode.

The default mode performs static checks. ``--execute-standard`` copies the
package to a temporary directory, executes the research notebook in standard
mode, verifies the exported PNG, JSON, and manifest, and confirms that
preview generation can be disabled without creating files.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat

NOTEBOOKS = (
    "qtp_explorer_portable.ipynb",
    "qtp_explorer.ipynb",
    "qtp_exhibit.ipynb",
)
WIDGET_MIME = {
    "application/vnd.jupyter.widget-view+json",
    "application/vnd.jupyter.widget-state+json",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_notebook(path: Path, *, source_like: bool) -> dict[str, Any]:
    nb = nbformat.read(path, as_version=4)
    errors: list[str] = []
    widget_mime: list[str] = []
    error_outputs: list[str] = []
    code_cells = 0
    markdown_cells = sum(1 for cell in nb.cells if cell.cell_type == "markdown")

    if nb.metadata.get("widgets"):
        errors.append("saved widget state is present in notebook metadata")

    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        code_cells += 1
        try:
            ast.parse(cell.source, filename=f"{path.name}:cell-{index}")
        except SyntaxError as exc:
            errors.append(f"cell {index} syntax error: {exc}")
        if source_like and (cell.get("outputs") or cell.get("execution_count") is not None):
            errors.append(f"source cell {index} contains saved execution output")
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                error_outputs.append(f"cell {index}: {output.get('ename')}: {output.get('evalue')}")
            data = output.get("data", {})
            for key in WIDGET_MIME.intersection(data):
                widget_mime.append(f"cell {index}: {key}")

    source = "\n".join(cell.source for cell in nb.cells)
    code_source = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    function_defs = []
    try:
        parsed = ast.parse(code_source or "pass")
        function_defs = [
            node.name
            for node in ast.walk(parsed)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
    except SyntaxError:
        function_defs = []

    if source_like and path.name in {
        "qtp_explorer_portable.ipynb",
        "qtp_explorer.ipynb",
    }:
        for marker in (
            "from qtp_notebook_app import show_research_app",
            "show_research_app()",
        ):
            if marker not in source:
                errors.append(f"research entry-point marker is absent: {marker}")
        if code_cells > 2:
            errors.append(
                f"research entry notebook has {code_cells} code cells; expected at most 2"
            )
        if function_defs:
            errors.append(
                "research entry notebook contains function definitions: " + ", ".join(function_defs)
            )
        if path.name == "qtp_explorer.ipynb" and (
            "Python executable:" in source or "Kernel executable:" in source
        ):
            errors.append("research entry notebook exposes environment-check output")

    if source_like and path.name == "qtp_exhibit.ipynb":
        for marker in (
            "from qtp_exhibit import show_exhibit",
            "show_exhibit()",
        ):
            if marker not in source:
                errors.append(f"exhibit entry-point marker is absent: {marker}")
        if len(nb.cells) != 1 or code_cells != 1 or markdown_cells != 0:
            errors.append(
                f"exhibit notebook shape is cells={len(nb.cells)}, code={code_cells}, "
                f"markdown={markdown_cells}; expected exactly one code cell and no markdown"
            )
        if function_defs:
            errors.append(
                "exhibit notebook contains function definitions: " + ", ".join(function_defs)
            )
        for forbidden in (
            "Kernel executable",
            "provenance JSON",
            "verification report",
            "RNG seed",
        ):
            if forbidden in source:
                errors.append(f"exhibit notebook exposes technical visitor content: {forbidden}")

    if widget_mime:
        errors.append("widget MIME output is present: " + "; ".join(widget_mime))
    if error_outputs:
        errors.append("Python error output is present: " + "; ".join(error_outputs))

    return {
        "path": path.name,
        "ok": not errors,
        "code_cells": code_cells,
        "markdown_cells": markdown_cells,
        "total_cells": len(nb.cells),
        "kernelspec": dict(nb.metadata.get("kernelspec", {})),
        "saved_widget_state": bool(nb.metadata.get("widgets")),
        "widget_mime_outputs": widget_mime,
        "error_outputs": error_outputs,
        "errors": errors,
        "sha256": _sha256(path),
    }


def _static_checks(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    modules = []
    for path in sorted(root.glob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
            modules.append({"path": path.name, "ok": True, "sha256": _sha256(path)})
        except (SyntaxError, UnicodeError) as exc:
            modules.append({"path": path.name, "ok": False, "error": str(exc)})
            errors.append(f"Python compile failed for {path.name}: {exc}")

    notebook_reports = []
    for name in NOTEBOOKS:
        path = root / name
        if not path.exists():
            errors.append(f"missing notebook: {name}")
            continue
        report = _check_notebook(path, source_like=True)
        notebook_reports.append(report)
        errors.extend(f"{name}: {message}" for message in report["errors"])

    expected_kernels = {
        "qtp_explorer_portable.ipynb": "python3",
        "qtp_explorer.ipynb": "qtp-display",
        "qtp_exhibit.ipynb": "qtp-display",
    }
    for report in notebook_reports:
        expected = expected_kernels[report["path"]]
        actual = report["kernelspec"].get("name")
        if actual != expected:
            message = f"{report['path']}: expected kernelspec {expected!r}, found {actual!r}"
            errors.append(message)
            report["errors"].append(message)
            report["ok"] = False

    return {
        "ok": not errors,
        "python_modules": modules,
        "notebooks": notebook_reports,
        "errors": errors,
    }


def _execute_standard_mode(root: Path, *, keep_temp: bool, timeout: int) -> dict[str, Any]:
    temp_parent = Path(tempfile.mkdtemp(prefix="qtp_notebook_verification_"))
    package = temp_parent / "package"
    output_root = temp_parent / "output"
    shutil.copytree(
        root,
        package,
        ignore=shutil.ignore_patterns(
            ".mypy_cache",
            ".pytest_cache",
            ".qtp_runtime",
            ".ruff_cache",
            "display_notebook_output",
            "executed_standard_output",
            "verification_runtime",
            "__pycache__",
            "*.pyc",
        ),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    executed_name = "qtp_explorer_portable.standard_execution.ipynb"
    env = os.environ.copy()
    env.update(
        {
            "QTP_INTERACTIVE_CONTROLS_READY": "0",
            "QTP_AUTO_STANDARD_PREVIEW": "1",
            "QTP_OUTPUT_DIR": str(output_root),
            "MPLBACKEND": "Agg",
        }
    )
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "qtp_explorer_portable.ipynb",
        "--output",
        executed_name,
        f"--ExecutePreprocessor.timeout={int(timeout)}",
        "--ExecutePreprocessor.kernel_name=python3",
    ]
    standard_log = temp_parent / "standard_nbconvert.log"
    with standard_log.open("w", encoding="utf-8") as log_stream:
        proc = subprocess.run(
            command,
            cwd=package,
            env=env,
            text=True,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            timeout=timeout + 120,
        )
    standard_output = standard_log.read_text(encoding="utf-8", errors="replace")
    errors: list[str] = []
    if proc.returncode != 0:
        errors.append(f"nbconvert returned {proc.returncode}")

    executed_path = package / executed_name
    executed_report = None
    if executed_path.exists():
        executed_report = _check_notebook(executed_path, source_like=False)
        errors.extend(executed_report["errors"])
    else:
        errors.append("executed standard-preview notebook was not created")

    summaries = sorted(output_root.rglob("*_strong_qtp_summary.json"))
    pngs = sorted(output_root.rglob("*_strong_qtp.png"))
    manifests = sorted(output_root.rglob("run_manifest.csv"))
    movies = sorted(output_root.rglob("*.mp4"))
    summary_data: dict[str, Any] = {}
    if len(summaries) != 1:
        errors.append(f"expected one summary JSON, found {len(summaries)}")
    else:
        summary_data = json.loads(summaries[0].read_text())
        execution = summary_data.get("notebook_execution", {})
        status = (
            summary_data.get("certification_scope", {})
            .get("current_output_verification", {})
            .get("status")
        )
        if execution.get("mode") != "standard_notebook":
            errors.append("summary notebook_execution.mode is not 'standard_notebook'")
        if not execution.get("configuration_sha256"):
            errors.append("summary lacks notebook execution configuration digest")
        if status != "pass":
            errors.append(f"saved-output verification status is {status!r}, not 'pass'")
        if summary_data.get("claim_level") != "exploratory_run":
            errors.append("standard preview is not marked exploratory_run")
    if len(pngs) != 1:
        errors.append(f"expected one PNG, found {len(pngs)}")
    if len(manifests) != 1:
        errors.append(f"expected one run manifest, found {len(manifests)}")
    if movies:
        errors.append("standard preview unexpectedly created an MP4")

    # The opt-out path must return without starting a simulation or creating
    # run files. The notebook itself is already covered by the static checks,
    # so this path can be tested by calling the imported interface directly.
    disabled_output_root = temp_parent / "disabled_output"
    disabled_command = [
        sys.executable,
        "-c",
        "from qtp_notebook_app import show_research_app; show_research_app()",
    ]
    disabled_env = os.environ.copy()
    disabled_env.update(
        {
            "QTP_INTERACTIVE_CONTROLS_READY": "0",
            "QTP_AUTO_STANDARD_PREVIEW": "0",
            "QTP_OUTPUT_DIR": str(disabled_output_root),
            "MPLBACKEND": "Agg",
        }
    )
    disabled_proc = subprocess.run(
        disabled_command,
        cwd=package,
        env=disabled_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=min(timeout, 120),
    )
    disabled_output = disabled_proc.stdout
    disabled_errors: list[str] = []
    if disabled_proc.returncode != 0:
        disabled_errors.append(f"preview-disabled interface returned {disabled_proc.returncode}")
    disabled_artifacts = []
    if disabled_output_root.exists():
        disabled_artifacts = [p for p in disabled_output_root.rglob("*") if p.is_file()]
    if disabled_artifacts:
        disabled_errors.append(
            f"QTP_AUTO_STANDARD_PREVIEW=0 created {len(disabled_artifacts)} file(s)"
        )
    disabled_report = {
        "mode": "direct_import",
        "returncode": disabled_proc.returncode,
    }
    errors.extend(disabled_errors)

    # Build the exhibit HTML from the same module imported by the one-cell
    # museum notebook. Browser behavior is tested separately by
    # verify_browser_interfaces.py.
    exhibit_code = r"""
import json
from qtp_exhibit import exhibit_html
markup = exhibit_html(embed_media=False, autoplay=False)
checks = {
    "has_exhibit_root": 'data-qtp-exhibit="true"' in markup,
    "has_three_cases": all(f'data-case="{case}"' in markup for case in ('spot','labyrinth','stripe')),
    "has_video": '<video' in markup,
    "has_physical_time": 'physicalTimeFromFraction' in markup,
    "embedded_media_absent": 'data:video/mp4;base64' not in markup,
}
checks["ok"] = all(checks.values())
print(json.dumps(checks))
"""
    exhibit_command = [sys.executable, "-c", exhibit_code]
    exhibit_proc = subprocess.run(
        exhibit_command,
        cwd=package,
        env={**os.environ, "MPLBACKEND": "Agg"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=min(timeout, 120),
    )
    exhibit_errors: list[str] = []
    exhibit_output_summary: dict[str, Any] = {}
    if exhibit_proc.returncode != 0:
        exhibit_errors.append(f"exhibit HTML check returned {exhibit_proc.returncode}")
    else:
        try:
            exhibit_output_summary = json.loads(exhibit_proc.stdout.strip().splitlines()[-1])
        except Exception as exc:
            exhibit_errors.append(f"could not parse exhibit HTML check: {exc}")
        if exhibit_output_summary and not exhibit_output_summary.get("ok"):
            exhibit_errors.append("exhibit HTML check failed")
    exhibit_report = _check_notebook(package / "qtp_exhibit.ipynb", source_like=True)
    exhibit_errors.extend(exhibit_report["errors"])
    errors.extend(exhibit_errors)

    result = {
        "ok": not errors,
        "command": command,
        "returncode": proc.returncode,
        "stdout_tail": "\n".join(standard_output.splitlines()[-80:]),
        "temporary_directory": str(temp_parent) if keep_temp else None,
        "executed_notebook": executed_report,
        "artifacts": {
            "summary_json": [str(p.relative_to(temp_parent)) for p in summaries],
            "png": [str(p.relative_to(temp_parent)) for p in pngs],
            "run_manifest": [str(p.relative_to(temp_parent)) for p in manifests],
            "mp4": [str(p.relative_to(temp_parent)) for p in movies],
        },
        "exhibit_execution": {
            "ok": not exhibit_errors,
            "command": exhibit_command,
            "returncode": exhibit_proc.returncode,
            "stdout_tail": "\n".join(exhibit_proc.stdout.splitlines()[-40:]),
            "notebook": exhibit_report,
            "output_summary": exhibit_output_summary,
            "errors": exhibit_errors,
        },
        "disabled_execution": {
            "ok": not disabled_errors,
            "command": disabled_command,
            "returncode": disabled_proc.returncode,
            "stdout_tail": "\n".join(disabled_output.splitlines()[-40:]),
            "interface_check": disabled_report,
            "artifacts_created": [str(p.relative_to(temp_parent)) for p in disabled_artifacts],
            "errors": disabled_errors,
        },
        "summary_excerpt": {
            "case": summary_data.get("case"),
            "run_scope": summary_data.get("run_scope"),
            "claim_level": summary_data.get("claim_level"),
            "notebook_execution": summary_data.get("notebook_execution"),
            "current_output_verification": summary_data.get("certification_scope", {}).get(
                "current_output_verification"
            ),
        },
        "errors": errors,
    }
    if not keep_temp:
        shutil.rmtree(temp_parent, ignore_errors=True)
    return result


def _preview_check(root: Path, *, timeout: int) -> dict[str, Any]:
    """Exercise qualitative controls and the live-preview scope checks."""
    code = r"""
import json
import qtp_display as d
checks={}
checks['spot_strength_labels']=[label for label,_ in d.qualitative_options('spot','strength')]
checks['spot_size_labels']=[label for label,_ in d.qualitative_options('spot','size')]
checks['labyrinth_roughness_labels']=[label for label,_ in d.qualitative_options('labyrinth','roughness')]
checks['reference_values']={
    'bump_amp':d.DEFAULT_PRESETS['spot']['bump_amp'],
    'bump_width':d.DEFAULT_PRESETS['spot']['bump_width'],
    'noise_scale':d.DEFAULT_PRESETS['labyrinth']['noise_scale'],
}
runs={}
for case in ('spot','labyrinth','stripe'):
    preset=d.make_live_preview_preset(case,seed=3,bump_amp=0.09,bump_width=2.0,noise_scale=0.0075)
    preset.update(T=0.1,dt=0.05,frame_dt=0.1,L=24)
    result=d.run_case(case,run_scope='preview',**preset)
    summary=d.summarize_result(result)
    ok,detail=d.check_preview_scope(result,summary)
    runs[case]={'ok':ok,'detail':detail,'claim_level':summary.claim_level,
                'theorem_level_claimed':summary.theorem_level_claimed}
checks['runs']=runs
checks['ok']=(
    'Reference' in checks['spot_strength_labels'] and
    'Reference' in checks['spot_size_labels'] and
    'Reference' in checks['labyrinth_roughness_labels'] and
    all(item['ok'] for item in runs.values())
)
print(json.dumps(checks))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    errors = []
    payload = {}
    if proc.returncode != 0:
        errors.append(f"preview test returned {proc.returncode}")
    else:
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as exc:
            errors.append(f"could not parse preview test output: {exc}")
    if payload and not payload.get("ok"):
        errors.append("live-preview consistency check failed")
    return {
        "ok": not errors,
        "returncode": proc.returncode,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]),
        "payload": payload,
        "errors": errors,
    }


def _sanitize_public_paths(value, root: Path):
    """Remove build-machine paths from reports intended for redistribution."""
    if isinstance(value, dict):
        return {k: _sanitize_public_paths(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_public_paths(v, root) for v in value]
    if isinstance(value, str):
        text = value.replace(str(root), ".").replace(sys.executable, "{python}")
        text = re.sub(r"/tmp/qtp_notebook_verification_[^/\s]+", "<runtime>", text)
        return text
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="package root")
    parser.add_argument(
        "--execute-standard",
        action="store_true",
        help="execute the research notebook through the standard preview path",
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--preview-test",
        dest="preview_test",
        action="store_true",
        help="run a small in-memory consistency test for the live preview",
    )
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument(
        "--report",
        help="write the JSON report to this path; default: verification_runtime/notebook_package_report.json",
    )
    parser.add_argument(
        "--json-stdout",
        action="store_true",
        help="also print the complete JSON report to standard output",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "release_version": (root / "VERSION").read_text().strip()
        if (root / "VERSION").exists()
        else None,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "root": ".",
        "static": _static_checks(root),
        "standard_execution": None,
        "preview_test": None,
    }
    if args.preview_test:
        report["preview_test"] = _preview_check(root, timeout=min(args.timeout, 180))
    if args.execute_standard:
        report["standard_execution"] = _execute_standard_mode(
            root, keep_temp=args.keep_temp, timeout=args.timeout
        )
    report["ok"] = bool(
        report["static"]["ok"]
        and (report["standard_execution"] is None or report["standard_execution"]["ok"])
        and (report["preview_test"] is None or report["preview_test"]["ok"])
    )

    report = _sanitize_public_paths(report, root)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    report_path = (
        Path(args.report)
        if args.report
        else root / "verification_runtime" / "notebook_package_report.json"
    )
    if not report_path.is_absolute():
        report_path = root / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rendered + "\n")

    if args.json_stdout:
        print(rendered)
    else:
        status = "passed" if report["ok"] else "FAILED"
        print(f"notebook package verification {status}")
        print(f"  static checks: {'PASS' if report['static']['ok'] else 'FAIL'}")
        if report["standard_execution"] is not None:
            print(
                "  standard notebook mode: "
                + ("PASS" if report["standard_execution"]["ok"] else "FAIL")
            )
        if report["preview_test"] is not None:
            print(
                "  live-preview consistency: "
                + ("PASS" if report["preview_test"]["ok"] else "FAIL")
            )
        try:
            shown_path = report_path.relative_to(root)
        except ValueError:
            shown_path = report_path
        print(f"  report: {shown_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
