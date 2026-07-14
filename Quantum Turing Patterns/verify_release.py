#!/usr/bin/env python3
"""Verify the public display package and its integrity manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME_DIRS = {
    ".mypy_cache",
    ".pytest_cache",
    ".qtp_runtime",
    ".ruff_cache",
    "__pycache__",
    "display_notebook_output",
    "executed_standard_output",
    "reproduction_output",
    "verification_runtime",
}
GENERATED_SUFFIXES = {".pyc", ".pyo"}
FORBIDDEN_DIR_NAMES = {".git", ".idea", ".ipynb_checkpoints", ".vscode"}
FORBIDDEN_FILE_NAMES = {".DS_Store", "Thumbs.db"}
FORBIDDEN_SUFFIXES = {".bak", ".orig", ".rej", ".swo", ".swp", ".tmp"}

REQUIRED_FILES = {
    "AUTHORS.md",
    "BUILD_INFO.json",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "VERSION",
    "environment.yml",
    "requirements.txt",
    "requirements-reference.txt",
    "make_display_assets.py",
    "make_manifest.py",
    "qtp_display.py",
    "qtp_exhibit.py",
    "qtp_kernels.py",
    "qtp_movie_renderer.py",
    "qtp_notebook_app.py",
    "qtp_observables.py",
    "qtp_explorer.ipynb",
    "qtp_explorer_portable.ipynb",
    "qtp_exhibit.ipynb",
    "launch_qtp_explorer.py",
    "launch_qtp_exhibit.py",
    "start_qtp_explorer.sh",
    "start_qtp_explorer.bat",
    "start_qtp_exhibit.sh",
    "start_qtp_exhibit.bat",
    "check_qtp_environment.py",
    "verify_browser_interfaces.py",
    "verify_display.py",
    "verify_notebook_package.py",
    "verify_observable_contract.py",
    "verify_release.py",
    "MANIFEST.sha256",
    "bundle_manifest.csv",
}


def is_runtime_artifact(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    return any(part in RUNTIME_DIRS for part in rel.parts) or path.suffix in GENERATED_SUFFIXES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_version() -> str:
    raw = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", raw):
        raise ValueError(f"invalid VERSION value: {raw!r}")
    return raw


def check_metadata(version: str, errors: list[str]) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "# Quantum Turing Patterns — Interactive Display" not in readme:
        errors.append("README.md has an unexpected title")
    if f"**Release:** {version}" not in readme:
        errors.append(f"README.md does not identify release {version}")

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    version_pattern = rf'^version:\s*["\']?{re.escape(version)}["\']?\s*$'
    if not re.search(version_pattern, citation, re.MULTILINE):
        errors.append(f"CITATION.cff version is not {version}")
    if 'family-names: "Ikeda"' not in citation or 'given-names: "Kazuki"' not in citation:
        errors.append("CITATION.cff does not identify Kazuki Ikeda")
    if not re.search(r"^license:\s*MIT\s*$", citation, re.MULTILINE):
        errors.append("CITATION.cff license is not MIT")

    info = json.loads((ROOT / "BUILD_INFO.json").read_text(encoding="utf-8"))
    expected = {
        "version": version,
        "package_type": "display",
        "author": "Kazuki Ikeda",
        "license": "MIT",
    }
    for key, value in expected.items():
        if str(info.get(key)) != value:
            errors.append(f"BUILD_INFO.json {key} is {info.get(key)!r}, expected {value!r}")
    if not str(info.get("reproduction_scope", "")).strip():
        errors.append("BUILD_INFO.json does not describe the reproduction scope")
    if not isinstance(info.get("reference_generation_environment"), dict):
        errors.append("BUILD_INFO.json lacks the reference generation environment")

    authors = (ROOT / "AUTHORS.md").read_text(encoding="utf-8")
    if "Kazuki Ikeda" not in authors:
        errors.append("AUTHORS.md does not identify Kazuki Ikeda")

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    if not license_text.startswith("MIT License") or "Kazuki Ikeda" not in license_text:
        errors.append("LICENSE is not the MIT license for Kazuki Ikeda")


def check_tree(errors: list[str], *, pristine: bool) -> None:
    reported_runtime_dirs: set[Path] = set()
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)

        if path.is_symlink():
            errors.append(f"symbolic link in package tree: {rel}")
            continue
        if any(part in FORBIDDEN_DIR_NAMES for part in rel.parts):
            errors.append(f"development directory in package tree: {rel}")
            continue
        if path.is_file() and (
            path.name in FORBIDDEN_FILE_NAMES
            or path.suffix.lower() in FORBIDDEN_SUFFIXES
            or path.name.endswith("~")
        ):
            errors.append(f"temporary or editor file in package tree: {rel}")
            continue

        runtime_parts = [index for index, part in enumerate(rel.parts) if part in RUNTIME_DIRS]
        if runtime_parts:
            if pristine:
                top = Path(*rel.parts[: runtime_parts[0] + 1])
                if top not in reported_runtime_dirs:
                    errors.append(f"generated directory in package tree: {top}")
                    reported_runtime_dirs.add(top)
            continue

        if path.is_file() and path.suffix in GENERATED_SUFFIXES and pristine:
            errors.append(f"compiled Python file in package tree: {rel}")


def payload_files(*, include_manifest: bool) -> set[str]:
    excluded = {"bundle_manifest.csv"}
    if not include_manifest:
        excluded.add("MANIFEST.sha256")
    return {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.relative_to(ROOT).as_posix() not in excluded
        and not is_runtime_artifact(path)
    }


def check_manifest(errors: list[str]) -> None:
    manifest = ROOT / "MANIFEST.sha256"
    if not manifest.exists():
        errors.append("MANIFEST.sha256 is missing")
        return

    listed: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, rel = line.split("  ", 1)
        except ValueError:
            errors.append(f"invalid manifest line: {line!r}")
            continue
        listed[rel] = digest
        path = ROOT / rel
        if not path.is_file():
            errors.append(f"manifest path missing: {rel}")
        elif sha256(path) != digest:
            errors.append(f"manifest hash mismatch: {rel}")

    current = payload_files(include_manifest=False)
    missing = sorted(current - set(listed))
    extra = sorted(set(listed) - current)
    if missing:
        errors.append("files absent from MANIFEST.sha256: " + ", ".join(missing[:12]))
    if extra:
        errors.append("stale MANIFEST.sha256 entries: " + ", ".join(extra[:12]))


def check_bundle(errors: list[str]) -> None:
    manifest = ROOT / "bundle_manifest.csv"
    if not manifest.exists():
        errors.append("bundle_manifest.csv is missing")
        return

    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    listed: set[str] = set()
    for row in rows:
        rel = row.get("path", "")
        listed.add(rel)
        path = ROOT / rel
        if not path.is_file():
            errors.append(f"bundle path missing: {rel}")
            continue
        try:
            expected_size = int(row.get("size_bytes", ""))
        except ValueError:
            errors.append(f"invalid bundle size for {rel}")
            continue
        if path.stat().st_size != expected_size:
            errors.append(f"bundle size mismatch: {rel}")
        if sha256(path) != row.get("sha256"):
            errors.append(f"bundle hash mismatch: {rel}")

    current = payload_files(include_manifest=True)
    missing = sorted(current - listed)
    extra = sorted(listed - current)
    if missing:
        errors.append("files absent from bundle_manifest.csv: " + ", ".join(missing[:12]))
    if extra:
        errors.append("stale bundle_manifest.csv entries: " + ", ".join(extra[:12]))


def check_required(version: str, errors: list[str]) -> None:
    for rel in sorted(REQUIRED_FILES):
        path = ROOT / rel
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty required file: {rel}")

    display = ROOT / "display"
    for case in ("spot", "labyrinth", "stripe"):
        for suffix in (
            "strong_qtp.png",
            "strong_qtp_summary.json",
            "strong_qtp.mp4",
            "exhibit.mp4",
            "thumb.jpg",
        ):
            path = display / f"display_{case}_{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing or empty display asset: {path.relative_to(ROOT)}")
        summary_path = display / f"display_{case}_strong_qtp_summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"invalid JSON summary {summary_path.name}: {exc}")
            else:
                if str(summary.get("case")) != case:
                    errors.append(f"unexpected case value in {summary_path.name}")
                if str(summary.get("release_version")) != version:
                    errors.append(f"unexpected release version in {summary_path.name}")

    for rel in (
        "display/display_exhibit_manifest.csv",
        "display/display_movie_manifest.csv",
        "display/display_verification_report.csv",
    ):
        path = ROOT / rel
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty display record: {rel}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pristine",
        action="store_true",
        help="require a clean package tree with no generated reports, reproduced assets, or caches",
    )
    args = parser.parse_args()

    errors: list[str] = []
    try:
        version = load_version()
        check_metadata(version, errors)
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1

    check_tree(errors, pristine=args.pristine)
    check_manifest(errors)
    check_bundle(errors)
    check_required(version, errors)

    if errors:
        print("package verification FAILED")
        for error in errors:
            print("-", error)
        return 1

    mode = "pristine" if args.pristine else "routine"
    print(f"package verification passed: display {version} ({mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
