from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

MANIFEST_NAME = "MANIFEST.sha256"
BUNDLE_NAME = "bundle_manifest.csv"
RUNTIME_TOP_LEVEL = {
    ".mypy_cache",
    ".pytest_cache",
    ".qtp_runtime",
    ".ruff_cache",
    "display_notebook_output",
    "executed_standard_output",
    "reproduction_output",
    "verification_runtime",
}
RUNTIME_FILES = {"environment_check_report.json"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _package_files(root: Path, *, include_manifest: bool) -> list[Path]:
    excluded = {BUNDLE_NAME}
    if not include_manifest:
        excluded.add(MANIFEST_NAME)
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.as_posix() in excluded:
            continue
        if rel.parts and rel.parts[0] in RUNTIME_TOP_LEVEL:
            continue
        if rel.name in RUNTIME_FILES:
            continue
        if "__pycache__" in rel.parts or path.suffix == ".pyc":
            continue
        files.append(path)
    return files


def main(root: Path = Path(".")) -> None:
    root = root.resolve()

    # The text manifest binds all payload files except the two manifests.
    manifest_rows = []
    for path in _package_files(root, include_manifest=False):
        rel = path.relative_to(root).as_posix()
        manifest_rows.append((rel, sha256(path)))
    manifest_path = root / MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        for rel, digest in manifest_rows:
            stream.write(f"{digest}  {rel}\n")

    # The CSV bundle additionally binds MANIFEST.sha256 itself.
    bundle_rows = []
    for path in _package_files(root, include_manifest=True):
        rel = path.relative_to(root).as_posix()
        bundle_rows.append((rel, path.stat().st_size, sha256(path)))
    bundle_path = root / BUNDLE_NAME
    with bundle_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "size_bytes", "sha256"])
        writer.writerows(bundle_rows)

    print(
        f"wrote {len(manifest_rows)} payload hashes to {manifest_path} and "
        f"{len(bundle_rows)} bundle rows to {bundle_path}"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("."))
