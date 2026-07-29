#!/usr/bin/env python3
"""Verify pattern figures, movies, summaries, and exhibit records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
import numpy as np
from PIL import Image

PATTERN_FLAGS = (
    "no_nan",
    "spectral_selection_passed",
    "persistence_passed",
    "pattern_checks_passed",
)
REFERENCE_CASES = ("spot", "labyrinth", "stripe")
NUMERICAL_RTOL = 1e-7
NUMERICAL_ATOL = 1e-10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _movie_metadata(path: Path) -> dict[str, Any]:
    reader = imageio.get_reader(str(path), "ffmpeg")
    try:
        try:
            count = int(reader.count_frames())
        except Exception:
            metadata = reader.get_meta_data()
            count = int(
                round(
                    float(metadata.get("duration", 0.0))
                    * float(metadata.get("fps", 0.0))
                )
            )
        metadata = reader.get_meta_data()
        width, height = tuple(metadata.get("size", (None, None)))
        fps = float(metadata.get("fps", 0.0))
        duration = float(metadata.get("duration", 0.0))
    finally:
        reader.close()
    return {
        "n_frames": count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
    }


def _add(
    rows: list[dict[str, Any]],
    check: str,
    source: Path | str,
    ok: bool,
    observed: Any,
    expected: Any,
    tolerance: str = "exact",
    scope: str = "pattern display",
) -> None:
    rows.append(
        {
            "check": check,
            "source": str(source),
            "ok": bool(ok),
            "observed": observed,
            "expected": expected,
            "tolerance": tolerance,
            "scope": scope,
        }
    )


def _summary_paths(
    display_dir: Path,
    *,
    summary_paths: Iterable[Path] | None,
    recursive: bool,
) -> list[Path]:
    if summary_paths is not None:
        return [Path(path) for path in summary_paths]
    pattern = "**/*_pattern_summary.json" if recursive else "*_pattern_summary.json"
    return sorted(display_dir.glob(pattern))


def _scope_consistent(data: dict[str, Any], allow_exploration: bool) -> tuple[bool, dict[str, Any]]:
    case = str(data.get("case", ""))
    level = str(data.get("claim_level", ""))
    verification = data.get("verification") or {}
    theorem = bool(verification.get("theorem_level_claimed"))
    exact = bool((data.get("certification_scope") or {}).get("reference_configuration_exact"))

    if level == "theorem_level_reference":
        ok = case == "stripe" and exact and theorem
    elif level == "numerical_reference_demonstration":
        ok = case in {"spot", "labyrinth"} and exact and not theorem
    elif level == "exploratory_run":
        ok = allow_exploration and not theorem
    else:
        ok = False
    return ok, {
        "case": case,
        "claim_level": level,
        "reference_configuration_exact": exact,
        "theorem_level_claimed": theorem,
    }


def _verify_summary(
    summary_path: Path,
    *,
    allow_exploration: bool,
    require_movie: bool,
    rows: list[dict[str, Any]],
    movie_rows: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any] | None]:
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _add(
            rows,
            f"display_file_integrity_summary_{summary_path.stem}",
            summary_path,
            False,
            repr(exc),
            "valid JSON",
        )
        return False, None

    case = str(data.get("case", summary_path.stem))
    verification = data.get("verification") or {}
    level = str(data.get("claim_level", ""))
    exploratory = level == "exploratory_run"

    required_keys = {
        "schema_version",
        "case",
        "run_scope",
        "claim_level",
        "release_version",
        "parameters",
        "movie_plan",
        "computed_quantities",
        "persistence_checks",
        "finite_value_checks",
        "certification_scope",
        "verification",
    }
    missing = sorted(required_keys - set(data))
    _add(
        rows,
        f"display_file_integrity_summary_{case}",
        summary_path,
        not missing,
        {"missing": missing, "size_bytes": summary_path.stat().st_size},
        "all required sections present",
    )

    flag_values = {flag: bool(verification.get(flag)) for flag in PATTERN_FLAGS}
    required_ok = flag_values["no_nan"] and (
        exploratory and allow_exploration or all(flag_values.values())
    )
    _add(
        rows,
        f"display_pattern_checks_{case}",
        summary_path,
        required_ok,
        flag_values,
        (
            "finite values required for exploratory runs"
            if exploratory
            else "all pattern checks true"
        ),
        scope="pattern verification",
    )
    for flag, value in flag_values.items():
        expected = True if not exploratory or flag == "no_nan" else "reported"
        ok = value if expected is True else True
        _add(
            rows,
            f"display_pattern_verification_{case}_{flag}",
            summary_path,
            ok,
            value,
            expected,
            scope="pattern verification",
        )

    scope_ok, scope_state = _scope_consistent(data, allow_exploration)
    _add(
        rows,
        f"display_scope_{case}",
        summary_path,
        scope_ok,
        scope_state,
        "reference and exploratory scope rules",
        scope="result scope",
    )

    prefix = summary_path.name[: -len("_summary.json")]
    figure_path = summary_path.with_name(prefix + ".png")
    movie_path = summary_path.with_name(prefix + ".mp4")

    figure_ok = False
    figure_observed: Any = {"exists": figure_path.exists()}
    if figure_path.exists():
        try:
            with Image.open(figure_path) as image:
                image.verify()
            with Image.open(figure_path) as image:
                figure_observed = {
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "size_bytes": figure_path.stat().st_size,
                }
                figure_ok = image.width >= 800 and image.height >= 600
        except Exception as exc:
            figure_observed = repr(exc)
    _add(
        rows,
        f"display_file_integrity_png_{case}",
        figure_path,
        figure_ok,
        figure_observed,
        "readable PNG at least 800 x 600",
    )

    movie_record = data.get("movie_file") or {}
    movie_expected = bool(require_movie or movie_record)
    movie_ok = not movie_expected
    movie_observed: Any = {"exists": movie_path.exists()}
    if movie_path.exists():
        try:
            metadata = _movie_metadata(movie_path)
            plan = data.get("movie_plan") or {}
            expected_frames = int(plan.get("n_frames_expected") or metadata["n_frames"])
            expected_fps = float(plan.get("fps") or metadata["fps"])
            checks = {
                "nonempty": movie_path.stat().st_size > 0,
                "frame_count": metadata["n_frames"] == expected_frames,
                "fps": abs(metadata["fps"] - expected_fps) <= 0.2,
                "square_full_frame": metadata["width"] == metadata["height"]
                and metadata["width"] >= 960,
                "hash": not movie_record
                or _sha256(movie_path) == movie_record.get("sha256"),
                "size": not movie_record
                or movie_path.stat().st_size == int(movie_record.get("size_bytes", -1)),
                "layout": plan.get("layout") == "single_pattern_panel",
            }
            movie_ok = all(checks.values())
            movie_observed = {"metadata": metadata, "checks": checks}
            movie_rows.append(
                {
                    "case": case,
                    "mode": "pattern",
                    "filename": movie_path.name,
                    "sha256": _sha256(movie_path),
                    "size_bytes": movie_path.stat().st_size,
                    **metadata,
                    "physical_final_time": (plan.get("frame_times") or [None])[-1],
                    "layout": plan.get("layout"),
                    "ok": movie_ok,
                }
            )
        except Exception as exc:
            movie_observed = repr(exc)
            movie_ok = False
    _add(
        rows,
        f"display_file_integrity_movie_{case}",
        movie_path,
        movie_ok,
        movie_observed,
        "readable pattern-only MP4 matching summary metadata" if movie_expected else "optional",
        tolerance="exact; fps ±0.2",
    )

    return bool(not missing and required_ok and scope_ok and figure_ok and movie_ok), data


def _verify_exhibit_manifest(display_dir: Path, rows: list[dict[str, Any]]) -> bool:
    path = display_dir / "display_exhibit_manifest.csv"
    if not path.exists():
        _add(
            rows,
            "display_exhibit_manifest",
            path,
            False,
            "missing",
            "three exhibit records",
            scope="exhibit",
        )
        return False

    with path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    overall = len(records) == 3 and {row.get("case") for row in records} == set(REFERENCE_CASES)
    for record in records:
        case = str(record.get("case"))
        movie_path = display_dir / str(record.get("filename"))
        expected_fields = {
            "case",
            "filename",
            "sha256",
            "size_bytes",
            "n_frames",
            "fps",
            "width",
            "height",
            "physical_final_time",
            "field_vmin",
            "field_vmax",
            "pattern_grid_L",
            "layout",
            "font_family",
            "axis_label_font_size",
            "axis_label_scale_reference",
        }
        checks: dict[str, bool] = {
            "exists": movie_path.is_file(),
            "layout": record.get("layout") == "single_pattern_panel",
            "schema": set(record) == expected_fields,
        }
        observed: dict[str, Any] = {"manifest": record, "checks": checks}
        if movie_path.is_file():
            try:
                metadata = _movie_metadata(movie_path)
                checks.update(
                    {
                        "hash": _sha256(movie_path) == record.get("sha256"),
                        "size": movie_path.stat().st_size == int(record.get("size_bytes", -1)),
                        "n_frames": metadata["n_frames"] == int(record.get("n_frames", -1)),
                        "fps": abs(metadata["fps"] - float(record.get("fps", 0.0))) <= 0.2,
                        "dimensions": metadata["width"] == int(record.get("width", -1))
                        and metadata["height"] == int(record.get("height", -1)),
                        "square_full_frame": metadata["width"] == metadata["height"]
                        and metadata["width"] >= 960,
                    }
                )
                observed["metadata"] = metadata
            except Exception as exc:
                checks["readable"] = False
                observed["error"] = repr(exc)
        ok = all(checks.values())
        overall = overall and ok
        _add(
            rows,
            f"display_exhibit_movie_{case}",
            movie_path,
            ok,
            observed,
            "single-panel square exhibit movie with matching integrity record",
            tolerance="exact; fps ±0.2",
            scope="exhibit",
        )

    _add(
        rows,
        "display_exhibit_manifest",
        path,
        overall,
        {"records": len(records), "cases": sorted(row.get("case") for row in records)},
        "one record for each reference case",
        scope="exhibit",
    )
    return overall


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten_numeric(item, name))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            name = f"{prefix}[{index}]"
            output.update(_flatten_numeric(item, name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)
    return output


def _compare_reference(
    data: dict[str, Any],
    reference_path: Path,
    rows: list[dict[str, Any]],
) -> bool:
    case = str(data.get("case", reference_path.stem))
    if not reference_path.exists():
        _add(
            rows,
            f"display_reference_comparison_{case}",
            reference_path,
            False,
            "missing",
            "bundled reference summary",
            scope="reproduction",
        )
        return False
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    selected = {
        "parameters": data.get("parameters", {}),
        "computed_quantities": data.get("computed_quantities", {}),
        "persistence_checks": data.get("persistence_checks", {}),
        "movie_plan": {
            "frame_times": (data.get("movie_plan") or {}).get("frame_times", []),
            "field_vmin": (data.get("movie_plan") or {}).get("field_vmin"),
            "field_vmax": (data.get("movie_plan") or {}).get("field_vmax"),
        },
    }
    expected = {
        "parameters": reference.get("parameters", {}),
        "computed_quantities": reference.get("computed_quantities", {}),
        "persistence_checks": reference.get("persistence_checks", {}),
        "movie_plan": {
            "frame_times": (reference.get("movie_plan") or {}).get("frame_times", []),
            "field_vmin": (reference.get("movie_plan") or {}).get("field_vmin"),
            "field_vmax": (reference.get("movie_plan") or {}).get("field_vmax"),
        },
    }
    actual_values = _flatten_numeric(selected)
    expected_values = _flatten_numeric(expected)
    common = sorted(set(actual_values) & set(expected_values))
    differences = {
        key: abs(actual_values[key] - expected_values[key])
        for key in common
        if not np.isclose(
            actual_values[key],
            expected_values[key],
            rtol=NUMERICAL_RTOL,
            atol=NUMERICAL_ATOL,
        )
    }
    missing = sorted(set(expected_values) - set(actual_values))
    ok = bool(common and not differences and not missing)
    _add(
        rows,
        f"display_reference_comparison_{case}",
        reference_path,
        ok,
        {
            "compared_values": len(common),
            "max_absolute_difference": max(
                (abs(actual_values[key] - expected_values[key]) for key in common),
                default=0.0,
            ),
            "differences": differences,
            "missing": missing,
        },
        f"rtol={NUMERICAL_RTOL}, atol={NUMERICAL_ATOL}",
        tolerance=f"rtol={NUMERICAL_RTOL}; atol={NUMERICAL_ATOL}",
        scope="reproduction",
    )
    return ok



def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _refresh_run_manifest(run_dir: Path) -> None:
    manifest_path = run_dir / "run_manifest.csv"
    if not manifest_path.exists():
        return
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    refreshed: list[dict[str, Any]] = []
    for record in records:
        relative = record.get("relative_path") or record.get("file") or ""
        path = run_dir / relative
        if not path.is_file():
            continue
        refreshed.append(
            {
                "file": record.get("file") or path.name,
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["file", "relative_path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(refreshed)


def _record_verification_state(
    summary_path: Path,
    *,
    ok: bool,
    report_path: Path,
    root: Path,
    bundled_reference: bool,
) -> None:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    scope = data.setdefault("certification_scope", {})
    state = {
        "status": "pass" if ok else "fail",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "report": _relative_path(report_path, root),
        "file_integrity": bool(ok),
        "pattern_checks": bool(ok),
    }
    scope["current_output_verification"] = state
    scope["current_output_verified"] = bool(ok)
    if bundled_reference:
        scope["bundled_reference_verification"] = state.copy()
        scope["bundled_reference_verified"] = bool(ok)
    summary_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    _refresh_run_manifest(summary_path.parent)


def _portable_sources(rows: list[dict[str, Any]], root: Path) -> None:
    for row in rows:
        raw = str(row.get("source", ""))
        path = Path(raw)
        if not path.is_absolute():
            continue
        try:
            row["source"] = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            row["source"] = path.name

def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def verify_display(
    root,
    *,
    display_dir=None,
    allow_exploration: bool = False,
    require_movie: bool = True,
    summary_paths=None,
    recursive: bool = False,
    record_state: bool = False,
    report_dir=None,
    reference_dir=None,
):
    root = Path(root).resolve()
    display_dir = Path(display_dir or root / "display").resolve()
    report_dir = Path(report_dir or root / "verification_runtime").resolve()
    reference_dir = Path(reference_dir).resolve() if reference_dir else None

    rows: list[dict[str, Any]] = []
    movie_rows: list[dict[str, Any]] = []
    summaries = _summary_paths(
        display_dir,
        summary_paths=summary_paths,
        recursive=recursive,
    )
    if not summaries:
        _add(
            rows,
            "display_summary_set",
            display_dir,
            False,
            0,
            "at least one *_pattern_summary.json",
        )

    overall = bool(summaries)
    parsed: list[tuple[Path, dict[str, Any], bool]] = []
    for summary_path in summaries:
        ok, data = _verify_summary(
            summary_path,
            allow_exploration=allow_exploration,
            require_movie=require_movie,
            rows=rows,
            movie_rows=movie_rows,
        )
        overall = overall and ok
        if data is not None:
            parsed.append((summary_path, data, ok))

    bundled_set = display_dir.name == "display" and {
        data.get("case") for _, data, _ in parsed
    } == set(REFERENCE_CASES)
    if bundled_set:
        overall = _verify_exhibit_manifest(display_dir, rows) and overall

    if reference_dir is not None:
        for summary_path, data, _summary_ok in parsed:
            case = str(data.get("case"))
            reference_path = reference_dir / f"display_{case}_pattern_summary.json"
            overall = _compare_reference(data, reference_path, rows) and overall

    _portable_sources(rows, root)
    _write_csv(
        report_dir / "display_verification_report.csv",
        rows,
        ["check", "source", "ok", "observed", "expected", "tolerance", "scope"],
    )
    _write_csv(
        report_dir / "display_movie_manifest.csv",
        movie_rows,
        [
            "case",
            "mode",
            "filename",
            "sha256",
            "size_bytes",
            "n_frames",
            "fps",
            "width",
            "height",
            "duration",
            "physical_final_time",
            "layout",
            "ok",
        ],
    )
    final_ok = bool(overall and all(bool(row["ok"]) for row in rows))
    if record_state:
        report_path = report_dir / "display_verification_report.csv"
        for summary_path, _data, summary_ok in parsed:
            _record_verification_state(
                summary_path,
                ok=bool(summary_ok),
                report_path=report_path,
                root=root,
                bundled_reference=bundled_set,
            )
    return final_ok, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("."), help="package root")
    parser.add_argument(
        "--display-dir",
        type=Path,
        default=None,
        help="directory containing pattern PNG, MP4, and JSON files",
    )
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--allow-exploration", action="store_true")
    parser.add_argument("--no-require-movie", action="store_true")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--record-state",
        action="store_true",
        help="record the verification result in each summary",
    )
    args = parser.parse_args(argv)

    root = args.out.resolve()
    display_dir = args.display_dir or root / "display"
    report_dir = args.report_dir or root / "verification_runtime"
    ok, rows = verify_display(
        root,
        display_dir=display_dir,
        allow_exploration=args.allow_exploration,
        require_movie=not args.no_require_movie,
        recursive=args.recursive,
        record_state=args.record_state,
        report_dir=report_dir,
        reference_dir=args.reference_dir,
    )
    passed = sum(bool(row["ok"]) for row in rows)
    print(f"pattern display verification: {passed}/{len(rows)} PASS")
    print(f"report: {Path(report_dir) / 'display_verification_report.csv'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
