from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
from PIL import Image

DIAGNOSTIC_FLAGS = [
    "no_nan",
    "physical_covariance",
    "npt_diagnostic",
    "stable_gaussian_reduction",
    "pattern_excess_positive",
    "spectral_selection_passed",
    "persistence_passed",
    "strong_qtp",
    "diagnostics_passed",
]
CERTIFIED_REFERENCE_FLAGS = DIAGNOSTIC_FLAGS + ["theorem_level_claimed"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mp4_info(path: Path):
    reader = imageio.get_reader(str(path), "ffmpeg")
    try:
        try:
            n_frames = int(reader.count_frames())
        except Exception:
            meta = reader.get_meta_data()
            n_frames = int(round(float(meta.get("duration", 0.0)) * float(meta.get("fps", 0.0))))
        meta = reader.get_meta_data()
        size = tuple(meta.get("size", (None, None)))
        fps = float(meta.get("fps", 0.0))
        duration = float(meta.get("duration", 0.0))
    finally:
        reader.close()
    return n_frames, size, fps, duration


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        try:
            return str(path.relative_to(root))
        except Exception:
            return str(path)


def _row(
    check: str,
    path: Path | str,
    ok: bool,
    observed: Any,
    expected: Any,
    tolerance: str,
    detail: str,
    root: Path,
) -> dict:
    return {
        "check": check,
        "path": _rel(Path(path), root) if not isinstance(path, str) or path else str(path),
        "ok": bool(ok),
        "observed": observed,
        "expected": expected,
        "tolerance": tolerance,
        "detail": detail,
    }


def _resolve_summaries(
    root: Path, display: Path, summary_paths: list[Path] | None, recursive: bool
) -> list[Path]:
    if summary_paths is None:
        pattern = "**/*_strong_qtp_summary.json" if recursive else "*_strong_qtp_summary.json"
        return sorted(display.glob(pattern))
    out = []
    for item in summary_paths:
        p = Path(item)
        candidates = [p] if p.is_absolute() else [root / p, display / p, display / p.name]
        for cand in candidates:
            if cand.exists():
                out.append(cand)
                break
        else:
            out.append(candidates[0])
    return out


def _compare_values(
    observed: Any,
    expected: Any,
    *,
    path: str,
    rtol: float,
    atol: float,
    mismatches: list[str],
) -> None:
    if isinstance(observed, bool) or isinstance(expected, bool):
        if observed is not expected:
            mismatches.append(f"{path}: {observed!r} != {expected!r}")
        return
    if isinstance(observed, (int, float)) and isinstance(expected, (int, float)):
        scale = max(abs(float(observed)), abs(float(expected)))
        if abs(float(observed) - float(expected)) > atol + rtol * scale:
            mismatches.append(f"{path}: {observed!r} != {expected!r}")
        return
    if isinstance(observed, dict) and isinstance(expected, dict):
        observed_keys = set(observed)
        expected_keys = set(expected)
        if observed_keys != expected_keys:
            missing = sorted(expected_keys - observed_keys)
            extra = sorted(observed_keys - expected_keys)
            if missing:
                mismatches.append(f"{path}: missing keys {missing}")
            if extra:
                mismatches.append(f"{path}: unexpected keys {extra}")
        for key in sorted(observed_keys & expected_keys):
            _compare_values(
                observed[key],
                expected[key],
                path=f"{path}.{key}",
                rtol=rtol,
                atol=atol,
                mismatches=mismatches,
            )
        return
    if isinstance(observed, list) and isinstance(expected, list):
        if len(observed) != len(expected):
            mismatches.append(f"{path}: length {len(observed)} != {len(expected)}")
            return
        for index, (left, right) in enumerate(zip(observed, expected)):
            _compare_values(
                left,
                right,
                path=f"{path}[{index}]",
                rtol=rtol,
                atol=atol,
                mismatches=mismatches,
            )
        return
    if observed != expected:
        mismatches.append(f"{path}: {observed!r} != {expected!r}")


def _reference_sections(data: dict) -> dict[str, Any]:
    scope = data.get("certification_scope", {})
    return {
        "schema_version": data.get("schema_version"),
        "case": data.get("case"),
        "mode": data.get("mode"),
        "run_scope": data.get("run_scope"),
        "claim_level": data.get("claim_level"),
        "parameters": data.get("parameters", {}),
        "initial_condition": data.get("initial_condition", {}),
        "storage_precision": data.get("storage_precision", {}),
        "grid": data.get("grid", {}),
        "movie_plan": data.get("movie_plan", {}),
        "computed_quantities": data.get("computed_quantities", {}),
        "numerical_observable_contract": data.get("numerical_observable_contract", {}),
        "persistence_diagnostics": data.get("persistence_diagnostics", {}),
        "finite_value_checks": data.get("finite_value_checks", {}),
        "baseline": data.get("baseline", {}),
        "verification_tolerances": data.get("verification_tolerances", {}),
        "verification_margins": data.get("verification_margins", {}),
        "scope": {
            "mode": scope.get("mode"),
            "run_scope": scope.get("run_scope"),
            "claim_level": scope.get("claim_level"),
            "reference_configuration_exact": scope.get("reference_configuration_exact"),
            "diagnostics_passed": scope.get("diagnostics_passed"),
            "finite_run_state": scope.get("finite_run_state"),
            "theorem_level_claimed": scope.get("theorem_level_claimed"),
            "certified_strong_qtp_claimed": scope.get("certified_strong_qtp_claimed"),
        },
        "verification": data.get("verification", {}),
    }


def _compare_reference_summary(candidate: dict, reference: dict) -> list[str]:
    mismatches: list[str] = []
    _compare_values(
        _reference_sections(candidate),
        _reference_sections(reference),
        path="summary",
        rtol=1.0e-7,
        atol=1.0e-10,
        mismatches=mismatches,
    )
    return mismatches


def _verify_exhibit_manifest(root: Path, display: Path) -> tuple[bool, list[dict]]:
    """Verify the museum-exhibit MP4 transformation and its dedicated manifest."""
    manifest_path = display / "display_exhibit_manifest.csv"
    rows: list[dict] = []
    if not manifest_path.exists():
        rows.append(
            _row(
                "display_exhibit_manifest_present",
                manifest_path,
                False,
                "absent",
                "display_exhibit_manifest.csv with spot/labyrinth/stripe rows",
                "exact",
                "museum exhibit manifest",
                root,
            )
        )
        return False, rows
    try:
        entries = list(csv.DictReader(manifest_path.open(newline="")))
    except Exception as exc:
        rows.append(
            _row(
                "display_exhibit_manifest_readable",
                manifest_path,
                False,
                f"{exc.__class__.__name__}: {exc}",
                "readable CSV",
                "exact",
                "museum exhibit manifest",
                root,
            )
        )
        return False, rows

    by_case = {str(row.get("case", "")): row for row in entries}
    expected_final = {"spot": 50.0, "labyrinth": 80.0, "stripe": 160.0}
    complete = set(by_case) == set(expected_final) and len(entries) == 3
    rows.append(
        _row(
            "display_exhibit_manifest_cases",
            manifest_path,
            complete,
            sorted(by_case),
            sorted(expected_final),
            "exact",
            "museum exhibit manifest case set",
            root,
        )
    )
    ok_all = complete
    field_scales = set()
    diagnostic_scales = set()
    for case, final_time in expected_final.items():
        row = by_case.get(case, {})
        path = display / str(row.get("filename") or f"display_{case}_exhibit.mp4")
        ok_file = path.exists() and path.stat().st_size > 0
        observed: dict[str, Any] = {"exists": ok_file}
        checks = {"exists": ok_file}
        try:
            if ok_file:
                n_frames, size, fps, duration = _mp4_info(path)
                width, height = size
                observed.update(
                    n_frames=n_frames,
                    fps=fps,
                    width=width,
                    height=height,
                    duration=duration,
                    sha256=_sha256(path),
                    size_bytes=path.stat().st_size,
                )
                expected_grid = 192 if case == "stripe" else 128
                checks.update(
                    n_frames=(n_frames == 61 and int(row.get("n_frames") or 0) == 61),
                    fps=(
                        abs(float(fps) - 8.0) <= 0.2
                        and abs(float(row.get("fps") or 0.0) - 8.0) <= 0.2
                    ),
                    dimensions=(
                        int(width) == 1440
                        and int(height) == 800
                        and int(row.get("width") or 0) == 1440
                        and int(row.get("height") or 0) == 800
                    ),
                    hash=(observed["sha256"] == row.get("sha256")),
                    size=(int(observed["size_bytes"]) == int(row.get("size_bytes") or -1)),
                    physical_time=(
                        abs(float(row.get("physical_final_time") or -1) - final_time) <= 1e-9
                    ),
                    native_diagnostic_grid=(
                        int(row.get("pattern_grid_L") or 0) == expected_grid
                        and int(row.get("diagnostic_grid_L") or 0) == expected_grid
                        and int(row.get("diagnostic_spatial_stride") or 0) == 1
                    ),
                    exhibit_gap=(int(row.get("inter_panel_extra_gap_px") or 0) >= 38),
                    y_label_spacing=(int(row.get("y_label_x_offset_px") or 0) >= 76),
                    visitor_title=(
                        row.get("visitor_right_panel_title") == "Entanglement diagnostic"
                    ),
                    times_typography=(str(row.get("font_family") or "").lower() == "times"),
                    axis_label_common_scale=(
                        str(row.get("axis_label_scale_reference") or "") == r"$x/L$"
                    ),
                    axis_label_font_size=(int(row.get("axis_label_font_size") or 0) == 19),
                )
                summary_path = display / f"display_{case}_strong_qtp_summary.json"
                if summary_path.exists():
                    plan = json.loads(summary_path.read_text()).get("movie_plan", {})
                    summary_final = float(plan.get("expected_final_time") or -1)
                    summary_frames = int(plan.get("n_frames_expected") or 0)
                    checks["summary_binding"] = (
                        abs(summary_final - final_time) <= 1e-9 and summary_frames == 61
                    )
                    observed["summary_final_time"] = summary_final
                    observed["summary_n_frames"] = summary_frames
                else:
                    checks["summary_binding"] = False
        except Exception as exc:
            observed["error"] = f"{exc.__class__.__name__}: {exc}"
            checks["readable"] = False
        for pair, target in [
            (("field_vmin", "field_vmax"), field_scales),
            (("diagnostic_vmin", "diagnostic_vmax"), diagnostic_scales),
        ]:
            try:
                target.add(tuple(float(row[key]) for key in pair))
            except Exception:
                target.add(("invalid", case))
        case_ok = all(bool(value) for value in checks.values())
        ok_all = ok_all and case_ok
        rows.append(
            _row(
                f"display_exhibit_movie_{case}",
                path,
                case_ok,
                {"manifest": row, "observed": observed, "checks": checks},
                {
                    "n_frames": 61,
                    "fps": 8,
                    "width": 1440,
                    "height": 800,
                    "physical_final_time": final_time,
                    "hash_and_size": "exact",
                    "summary_binding": True,
                    "diagnostic_grid": "native stride-1",
                    "inter_panel_extra_gap_px": ">=38",
                    "y_label_x_offset_px": ">=76",
                    "visitor_right_panel_title": "Entanglement diagnostic",
                    "font_family": "Times",
                    "axis_label_scale_reference": r"$x/L$",
                    "axis_label_font_size": 19,
                },
                "exact; fps ±0.2",
                "museum exhibit MP4 integrity and scientific time binding",
                root,
            )
        )
    common_scale_ok = len(field_scales) == 1 and len(diagnostic_scales) == 1
    rows.append(
        _row(
            "display_exhibit_common_scales",
            manifest_path,
            common_scale_ok,
            {
                "field_scales": sorted(map(str, field_scales)),
                "diagnostic_scales": sorted(map(str, diagnostic_scales)),
            },
            "one common field scale and one common diagnostic scale across all cases",
            "exact",
            "museum exhibit comparison scales",
            root,
        )
    )
    ok_all = ok_all and common_scale_ok
    return bool(ok_all), rows


def _flag_value(verification: dict, flag: str):
    if flag == "persistence_passed":
        return verification.get("persistence_passed", False)
    return verification.get(flag, False)


def _reported_flag(verification: dict, flag: str):
    return verification.get(flag)


def _infer_claim_level(data: dict) -> str:
    explicit = data.get("claim_level") or data.get("certification_scope", {}).get("claim_level")
    return str(explicit) if explicit else "unknown"


def _claim_requirements(claim_level: str, allow_exploration: bool) -> tuple[list[str], bool, str]:
    if claim_level == "theorem_level_reference":
        return DIAGNOSTIC_FLAGS + ["theorem_level_claimed"], True, "THEOREM_LEVEL_REFERENCE"
    if claim_level == "numerical_reference_demonstration":
        return DIAGNOSTIC_FLAGS, True, "NUMERICAL_REFERENCE_DEMONSTRATION"
    if claim_level == "exploratory_run":
        return ["no_nan"], bool(allow_exploration), "EXPLORATORY_RUN"
    return DIAGNOSTIC_FLAGS, False, "UNKNOWN_CLAIM_LEVEL"


def _verify_run_manifest(run_dir: Path) -> tuple[bool, str]:
    manifest_path = run_dir / "run_manifest.csv"
    if not manifest_path.exists():
        return True, "optional-absent"
    try:
        rows = list(csv.DictReader(manifest_path.open(newline="")))
    except Exception as exc:
        return False, f"could not read run_manifest.csv: {exc.__class__.__name__}: {exc}"
    checked = 0
    for row in rows:
        rel = row.get("relative_path") or row.get("file") or ""
        if not rel:
            return False, "blank manifest path entry"
        p = run_dir / rel
        if not p.exists():
            return False, f"missing {rel}"
        expected_size = row.get("size_bytes")
        if expected_size not in (None, "") and p.stat().st_size != int(expected_size):
            return False, f"size mismatch for {rel}: {p.stat().st_size} vs {expected_size}"
        expected_sha = row.get("sha256")
        if expected_sha and _sha256(p) != expected_sha:
            return False, f"sha256 mismatch for {rel}"
        checked += 1
    return True, f"optional-present; checked {checked} file hashes/sizes"


def _refresh_run_manifest(run_dir: Path):
    manifest_path = run_dir / "run_manifest.csv"
    if not manifest_path.exists():
        return
    rows = list(csv.DictReader(manifest_path.open(newline="")))
    refreshed = []
    for row in rows:
        rel = row.get("relative_path") or row.get("file") or ""
        p = run_dir / rel
        if not p.exists():
            continue
        refreshed.append(
            {
                "file": row.get("file") or p.name,
                "relative_path": rel,
                "size_bytes": p.stat().st_size,
                "sha256": _sha256(p),
            }
        )
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "relative_path", "size_bytes", "sha256"])
        w.writeheader()
        w.writerows(refreshed)


def _record_verification_state(
    js: Path,
    *,
    ok: bool,
    file_integrity_ok: bool,
    scientific_ok: bool,
    report_path: Path,
    root: Path,
    bundled_reference: bool,
):
    data = json.loads(js.read_text())
    scope = data.setdefault("certification_scope", {})
    now = datetime.now(timezone.utc).isoformat()
    state = {
        "status": "pass" if ok else "fail",
        "checked_at": now,
        "report": _rel(report_path, root),
        "file_integrity": bool(file_integrity_ok),
        "scientific_diagnostics": bool(scientific_ok),
    }
    scope["current_output_verification"] = state
    scope["current_output_verified"] = bool(ok)
    if bundled_reference:
        scope["bundled_reference_verified"] = bool(ok)
        scope["bundled_reference_verification"] = state
    js.write_text(json.dumps(data, indent=2))
    _refresh_run_manifest(js.parent)


def verify_display(
    root: Path,
    display_dir: Path | None = None,
    allow_exploration: bool = False,
    require_movie: bool = True,
    summary_paths: list[Path] | None = None,
    recursive: bool = False,
    *,
    record_state: bool = False,
    report_dir: Path | None = None,
    reference_dir: Path | None = None,
):
    root = Path(root)
    display = Path(display_dir) if display_dir is not None else root / "display"
    if not display.is_absolute():
        display = root / display
    display.mkdir(parents=True, exist_ok=True)
    if report_dir is None:
        report_dir = display if record_state else root / "verification_runtime"
    report_dir = Path(report_dir)
    if not report_dir.is_absolute():
        report_dir = root / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    if reference_dir is not None:
        reference_dir = Path(reference_dir)
        if not reference_dir.is_absolute():
            reference_dir = root / reference_dir

    rows: list[dict] = []
    manifest: list[dict] = []
    pending_states: list[dict] = []
    summaries = [
        s for s in _resolve_summaries(root, display, summary_paths, recursive) if s.exists()
    ]
    ok_all = bool(summaries)
    if not summaries:
        rows.append(
            _row(
                "display_summaries_present",
                display,
                False,
                0,
                "at least one *_strong_qtp_summary.json",
                "nonempty",
                "display JSON summaries",
                root,
            )
        )

    for js in summaries:
        run_dir = js.parent
        data = json.loads(js.read_text())
        prefix = js.name.replace("_summary.json", "")
        png = run_dir / f"{prefix}.png"
        mp4 = run_dir / f"{prefix}.mp4"
        case = str(data.get("case", prefix))
        mode = data.get("mode") or data.get("certification_scope", {}).get("mode", "certified_demo")
        run_scope = data.get("run_scope") or data.get("certification_scope", {}).get(
            "run_scope", "reference"
        )
        claim_level = _infer_claim_level(data)
        required_flags, claim_level_allowed, claim_status = _claim_requirements(
            claim_level, allow_exploration
        )
        expected_n = int(data.get("movie_plan", {}).get("n_frames_expected") or 0)
        expected_final_time = data.get("movie_plan", {}).get("expected_final_time")
        expected_fps = float(data.get("movie_plan", {}).get("fps") or 8)

        ok_png = False
        png_size = None
        png_error = ""
        try:
            if png.exists() and png.stat().st_size > 0:
                with Image.open(png) as im:
                    im.verify()
                with Image.open(png) as im:
                    png_size = f"{im.size[0]}x{im.size[1]}"
                ok_png = True
        except Exception as exc:
            png_error = f"{exc.__class__.__name__}: {exc}"

        movie_file_meta = data.get("movie_file") or {}
        movie_expected_by_json = bool(movie_file_meta)
        expected_movie_hash = movie_file_meta.get("sha256")
        expected_movie_size = movie_file_meta.get("size_bytes")
        mp4_exists = mp4.exists() and mp4.stat().st_size > 0
        movie_check_mode = (
            "required"
            if require_movie
            else ("optional-present" if mp4_exists else "optional-absent")
        )
        n_frames = width = height = fps = duration = None
        movie_last_time = None
        movie_hash_observed = ""
        movie_readable = frames_match = final_match = fps_match = hash_match = size_match = False
        movie_error = ""
        if mp4_exists:
            try:
                n_frames, size, fps, duration = _mp4_info(mp4)
                width, height = size
                movie_readable = True
                frames_match = bool(expected_n == 0 or n_frames == expected_n)
                frame_times = data.get("movie_plan", {}).get("frame_times") or []
                if frame_times and n_frames and n_frames <= len(frame_times):
                    movie_last_time = frame_times[n_frames - 1]
                final_match = bool(
                    expected_final_time is None
                    or (
                        movie_last_time is not None
                        and abs(float(movie_last_time) - float(expected_final_time)) < 1e-9
                    )
                )
                fps_match = bool(fps is not None and abs(float(fps) - expected_fps) <= 0.2)
                movie_hash_observed = _sha256(mp4)
                hash_match = bool(
                    not expected_movie_hash or movie_hash_observed == expected_movie_hash
                )
                size_match = bool(
                    expected_movie_size is None or mp4.stat().st_size == int(expected_movie_size)
                )
            except Exception as exc:
                movie_error = f"could not read MP4: {exc.__class__.__name__}: {exc}"
                movie_hash_observed = movie_error
        movie_valid = bool(
            mp4_exists
            and movie_readable
            and frames_match
            and final_match
            and fps_match
            and hash_match
            and size_match
        )
        if require_movie:
            movie_requirement_ok = movie_valid
        elif mp4_exists:
            # Optional means existence is not required; any present artifact must be valid.
            movie_requirement_ok = movie_valid
        else:
            # If JSON says a movie was written, its absence is an integrity failure.
            movie_requirement_ok = not movie_expected_by_json
        if not mp4_exists and not require_movie and not movie_expected_by_json:
            frames_status = final_status = fps_status = hash_status = "skipped"
        else:
            frames_status, final_status, fps_status = frames_match, final_match, fps_match
            hash_status = bool(hash_match and size_match)

        manifest_ok, manifest_detail = _verify_run_manifest(run_dir)
        verification = data.get("verification", {})
        scientific_all_ok = all(bool(_flag_value(verification, flag)) for flag in DIAGNOSTIC_FLAGS)
        theorem_claimed = bool(_flag_value(verification, "theorem_level_claimed"))
        required_json_ok = all(bool(_flag_value(verification, flag)) for flag in required_flags)
        if claim_level == "numerical_reference_demonstration":
            claim_consistency_ok = not theorem_claimed
        elif claim_level == "exploratory_run":
            claim_consistency_ok = bool(claim_level_allowed and not theorem_claimed)
        elif claim_level == "theorem_level_reference":
            claim_consistency_ok = theorem_claimed
        else:
            claim_consistency_ok = False
        file_integrity_ok = bool(ok_png and manifest_ok and movie_requirement_ok)
        ok = bool(
            file_integrity_ok and required_json_ok and claim_consistency_ok and claim_level_allowed
        )
        ok_all = ok_all and ok

        manifest.append(
            {
                "case": case,
                "mode": mode,
                "run_scope": run_scope,
                "claim_level": claim_level,
                "claim_status": claim_status,
                "movie_check_mode": movie_check_mode,
                "png": _rel(png, display),
                "mp4": _rel(mp4, display),
                "json": _rel(js, display),
                "expected_final_time": expected_final_time,
                "movie_last_time": movie_last_time,
                "n_frames": n_frames,
                "expected_n_frames": expected_n,
                "width": width,
                "height": height,
                "fps": fps,
                "movie_hash_ok": hash_status,
                "png_size": png_size,
                "file_integrity_ok": file_integrity_ok,
                "required_diagnostics_ok": required_json_ok,
                "scientific_diagnostics_all_ok": scientific_all_ok,
                "theorem_level_claimed": theorem_claimed,
                "claim_consistency_ok": claim_consistency_ok,
                "run_manifest_ok": manifest_ok,
                "ok": ok,
            }
        )

        rows.extend(
            [
                _row(
                    f"display_file_integrity_{case}",
                    run_dir,
                    file_integrity_ok,
                    f"png={ok_png}; png_error={png_error or 'none'}; mp4={movie_check_mode}; movie_valid={movie_valid}; movie_error={movie_error or 'none'}; frames={frames_status}; final_time={final_status}; fps={fps_status}; hash_and_size={hash_status}; run_manifest={manifest_ok}",
                    "PNG readable; MP4 may be absent only when not requested and not recorded in JSON; every present MP4 must be readable and match frames/fps/final-time/hash/size; optional manifest hashes match",
                    "exact/boolean",
                    "display file integrity",
                    root,
                ),
                _row(
                    f"display_required_diagnostics_{case}",
                    js,
                    required_json_ok,
                    {k: _reported_flag(verification, k) for k in required_flags},
                    f"{required_flags} true for claim level {claim_level}",
                    "boolean",
                    "display required JSON diagnostics for selected claim level",
                    root,
                ),
                _row(
                    f"display_scientific_diagnostics_{case}",
                    js,
                    scientific_all_ok,
                    {k: _reported_flag(verification, k) for k in DIAGNOSTIC_FLAGS},
                    "all finite-run scientific diagnostics pass",
                    "boolean/reporting",
                    f"display scientific diagnostics; claim_status={claim_status}",
                    root,
                ),
                _row(
                    f"display_claim_level_{case}",
                    js,
                    claim_consistency_ok and claim_level_allowed,
                    {"claim_level": claim_level, "theorem_level_claimed": theorem_claimed},
                    "stripe exact reference may claim theorem level; spot/labyrinth references are numerical demonstrations; exploration never claims theorem level",
                    "categorical/boolean",
                    "display claim consistency",
                    root,
                ),
                _row(
                    f"display_run_manifest_{case}",
                    run_dir / "run_manifest.csv",
                    manifest_ok,
                    manifest_detail,
                    "optional; if present all listed files match size and SHA-256",
                    "sha256/size exact",
                    "output manifest integrity",
                    root,
                ),
                _row(
                    f"display_png_{case}",
                    png,
                    ok_png,
                    png_size or png_error,
                    "exists and is readable",
                    "nonzero/readable",
                    "display PNG",
                    root,
                ),
                _row(
                    f"display_movie_{case}",
                    mp4,
                    movie_requirement_ok,
                    {
                        "mode": movie_check_mode,
                        "exists": mp4_exists,
                        "readable": movie_readable,
                        "frames_match": frames_match,
                        "final_match": final_match,
                        "fps_match": fps_match,
                        "hash_match": hash_match,
                        "size_match": size_match,
                        "error": movie_error,
                    },
                    "absent allowed only in no-require mode without movie metadata; otherwise all checks true",
                    "exact/0.2 fps",
                    "display MP4 optional/required integrity",
                    root,
                ),
            ]
        )
        for flag in CERTIFIED_REFERENCE_FLAGS:
            required = flag in required_flags
            flag_ok = bool(_flag_value(verification, flag))
            rows.append(
                _row(
                    f"display_json_{case}_{flag}",
                    js,
                    flag_ok if required else True,
                    _reported_flag(verification, flag),
                    True if required else "reported only",
                    "boolean",
                    "display diagnostic/claim flag"
                    + ("" if required else " (reported only for this claim level)"),
                    root,
                )
            )

        if reference_dir is not None:
            reference_path = reference_dir / f"display_{case}_strong_qtp_summary.json"
            if reference_path.exists():
                reference_data = json.loads(reference_path.read_text())
                mismatches = _compare_reference_summary(data, reference_data)
                reference_ok = not mismatches
                observed = "all selected numerical and metadata fields agree"
                if mismatches:
                    observed = "; ".join(mismatches[:12])
                    if len(mismatches) > 12:
                        observed += f"; ... {len(mismatches) - 12} more"
            else:
                reference_ok = False
                observed = "reference summary is missing"
            rows.append(
                _row(
                    f"display_reference_comparison_{case}",
                    js,
                    reference_ok,
                    observed,
                    _rel(reference_path, root),
                    "recursive numeric comparison: rtol=1e-7, atol=1e-10",
                    "reproduced numerical summary compared with the bundled reference",
                    root,
                )
            )
            ok_all = bool(ok_all and reference_ok)

        pending_states.append(
            {
                "js": js,
                "ok": ok,
                "file_integrity_ok": file_integrity_ok,
                "scientific_ok": scientific_all_ok,
                "bundled_reference": js.parent.resolve() == (root / "display").resolve(),
            }
        )

    complete_display_set = summary_paths is None and not recursive
    if complete_display_set:
        exhibit_ok, exhibit_rows = _verify_exhibit_manifest(root, display)
        rows.extend(exhibit_rows)
        ok_all = bool(ok_all and exhibit_ok)

    manifest_path = report_dir / "display_movie_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        fieldnames = [
            "case",
            "mode",
            "run_scope",
            "claim_level",
            "claim_status",
            "movie_check_mode",
            "png",
            "mp4",
            "json",
            "expected_final_time",
            "movie_last_time",
            "n_frames",
            "expected_n_frames",
            "width",
            "height",
            "fps",
            "movie_hash_ok",
            "png_size",
            "file_integrity_ok",
            "required_diagnostics_ok",
            "scientific_diagnostics_all_ok",
            "theorem_level_claimed",
            "claim_consistency_ok",
            "run_manifest_ok",
            "ok",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest)

    report_path = report_dir / "display_verification_report.csv"
    with report_path.open("w", newline="") as f:
        fieldnames = ["check", "path", "ok", "observed", "expected", "tolerance", "detail"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    if record_state:
        for state in pending_states:
            _record_verification_state(
                state["js"],
                ok=state["ok"],
                file_integrity_ok=state["file_integrity_ok"],
                scientific_ok=state["scientific_ok"],
                report_path=report_path,
                root=root,
                bundled_reference=state["bundled_reference"],
            )

    return ok_all, rows


def main():
    ap = argparse.ArgumentParser(
        description="Verify QTP display outputs and write a detailed report."
    )
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument(
        "--display-dir",
        type=Path,
        default=None,
        help="directory containing *_strong_qtp PNG/MP4/JSON outputs; default OUT/display",
    )
    ap.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="write runtime CSV reports here; default OUT/verification_runtime in read-only mode",
    )
    ap.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="compare numerical summaries with the corresponding files in this directory",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="search DISPLAY_DIR recursively for run-specific notebook output directories",
    )
    ap.add_argument(
        "--allow-exploration",
        action="store_true",
        help="allow exploratory outputs; they still may not claim theorem level",
    )
    ap.add_argument(
        "--record-state",
        action="store_true",
        help="update summary JSON files with the verification result and refresh any run manifests",
    )
    ap.add_argument(
        "--no-require-movie",
        action="store_true",
        help="allow an absent MP4; any MP4 that is present is still fully validated",
    )
    args = ap.parse_args()
    ok, rows = verify_display(
        args.out,
        display_dir=args.display_dir,
        report_dir=args.report_dir,
        record_state=args.record_state,
        allow_exploration=args.allow_exploration,
        require_movie=not args.no_require_movie,
        recursive=args.recursive,
        reference_dir=args.reference_dir,
    )
    file_rows = [r for r in rows if r["check"].startswith("display_file_integrity_")]
    required_rows = [r for r in rows if r["check"].startswith("display_required_diagnostics_")]
    scientific_rows = [r for r in rows if r["check"].startswith("display_scientific_diagnostics_")]
    claim_rows = [r for r in rows if r["check"].startswith("display_claim_level_")]
    comparison_rows = [
        r for r in rows if r["check"].startswith("display_reference_comparison_")
    ]
    print(f"display verification {'passed' if ok else 'failed'}")
    print(f"  mode: {'record-state' if args.record_state else 'read-only'}")
    print(
        f"  file integrity: {sum(bool(r['ok']) for r in file_rows)} / {len(file_rows)}"
    )
    print(
        f"  required diagnostics for selected claim level: {sum(bool(r['ok']) for r in required_rows)} / {len(required_rows)}"
    )
    print(
        f"  scientific diagnostics passing: {sum(bool(r['ok']) for r in scientific_rows)} / {len(scientific_rows)}"
    )
    print(f"  claim consistency: {sum(bool(r['ok']) for r in claim_rows)} / {len(claim_rows)}")
    if comparison_rows:
        print(
            "  reference comparison: "
            f"{sum(bool(r['ok']) for r in comparison_rows)} / {len(comparison_rows)}"
        )
    print(
        f"  report rows passing in selected mode: {sum(bool(r['ok']) for r in rows)} / {len(rows)}"
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
