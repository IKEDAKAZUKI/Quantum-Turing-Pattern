#!/usr/bin/env python3
"""Reproduce the three reference patterns and their display assets."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

import qtp_display as disp

REFERENCE_CASES = (
    ("spot", "display_spot"),
    ("labyrinth", "display_labyrinth"),
    ("stripe", "display_stripe"),
)


def prepare_output_directory(path: Path, *, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path == Path.cwd().resolve():
        raise ValueError("The package directory cannot be used as the asset output directory.")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}. "
                "Choose another directory or pass --overwrite."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _progress_reporter(label: str):
    last_reported = -10

    def report(step: int, total: int, physical_time: float) -> None:
        nonlocal last_reported
        percent = 100 if total <= 0 else int(round(100 * step / total))
        bucket = min(100, 10 * (percent // 10))
        if bucket >= last_reported + 10 or step == total:
            print(f"  {label}: {bucket:3d}%  (t = {physical_time:.1f})", flush=True)
            last_reported = bucket

    return report


def reproduce_assets(output_dir: Path) -> None:
    results = []
    for case, prefix in REFERENCE_CASES:
        label = case.capitalize()
        print(f"Computing the {label} reference case...", flush=True)
        result = disp.run_case(
            case=case,
            run_scope="reference",
            progress_callback=_progress_reporter(label),
            **dict(disp.DEFAULT_PRESETS[case]),
        )
        results.append((case, prefix, result))

    pattern_vmax = max(abs(float(disp.VMIN)), abs(float(disp.VMAX)))
    for case, prefix, result in results:
        print(f"Writing figures and movies for {case.capitalize()}...", flush=True)
        figure_path = output_dir / f"{prefix}_pattern.png"
        movie_path = output_dir / f"{prefix}_pattern.mp4"
        summary_path = output_dir / f"{prefix}_pattern_summary.json"

        summary = disp.make_case_figure(result, figure_path)
        disp.save_summary(summary, summary_path, result=result)
        disp.make_case_movie(result, movie_path, pattern_vmax=pattern_vmax)
        disp.attach_movie_metadata(summary_path, movie_path)
        with Image.open(figure_path) as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((420, 260), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (420, 260), "white")
            offset = ((420 - thumbnail.width) // 2, (260 - thumbnail.height) // 2)
            canvas.paste(thumbnail, offset)
            canvas.save(
                output_dir / f"{prefix}_thumb.jpg",
                quality=86,
                optimize=True,
            )

    disp.write_exhibit_manifest(
        [result for _, _, result in results],
        output_dir,
        fps=8,
        pattern_vmin=-pattern_vmax,
        pattern_vmax=pattern_vmax,
    )
    print(f"Reference assets were written to {output_dir}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reproduction_output/display"),
        help="output directory; default: reproduction_output/display",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing nonempty output directory",
    )
    args = parser.parse_args(argv)

    try:
        output_dir = prepare_output_directory(args.out, overwrite=args.overwrite)
        reproduce_assets(output_dir)
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
