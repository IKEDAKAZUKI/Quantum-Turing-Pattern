"""Interactive notebook interface for exploring and exporting QTP simulations."""

import hashlib
import html
import json
import os
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path

from IPython.display import HTML, Markdown, Video, display

import qtp_display as disp
import verify_display

ROOT = Path(__file__).resolve().parent
DISPLAY_DIR = ROOT / "display"
OUTPUT_DIR = Path(
    os.environ.get("QTP_OUTPUT_DIR", str(ROOT / "display_notebook_output"))
).expanduser()


INTERACTIVE_CONTROLS_READY = os.environ.get("QTP_INTERACTIVE_CONTROLS_READY") == "1"


def _show_json_details(summary_path):
    data = Path(summary_path).read_text()
    display(
        HTML(
            "<details><summary>Show run details (JSON)</summary>"
            '<pre style="white-space:pre-wrap;font-size:11px;">'
            + html.escape(data)
            + "</pre></details>"
        )
    )
    href = html.escape(_relative_href(summary_path))
    display(HTML(f'Open run details (JSON): <a href="{href}" target="_blank">{href}</a>'))


def _make_run_dir(case, preset, base=OUTPUT_DIR):
    payload = json.dumps(preset, sort_keys=True, default=str).encode()
    tag = hashlib.sha1(payload).hexdigest()[:8]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base) / f"{case}_{stamp}_{tag}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _make_thumbnail(source_png, thumb_path, size=(360, 240)):
    try:
        from PIL import Image as PILImage

        with PILImage.open(source_png) as image:
            image = image.convert("RGB")
            image.thumbnail(size)
            image.save(thumb_path, format="JPEG", quality=82, optimize=True)
        return Path(thumb_path)
    except Exception:
        return None


def _cleanup_recent_runs(base=OUTPUT_DIR, keep_latest=None, max_age_hours=24):
    keep_latest = (
        int(os.environ.get("QTP_MAX_RUNS", "20")) if keep_latest is None else int(keep_latest)
    )
    base = Path(base)
    if not base.exists():
        return []
    now = datetime.now().timestamp()
    runs = sorted(
        [p for p in base.glob("*_*_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    removed = []
    for index, path in enumerate(runs):
        if (path / ".preserve").exists():
            continue
        age_hours = (now - path.stat().st_mtime) / 3600.0
        if index >= keep_latest or age_hours > max_age_hours:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
    return removed


def _relative_href(path):
    path = Path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _show_recent_runs(base=OUTPUT_DIR, limit=5):
    base = Path(base)
    runs = sorted(
        [p for p in base.glob("*_*_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]
    if not runs:
        return

    def link(path, label):
        if path is None:
            return ""
        return f'<a href="{html.escape(_relative_href(path))}">{html.escape(label)}</a>'

    rows = []
    for path in runs:
        thumb = next(iter(sorted(path.glob("*_thumb.jpg"))), None)
        png = next(iter(sorted(path.glob("*_strong_qtp.png"))), None)
        summary = next(iter(sorted(path.glob("*_strong_qtp_summary.json"))), None)
        movie = next(iter(sorted(path.glob("*_strong_qtp.mp4"))), None)
        manifest = path / "run_manifest.csv" if (path / "run_manifest.csv").exists() else None
        if thumb is not None:
            preview = (
                f'<img src="{html.escape(_relative_href(thumb))}" '
                f'alt="Thumbnail for {html.escape(path.name)}" '
                'style="width:180px;max-height:120px;object-fit:cover;'
                'border:1px solid #ddd;border-radius:6px;">'
            )
        else:
            preview = '<span style="color:#777;">no thumbnail</span>'
        links = " | ".join(
            filter(
                None,
                [
                    link(png, "PNG"),
                    link(summary, "JSON"),
                    link(movie, "MP4"),
                    link(manifest, "manifest"),
                ],
            )
        )
        rows.append(
            "<tr>"
            f'<td style="padding:8px;vertical-align:top;">{preview}</td>'
            f'<td style="padding:8px;vertical-align:top;"><code>{html.escape(path.name)}</code><br>'
            f'<span style="font-size:12px;color:#555;">{links}</span></td>'
            "</tr>"
        )
    display(
        HTML(
            "<b>Recent runs</b>"
            '<div style="font-size:12px;color:#555;">The gallery references lightweight JPEG thumbnails; '
            "full-resolution PNG files are not embedded in notebook outputs.</div>"
            '<table style="border-collapse:collapse;margin-top:6px;">' + "".join(rows) + "</table>"
        )
    )


def _show_time_scrubber(result, *, start_at_zero=False):
    """Display the precomputed time evolution in the browser."""
    scale = disp.reference_diagnostic_vmax(ROOT)
    display(
        HTML(
            "<b>Pattern formation through time</b><br>"
            '<span style="font-size:12px;color:#555;">Replay, pause, seek, change speed, or loop. '
            "The frames are prepared once, so playback does not rerun the simulation.</span>"
        )
    )
    display(
        HTML(
            disp.client_side_time_explorer_html(
                result,
                diagnostic_vmax=scale,
                initial="zero" if start_at_zero else "final",
            )
        )
    )


def _verification_summary_markdown(label, ok, rows):
    groups = {
        "Saved-file integrity": "display_file_integrity_",
        "Required diagnostics for selected scope": "display_required_diagnostics_",
        "Scientific diagnostics passing": "display_scientific_diagnostics_",
        "Result-scope checks": "display_claim_level_",
    }
    lines = [f"**{label}:** {'PASS' if ok else 'FAIL'}  "]
    for title, prefix in groups.items():
        selected = [row for row in rows if str(row.get("check", "")).startswith(prefix)]
        if selected:
            passed = sum(bool(row.get("ok")) for row in selected)
            lines.append(f"- {title}: {passed}/{len(selected)}  ")
    return "\n".join(lines)


def _redact_runtime_paths(text):
    text = str(text).replace(str(ROOT), ".")
    # Remove temporary paths from user-visible error messages.
    import re

    text = re.sub(r"/tmp/qtp_[^/\s]+", "<temporary-directory>", text)
    return text


def _error_panel(message, technical_details):
    technical_details = _redact_runtime_paths(technical_details)
    return HTML(
        '<div style="border-left:5px solid #a33;padding:10px 14px;background:#fff3f3;">'
        f"<b>{html.escape(message)}</b><br>"
        "The previous result remains available. Check the selected settings and try again."
        "<details><summary>Error details</summary>"
        f'<pre style="white-space:pre-wrap;font-size:11px;">{html.escape(technical_details)}</pre>'
        "</details></div>"
    )


def run_display(
    case="spot",
    preset=None,
    make_movie=True,
    display_dir=OUTPUT_DIR,
    progress_callback=None,
    run_scope="reference",
    execution_mode="programmatic",
):
    preset = dict(preset or disp.DEFAULT_PRESETS[case])
    base_dir = Path(display_dir)
    display_dir = _make_run_dir(case, preset, base=base_dir) if base_dir == OUTPUT_DIR else base_dir
    display_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{case}_notebook"
    figure_path = display_dir / f"{prefix}_strong_qtp.png"
    thumbnail_path = display_dir / f"{prefix}_thumb.jpg"
    movie_path = display_dir / f"{prefix}_strong_qtp.mp4"
    summary_path = display_dir / f"{prefix}_strong_qtp_summary.json"

    t0 = time.perf_counter()
    display(Markdown(f"Preparing the {case.capitalize()} initial condition..."))
    result = disp.run_case(case, progress_callback=progress_callback, run_scope=run_scope, **preset)
    result["run_scope"] = run_scope
    execution_payload = {
        "case": case,
        "preset": preset,
        "run_scope": run_scope,
        "make_movie": bool(make_movie),
    }
    result["notebook_execution"] = {
        "mode": str(execution_mode),
        "output_directory": Path(display_dir).name,
        "configuration_sha256": hashlib.sha256(
            json.dumps(execution_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
    }
    display(
        Markdown(
            f"Simulation complete: {len(result['times'])} saved frames. "
            "Computing the final diagnostic and preparing the dashboard..."
        )
    )
    summary = disp.make_case_figure(
        result,
        figure_path,
        title=f"{case.capitalize()}: pattern and finite-region NPT diagnostic",
    )
    disp.save_summary(summary, summary_path, result=result)
    thumbnail = _make_thumbnail(figure_path, thumbnail_path)

    display(
        HTML(
            f'<img src="{html.escape(_relative_href(figure_path))}" alt="{html.escape(case.capitalize())} pattern field, finite-region NPT diagnostic, spectrum, and finite-run status dashboard" style="max-width:100%;height:auto;">'
        )
    )
    display(HTML(disp.status_cards_html(summary)))
    display(
        Markdown(
            f"**Scope of this result:** {disp.claim_label(summary)}.  "
            "Numerical checks, theorem scope, and file integrity are reported separately."
        )
    )
    href = html.escape(_relative_href(figure_path))
    display(HTML(f'Open PNG: <a href="{href}" target="_blank">{href}</a>'))

    movie_written = False
    if make_movie:
        try:
            display(Markdown("Writing the MP4..."))
            diagnostic_vmax = float(result["E_frames"].max())
            disp.make_case_movie(
                result,
                movie_path,
                diagnostic_vmax=diagnostic_vmax,
                diagnostic_scale_scope="per_case_notebook_movie",
            )
            disp.attach_movie_metadata(
                summary_path,
                movie_path,
                diagnostic_vmax=diagnostic_vmax,
                diagnostic_scale_scope="per_case_notebook_movie",
            )
            movie_written = True
            href = html.escape(_relative_href(movie_path))
            display(HTML(f'Open MP4: <a href="{href}" target="_blank">{href}</a>'))
        except Exception:
            display(_error_panel("The movie export did not complete.", traceback.format_exc()))

    manifest_files = [figure_path, summary_path]
    if thumbnail is not None:
        manifest_files.append(thumbnail)
    if movie_written:
        manifest_files.append(movie_path)
    manifest_path = disp.write_run_manifest(display_dir, manifest_files)
    if manifest_path is not None:
        href = html.escape(_relative_href(manifest_path))
        display(HTML(f'Open run manifest: <a href="{href}" target="_blank">{href}</a>'))

    try:
        ok, rows = verify_display.verify_display(
            ROOT,
            display_dir=display_dir,
                allow_exploration=True,
            require_movie=bool(make_movie),
            summary_paths=[summary_path],
            recursive=True,
            record_state=True,
            report_dir=display_dir,
        )
        display(Markdown(_verification_summary_markdown("Saved-output verification", ok, rows)))
    except Exception:
        display(
            _error_panel(
                "The saved files could not be verified.", traceback.format_exc()
            )
        )

    _show_json_details(summary_path)
    if make_movie and movie_written:
        display(Video(str(movie_path), embed=False))
    _show_time_scrubber(result)
    _cleanup_recent_runs()
    _show_recent_runs()
    elapsed = time.perf_counter() - t0
    display(
        Markdown(
            f"Done in {elapsed:.1f} seconds. Output directory: `{_relative_href(display_dir)}`"
        )
    )
    return result, summary


DEFAULT_PREVIEW_CASE = "spot"  # 'spot', 'labyrinth', or 'stripe'
DEFAULT_PREVIEW_QUALITY = "preview"  # 'preview' or 'reference'
DEFAULT_PREVIEW_SEED = 4
DEFAULT_PREVIEW_BUMP_AMP = 0.12
DEFAULT_PREVIEW_BUMP_WIDTH = 2.4
DEFAULT_PREVIEW_NOISE_SCALE = 1e-2
DEFAULT_PREVIEW_MAKE_MOVIE = False

# Standard preview settings used when interactive controls are unavailable.
AUTO_STANDARD_PREVIEW = os.environ.get("QTP_AUTO_STANDARD_PREVIEW", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
STANDARD_PREVIEW_ONCE_PER_CONFIGURATION = True


def standard_preview_configuration():
    return {
        "case": str(DEFAULT_PREVIEW_CASE),
        "quality": str(DEFAULT_PREVIEW_QUALITY),
        "seed": int(DEFAULT_PREVIEW_SEED),
        "bump_amp": float(DEFAULT_PREVIEW_BUMP_AMP),
        "bump_width": float(DEFAULT_PREVIEW_BUMP_WIDTH),
        "noise_scale": float(DEFAULT_PREVIEW_NOISE_SCALE),
        "make_movie": bool(DEFAULT_PREVIEW_MAKE_MOVIE),
    }


def run_standard_preview(
    case=DEFAULT_PREVIEW_CASE,
    quality=DEFAULT_PREVIEW_QUALITY,
    seed=DEFAULT_PREVIEW_SEED,
    bump_amp=DEFAULT_PREVIEW_BUMP_AMP,
    bump_width=DEFAULT_PREVIEW_BUMP_WIDTH,
    noise_scale=DEFAULT_PREVIEW_NOISE_SCALE,
    make_movie=DEFAULT_PREVIEW_MAKE_MOVIE,
    execution_mode="standard_notebook",
):
    if case not in disp.DEFAULT_PRESETS:
        raise ValueError(f"Unknown case: {case}")
    if quality not in {"preview", "reference"}:
        raise ValueError("quality must be 'preview' or 'reference'")
    preset = dict(disp.DEFAULT_PRESETS[case])
    if quality == "preview":
        preset.update(
            disp.make_live_preview_preset(
                case,
                seed=seed,
                bump_amp=bump_amp,
                bump_width=bump_width,
                noise_scale=noise_scale,
            )
        )
        preset["L"] = 96 if case != "stripe" else 120
    if case == "spot":
        preset["seed"] = int(seed)
        preset["bump_amp"] = float(bump_amp)
        preset["bump_width"] = float(bump_width)
    elif case == "labyrinth":
        preset["seed"] = int(seed)
        preset["noise_scale"] = float(noise_scale)
    return run_display(
        case,
        preset=preset,
        make_movie=bool(make_movie),
        run_scope=quality,
        execution_mode=execution_mode,
    )


def run_standard_preview_current_config(execution_mode="standard_notebook"):
    config = standard_preview_configuration()
    config["execution_mode"] = execution_mode
    return run_standard_preview(**config)


def reset_standard_preview():
    """Allow the standard preview to run again for the current configuration."""
    globals().pop("_QTP_STANDARD_PREVIEW_STATE", None)


def show_research_app():
    """Display the research interface or run the standard preview."""
    global QTP_STANDARD_PREVIEW_RESULT, QTP_STANDARD_PREVIEW_SUMMARY
    import asyncio
    import secrets

    try:
        import ipywidgets as widgets
        from IPython.display import clear_output

        controls_imported = True
    except Exception:
        controls_imported = False

    CONTROLS_AVAILABLE = bool(controls_imported and INTERACTIVE_CONTROLS_READY)

    if CONTROLS_AVAILABLE:
        header = widgets.HTML(
            "<h2>Quantum Turing Patterns — Research Explorer</h2>"
            "<p>Choose a pattern and adjust the controls. A quick preview updates after each change. "
            "Use the save buttons when you want to keep a full-resolution result or movie.</p>"
            "<p><b>No numerical entry is required.</b> Exact values and the random seed are available "
            "under Advanced settings and are included with saved results.</p>"
        )
        selected_case = {"value": "spot"}
        case_buttons = {
            "spot": widgets.Button(description="Spot", tooltip="Localized multi-bump initial data"),
            "labyrinth": widgets.Button(
                description="Labyrinth", tooltip="Random-noise initial data"
            ),
            "stripe": widgets.Button(
                description="Stripe", tooltip="Period-12 theorem reference"
            ),
        }
        case_subtitles = {
            "spot": "localized islands and spots",
            "labyrinth": "interlaced maze-like domains",
            "stripe": "commensurate stripe branch",
        }
        case_cards = []
        for case_name in ("spot", "labyrinth", "stripe"):
            button = case_buttons[case_name]
            button.layout = widgets.Layout(width="100%", height="46px")
            thumb = DISPLAY_DIR / f"display_{case_name}_thumb.jpg"
            thumb_html = (
                f'<img src="{html.escape(_relative_href(thumb))}" alt="{case_name.capitalize()} morphology thumbnail" '
                'style="width:100%;height:110px;object-fit:cover;border-radius:7px;margin-bottom:5px;">'
                if thumb.exists()
                else ""
            )
            case_cards.append(
                widgets.VBox(
                    [
                        widgets.HTML(thumb_html),
                        button,
                        widgets.HTML(
                            f'<div style="font-size:12px;color:#555;text-align:center;">'
                            f"{case_subtitles[case_name]}</div>"
                        ),
                    ],
                    layout=widgets.Layout(border="1px solid #ddd", padding="8px"),
                )
            )

        bump_widget = widgets.SelectionSlider(
            options=list(disp.qualitative_options("spot", "strength")),
            value=disp.DEFAULT_PRESETS["spot"]["bump_amp"],
            description="Initial bump strength",
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="100%"),
        )
        bump_width_widget = widgets.SelectionSlider(
            options=list(disp.qualitative_options("spot", "size")),
            value=disp.DEFAULT_PRESETS["spot"]["bump_width"],
            description="Initial bump size",
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="100%"),
        )
        noise_widget = widgets.SelectionSlider(
            options=list(disp.qualitative_options("labyrinth", "roughness")),
            value=disp.DEFAULT_PRESETS["labyrinth"]["noise_scale"],
            description="Initial roughness",
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="100%"),
        )
        seed_widget = widgets.IntText(
            value=disp.CASE_DEFAULT_SEEDS["spot"],
            description="Random seed",
            style={"description_width": "initial"},
        )
        quality_widget = widgets.ToggleButtons(
            options=[("Quick export", "preview"), ("Full-resolution export", "reference")],
            value="preview",
            description="Export quality",
            style={"description_width": "initial"},
        )
        live_preview_widget = widgets.Checkbox(
            value=True,
            description="Update live preview automatically",
            indent=False,
        )
        scope_info = widgets.HTML(
            '<div style="border-left:4px solid #789;padding:7px 10px;margin:4px 0;background:#f6f8fb;">'
            "<b>Live preview:</b> a quick, low-resolution calculation that is not saved. "
            "<b>Save result:</b> creates the full dashboard, run details, verification report, and optional movie. "
            "The replay control shows the formation process from the initial state.</div>"
        )
        run_button = widgets.Button(
            description="Save result", button_style="success", icon="save"
        )
        run_movie_button = widgets.Button(
            description="Save result + movie", button_style="warning", icon="film"
        )
        restore_button = widgets.Button(description="Restore reference", icon="refresh")
        randomize_button = widgets.Button(description="New realization", icon="random")
        progress = widgets.IntProgress(value=0, min=0, max=100, description="Progress")
        status = widgets.HTML("Ready")
        log = widgets.Output()
        live_output = widgets.Output(
            layout=widgets.Layout(border="1px solid #ddd", padding="8px", width="100%")
        )

        exact_values = widgets.HTML()
        advanced = widgets.Accordion(
            children=[
                widgets.VBox(
                    [
                        seed_widget,
                        exact_values,
                        widgets.HTML(
                            "<small>The qualitative sliders map to exact deterministic values. "
                            "The selected values and random seed are included with saved results.</small>"
                        ),
                    ]
                )
            ]
        )
        advanced.set_title(0, "Advanced settings and exact values")

        all_controls = list(case_buttons.values()) + [
            seed_widget,
            bump_widget,
            bump_width_widget,
            noise_widget,
            quality_widget,
            live_preview_widget,
            run_button,
            run_movie_button,
            restore_button,
            randomize_button,
        ]
        preview_state = {"task": None, "token": 0, "running": False, "pending": False}

        def set_busy(flag):
            for control in all_controls:
                control.disabled = flag

        def update_case_styles():
            for name, button in case_buttons.items():
                button.button_style = "info" if name == selected_case["value"] else ""

        def update_visibility(*_):
            case = selected_case["value"]
            seed_widget.layout.display = "" if case in {"spot", "labyrinth"} else "none"
            bump_widget.layout.display = "" if case == "spot" else "none"
            bump_width_widget.layout.display = "" if case == "spot" else "none"
            noise_widget.layout.display = "" if case == "labyrinth" else "none"
            randomize_button.layout.display = "" if case in {"spot", "labyrinth"} else "none"

        def update_exact_values(*_):
            case = selected_case["value"]
            if case == "spot":
                text = (
                    f"<small><b>Exact current values:</b> seed={int(seed_widget.value)}, "
                    f"bump amplitude={float(bump_widget.value):.4g}, "
                    f"bump width={float(bump_width_widget.value):.4g} lattice sites.</small>"
                )
            elif case == "labyrinth":
                text = (
                    f"<small><b>Exact current values:</b> seed={int(seed_widget.value)}, "
                    f"noise scale={float(noise_widget.value):.4g}.</small>"
                )
            else:
                text = "<small><b>Exact current values:</b> period-compatible stripe reference preset.</small>"
            exact_values.value = text

        def restore_reference_preset(*_, schedule=True):
            case = selected_case["value"]
            if case in disp.CASE_DEFAULT_SEEDS:
                seed_widget.value = disp.CASE_DEFAULT_SEEDS[case]
            bump_widget.value = disp.DEFAULT_PRESETS["spot"]["bump_amp"]
            bump_width_widget.value = disp.DEFAULT_PRESETS["spot"]["bump_width"]
            noise_widget.value = disp.DEFAULT_PRESETS["labyrinth"]["noise_scale"]
            update_visibility()
            update_exact_values()
            if schedule:
                schedule_live_preview()

        def select_case(case):
            selected_case["value"] = case
            update_case_styles()
            restore_reference_preset(schedule=False)
            schedule_live_preview()

        for name, button in case_buttons.items():
            button.on_click(lambda _, case=name: select_case(case))

        def preset_from_controls():
            case = selected_case["value"]
            preset = dict(disp.DEFAULT_PRESETS[case])
            run_scope = quality_widget.value
            if run_scope == "preview":
                preview_preset = disp.make_live_preview_preset(
                    case,
                    seed=int(seed_widget.value),
                    bump_amp=float(bump_widget.value),
                    bump_width=float(bump_width_widget.value),
                    noise_scale=float(noise_widget.value),
                )
                preset.update(preview_preset)
                preset["L"] = 96 if case != "stripe" else 120
            if case == "spot":
                preset["seed"] = int(seed_widget.value)
                preset["bump_amp"] = float(bump_widget.value)
                preset["bump_width"] = float(bump_width_widget.value)
            elif case == "labyrinth":
                preset["seed"] = int(seed_widget.value)
                preset["noise_scale"] = float(noise_widget.value)
            return case, preset, run_scope

        def live_preview_preset_from_controls():
            """Return the settings for the current preview."""
            case = selected_case["value"]
            preset = disp.make_live_preview_preset(
                case,
                seed=int(seed_widget.value),
                bump_amp=float(bump_widget.value),
                bump_width=float(bump_width_widget.value),
                noise_scale=float(noise_widget.value),
            )
            return case, preset

        def _display_live_snapshot(result, summary):
            """Display the preview and its time evolution."""
            display(HTML(disp.status_cards_html(summary)))
            _show_time_scrubber(result, start_at_zero=False)

        def render_live_preview(request_token=None):
            if not live_preview_widget.value:
                return
            request_token = preview_state["token"] if request_token is None else request_token
            preview_state["running"] = True
            preview_state["pending"] = False
            try:
                with live_output:
                    clear_output(wait=True)
                    case, preset = live_preview_preset_from_controls()
                    started = time.perf_counter()
                    display(
                        HTML(
                            '<div style="border-left:5px solid #2f6f9f;padding:8px 12px;background:#eef7ff;">'
                            "<b>Updating preview...</b> This quick preview is not saved.</div>"
                        )
                    )
                    result = disp.run_case(case, run_scope="preview", **preset)
                    summary = disp.summarize_result(result)
                    scope_ok, scope_checks = disp.check_preview_scope(result, summary)
                    if not scope_ok:
                        raise RuntimeError(
                            f"preview scope check failed: {scope_checks}"
                        )
                    clear_output(wait=True)
                    display(
                        HTML(
                            f'<div style="border-left:5px solid #2f6f9f;padding:8px 12px;background:#eef7ff;">'
                            f"<b>Live preview updated in {time.perf_counter() - started:.1f} s.</b> "
                            "Move a control again to update it, or use Replay formation to follow the time evolution.</div>"
                        )
                    )
                    _display_live_snapshot(result, summary)
            except Exception:
                with live_output:
                    clear_output(wait=True)
                    display(
                        _error_panel("The live preview did not complete.", traceback.format_exc())
                    )
            finally:
                preview_state["running"] = False
                newer_request = request_token != preview_state["token"]
                if live_preview_widget.value and (preview_state["pending"] or newer_request):
                    preview_state["pending"] = False
                    schedule_live_preview()

        def cancel_preview_task():
            task = preview_state.get("task")
            if task is not None and not task.done():
                task.cancel()
            preview_state["task"] = None

        def schedule_live_preview(*_):
            if not live_preview_widget.value:
                return
            preview_state["token"] += 1
            token = preview_state["token"]
            update_exact_values()
            if preview_state["running"]:
                preview_state["pending"] = True
                return
            cancel_preview_task()

            async def delayed_render():
                try:
                    await asyncio.sleep(disp.LIVE_PREVIEW_DEBOUNCE_SECONDS)
                except asyncio.CancelledError:
                    return
                if (
                    token == preview_state["token"]
                    and live_preview_widget.value
                    and not preview_state["running"]
                ):
                    render_live_preview(token)

            try:
                preview_state["task"] = asyncio.get_running_loop().create_task(delayed_render())
            except RuntimeError:
                render_live_preview(token)

        def on_live_preview_toggle(change):
            if change["new"]:
                schedule_live_preview()
            else:
                cancel_preview_task()
                with live_output:
                    clear_output(wait=True)
                    display(Markdown("Live preview paused. You can still save a result."))

        def randomize_seed(*_):
            seed_widget.value = secrets.randbelow(10000)

        def run_from_controls(make_movie=False):
            cancel_preview_task()
            set_busy(True)
            try:
                with log:
                    clear_output(wait=True)
                    start = time.perf_counter()
                    case, preset, run_scope = preset_from_controls()
                    progress.value = 0
                    status.value = (
                        f"Computing the {case.capitalize()} {run_scope} result "
                        f"through t = {preset['T']:.1f}."
                    )
                    display(Markdown(status.value))

                    def callback(step, nsteps, current_time):
                        progress.value = max(progress.value, int(80 * step / max(nsteps, 1)))
                        status.value = f"Simulation time: {current_time:.1f} / {preset['T']:.1f}"

                    run_display(
                        case,
                        preset=preset,
                        make_movie=bool(make_movie),
                        progress_callback=callback,
                        run_scope=run_scope,
                        execution_mode="interactive_notebook",
                    )
                    progress.value = 100
                    status.value = f"Done in {time.perf_counter() - start:.1f} s."
            except Exception:
                with log:
                    display(
                        _error_panel("The simulation did not complete.", traceback.format_exc())
                    )
                progress.value = 0
                status.value = "Run failed; the previous result remains available."
            finally:
                set_busy(False)

        run_button.on_click(lambda _: run_from_controls(make_movie=False))
        run_movie_button.on_click(lambda _: run_from_controls(make_movie=True))
        restore_button.on_click(restore_reference_preset)
        randomize_button.on_click(randomize_seed)
        for observed in (seed_widget, bump_widget, bump_width_widget, noise_widget):
            observed.observe(schedule_live_preview, names="value")
        live_preview_widget.observe(on_live_preview_toggle, names="value")

        update_case_styles()
        update_visibility()
        update_exact_values()
        cards = widgets.GridBox(
            case_cards,
            layout=widgets.Layout(
                grid_template_columns="repeat(auto-fit, minmax(220px, 1fr))",
                grid_gap="10px",
                width="100%",
            ),
        )
        shape_grid = widgets.GridBox(
            [bump_widget, bump_width_widget, noise_widget],
            layout=widgets.Layout(
                grid_template_columns="repeat(auto-fit, minmax(300px, 1fr))",
                grid_gap="8px",
                width="100%",
            ),
        )
        export_grid = widgets.GridBox(
            [quality_widget],
            layout=widgets.Layout(
                grid_template_columns="repeat(auto-fit, minmax(300px, 1fr))",
                grid_gap="8px",
                width="100%",
            ),
        )
        controls = widgets.VBox(
            [
                header,
                cards,
                widgets.HTML("<h3>Shape controls</h3>"),
                shape_grid,
                widgets.HBox([restore_button, randomize_button]),
                scope_info,
                widgets.HTML("<h3>Live preview</h3>"),
                live_preview_widget,
                live_output,
                widgets.HTML("<h3>Save a result</h3>"),
                export_grid,
                widgets.HBox([run_button, run_movie_button]),
                advanced,
                progress,
                status,
                log,
            ],
            layout=widgets.Layout(border="1px solid #ddd", padding="12px", width="100%"),
        )
        display(controls)
        schedule_live_preview()
    else:
        auto_requested = bool(globals().get("AUTO_STANDARD_PREVIEW", True))
        auto_disabled_by_env = os.environ.get("QTP_AUTO_STANDARD_PREVIEW", "1").strip().lower() in {
            "0",
            "false",
            "no",
            "off",
        }
        auto_enabled = bool(auto_requested and not auto_disabled_by_env)

        if auto_enabled:
            config = standard_preview_configuration()
            signature = hashlib.sha256(
                json.dumps(config, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            state = globals().setdefault("_QTP_STANDARD_PREVIEW_STATE", {})
            display(
                HTML(
                    '<div style="border-left:5px solid #2f6f9f;padding:10px 14px;background:#eef7ff;">'
                    "<b>Interactive controls are unavailable in this session, so the standard preview is being generated instead.</b><br>"
                    "The calculation uses the default preview settings in the current notebook. "
                    "To enable the interactive interface, start the package with "
                    "<code>python launch_qtp_explorer.py</code>. To check the environment, run "
                    "<code>python check_qtp_environment.py</code>."
                    "</div>"
                )
            )

            already_done = bool(
                globals().get("STANDARD_PREVIEW_ONCE_PER_CONFIGURATION", True)
                and state.get("signature") == signature
                and state.get("status") == "pass"
            )
            if already_done:
                display(
                    Markdown(
                        "**The preview for this configuration has already been generated.**  "
                        "Change a preview value, call `reset_standard_preview()`, or invoke "
                        "`run_standard_preview_current_config()` manually to run it again."
                    )
                )
            else:
                state.clear()
                state.update({"signature": signature, "status": "running", "configuration": config})
                try:
                    QTP_STANDARD_PREVIEW_RESULT, QTP_STANDARD_PREVIEW_SUMMARY = (
                        run_standard_preview_current_config(
                            execution_mode="standard_notebook"
                        )
                    )
                    state.update({"status": "pass"})
                except Exception:
                    state.update({"status": "fail", "traceback": traceback.format_exc()})
                    display(
                        _error_panel(
                            "The standard notebook preview did not complete.",
                            state["traceback"],
                        )
                    )
        else:
            display(
                HTML(
                    '<div style="border-left:5px solid #c58b00;padding:10px 14px;background:#fff8e6;">'
                    "<b>Interactive controls are unavailable in this session.</b><br>"
                    "Automatic preview generation is disabled. Run "
                    "<code>run_standard_preview_current_config()</code>, or start the package with "
                    "<code>python launch_qtp_explorer.py</code> for the interactive interface. To check "
                    "the environment, run <code>python check_qtp_environment.py</code>."
                    "</div>"
                )
            )
