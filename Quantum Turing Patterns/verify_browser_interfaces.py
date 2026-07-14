#!/usr/bin/env python3
"""Browser tests for the research and museum interfaces."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _browser() -> tuple[Any, Any, str]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Browser verification requires Playwright. Install the package requirements first."
        ) from exc

    executable = shutil.which("chromium") or shutil.which("chromium-browser")
    manager = sync_playwright().start()
    kwargs: dict[str, Any] = {
        "headless": True,
        "args": ["--autoplay-policy=no-user-gesture-required", "--disable-dev-shm-usage"],
    }
    if executable:
        kwargs["executable_path"] = executable
    browser = manager.chromium.launch(**kwargs)
    return manager, browser, executable or "playwright-managed chromium"


def _synthetic_result() -> dict[str, Any]:
    n = 48
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X, Y = np.meshgrid(x, x)
    times = np.array([0.0, 1.0, 2.0])
    R = np.stack(
        [
            0.08 * np.cos(3 * X),
            0.22 * np.cos(3 * X) * (0.6 + 0.4 * np.cos(Y)),
            0.38 * np.cos(3 * X),
        ]
    ).astype(np.float32)
    E = np.stack(
        [
            np.zeros_like(X),
            0.004 * (1 + np.cos(6 * X)) / 2,
            0.012 * (1 + np.cos(6 * X)) / 2,
        ]
    ).astype(np.float32)
    return {
        "title": "Stripe playback test",
        "kind": "stripe",
        "regime": "stripe",
        "R_frames": R,
        "E_frames": E,
        "times": times,
        "E_times": times,
    }


def verify_research_explorer() -> dict[str, Any]:
    import qtp_display as display

    markup = display.client_side_time_explorer_html(
        _synthetic_result(), diagnostic_vmax=0.02, initial="final"
    )
    console_errors: list[str] = []
    manager, browser, executable = _browser()
    try:
        page = browser.new_page(viewport={"width": 1100, "height": 850})
        page.on(
            "console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None
        )
        page.on("pageerror", lambda exc: console_errors.append(str(exc)))
        page.set_content(f"<!doctype html><html><body>{markup}</body></html>", wait_until="load")
        slider = page.locator('input[data-role="slider"]')
        image = page.locator('img[data-role="image"]')
        status = page.locator('[data-role="status"]')
        replay = page.get_by_role("button", name="Replay formation")
        pause = page.get_by_role("button", name="Pause")
        checks = {
            "slider_present": slider.count() == 1,
            "image_present": image.count() == 1,
            "alt_text_present": bool(image.get_attribute("alt")),
            "starts_at_final": slider.input_value() == "2",
            "fixed_scale_disclosed": "Fixed diagnostic color scale"
            in page.locator("body").inner_text(),
            "status_prefix_not_duplicated": "local Gaussian excess: local Gaussian excess"
            not in status.inner_text(),
        }
        replay.click()
        page.wait_for_timeout(420)
        checks["replay_advances"] = int(slider.input_value()) > 0
        checks["status_updates"] = "Turing spectral selection" in status.inner_text()
        pause.click()
        paused = slider.input_value()
        page.wait_for_timeout(260)
        checks["pause_stops"] = slider.input_value() == paused
        checks["no_console_errors"] = not console_errors
        return {
            "ok": all(checks.values()),
            "browser": executable,
            "checks": checks,
            "console_errors": console_errors,
        }
    finally:
        browser.close()
        manager.stop()


def _exercise_exhibit_page(page, *, embedded_media: bool) -> tuple[dict[str, bool], list[str]]:
    console_errors: list[str] = []
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: console_errors.append(str(exc)))
    page.wait_for_function(
        "document.querySelector('video[data-role=video]').readyState >= 1", timeout=30000
    )
    body = page.locator("body")
    buttons = {
        name: page.get_by_role("button", name=name) for name in ("Spot", "Labyrinth", "Stripe")
    }
    video = page.locator('video[data-role="video"]')
    slider = page.locator('input[data-role="slider"]')
    time_label = page.locator('[data-role="time"]')
    visible_text = body.inner_text()
    initial_src = video.get_attribute("src") or ""
    checks: dict[str, bool] = {
        "one_h1": page.locator("h1").count() == 1,
        "three_case_buttons": all(button.count() == 1 for button in buttons.values()),
        "spot_selected_initially": buttons["Spot"].get_attribute("aria-pressed") == "true",
        "video_present": video.count() == 1,
        "video_has_accessible_label": bool(video.get_attribute("aria-label")),
        "slider_present": slider.count() == 1,
        "media_mode_expected": initial_src.startswith("data:video/mp4;base64,")
        if embedded_media
        else bool(initial_src),
        "technical_metadata_hidden": not any(
            term in visible_text
            for term in ("Python", "kernel", "JSON", "verification", "seed", "Export")
        ),
        "raw_html_source_hidden": '<div id="qtp_exhibit_' not in visible_text,
        "base64_source_hidden": "data:video/mp4;base64" not in visible_text,
        "speed_control_absent": "Speed" not in visible_text and page.locator("select").count() == 0,
        "redundant_caption_absent": "Pattern field (left)" not in visible_text,
    }

    # Physical simulation time must be derived from frame_times, not MP4 seconds.
    expected_midpoints = {"Spot": 25.0, "Labyrinth": 40.0, "Stripe": 80.0}
    midpoint_ok = True
    for name, expected in expected_midpoints.items():
        buttons[name].click()
        page.wait_for_function(
            "document.querySelector('video[data-role=video]').readyState >= 1", timeout=30000
        )
        page.get_by_role("button", name="Pause").click()
        page.evaluate("""() => {
            const s=document.querySelector('input[data-role=slider]');
            s.value='500'; s.dispatchEvent(new Event('input',{bubbles:true}));
            s.dispatchEvent(new Event('change',{bubbles:true}));
        }""")
        page.wait_for_timeout(120)
        text = time_label.inner_text()
        try:
            current = float(text.split("=")[1].split("/")[0].strip())
            final = float(text.split("/")[1].strip())
            midpoint_ok = (
                midpoint_ok and abs(current - expected) <= 1.5 and abs(final - 2 * expected) <= 0.1
            )
        except Exception:
            midpoint_ok = False
    checks["physical_midpoint_times"] = midpoint_ok

    # Observe the state change separately from subsequent playback.
    page.evaluate("""() => {
        const button = [...document.querySelectorAll('[data-case]')]
            .find(item => item.dataset.case === 'labyrinth');
        window.__qtpSwitchState = null;
        button.addEventListener('click', () => {
            const video = document.querySelector('video[data-role=video]');
            const slider = document.querySelector('input[data-role=slider]');
            const time = document.querySelector('[data-role=time]');
            window.__qtpSwitchState = {
                selected: button.getAttribute('aria-pressed') === 'true',
                source: video.getAttribute('src') || '',
                currentTime: Number(video.currentTime),
                slider: Number(slider.value),
                timeLabel: time.textContent.trim(),
            };
        }, {once: true});
    }""")
    buttons["Labyrinth"].click()
    page.wait_for_function("window.__qtpSwitchState !== null")
    switch_state = page.evaluate("window.__qtpSwitchState")
    checks["case_switch_changes_source"] = switch_state["source"] != initial_src
    checks["labyrinth_selected"] = bool(switch_state["selected"])
    checks["switch_resets_to_start"] = (
        switch_state["slider"] <= 5
        and switch_state["currentTime"] <= 0.05
        and switch_state["timeLabel"].startswith("t = 0.0")
    )
    page.wait_for_function(
        """() => {
            const video = document.querySelector('video[data-role=video]');
            const button = [...document.querySelectorAll('[data-case]')]
                .find(item => item.dataset.case === 'labyrinth');
            return video.readyState >= 1 && button.getAttribute('aria-pressed') === 'true';
        }""",
        timeout=30000,
    )
    page.wait_for_function(
        """() => {
            const video = document.querySelector('video[data-role=video]');
            return !video.paused && video.currentTime > 0.02;
        }""",
        timeout=30000,
    )
    checks["switch_autoplays"] = True

    page.evaluate("""() => {
        const button = document.querySelector('[data-action=replay]');
        window.__qtpReplayState = null;
        button.addEventListener('click', () => {
            const video = document.querySelector('video[data-role=video]');
            const slider = document.querySelector('input[data-role=slider]');
            window.__qtpReplayState = {
                currentTime: Number(video.currentTime),
                slider: Number(slider.value),
            };
        }, {once: true});
    }""")
    page.get_by_role("button", name="Replay formation").click()
    page.wait_for_function("window.__qtpReplayState !== null")
    replay_state = page.evaluate("window.__qtpReplayState")
    checks["replay_resets_to_start"] = (
        replay_state["slider"] <= 5 and replay_state["currentTime"] <= 0.05
    )
    page.wait_for_function(
        """() => {
            const video = document.querySelector('video[data-role=video]');
            return !video.paused && video.currentTime > 0.05;
        }""",
        timeout=30000,
    )
    checks["replay_advances"] = True
    page.get_by_role("button", name="Pause").click()
    page.wait_for_function("document.querySelector('video[data-role=video]').paused")
    paused_time = float(
        page.evaluate("document.querySelector('video[data-role=video]').currentTime")
    )
    page.wait_for_timeout(500)
    checks["pause_stops"] = (
        abs(
            float(page.evaluate("document.querySelector('video[data-role=video]').currentTime"))
            - paused_time
        )
        < 0.08
    )

    # Idle reset: Stripe -> no interaction -> Spot at rate 1 from start, autoplay.
    buttons["Stripe"].click()
    page.wait_for_function(
        """() => [...document.querySelectorAll('[data-case]')]
            .find(item => item.dataset.case === 'stripe')
            .getAttribute('aria-pressed') === 'true'"""
    )
    page.wait_for_function(
        """() => [...document.querySelectorAll('[data-case]')]
            .find(item => item.dataset.case === 'spot')
            .getAttribute('aria-pressed') === 'true'""",
        timeout=10000,
    )
    checks["idle_resets_to_spot"] = True
    checks["idle_restores_rate_one"] = (
        abs(
            float(page.evaluate("document.querySelector('video[data-role=video]').playbackRate"))
            - 1.0
        )
        < 1e-9
    )
    page.wait_for_function(
        """() => {
            const video = document.querySelector('video[data-role=video]');
            return !video.paused && video.currentTime > 0.02;
        }""",
        timeout=30000,
    )
    checks["idle_autoplays"] = True

    # No-scroll release targets, including smaller effective viewports.
    no_scroll = True
    for width, height in ((1920, 1080), (1366, 768), (1280, 800)):
        page.set_viewport_size({"width": width, "height": height})
        page.wait_for_timeout(80)
        no_scroll = no_scroll and bool(
            page.evaluate(
                "document.documentElement.scrollHeight <= window.innerHeight + 2 && "
                "document.documentElement.scrollWidth <= window.innerWidth + 2"
            )
        )
    checks["no_scroll_supported_viewports"] = no_scroll
    checks["no_console_errors"] = not console_errors
    return checks, console_errors


def verify_exhibit_player(root: Path) -> dict[str, Any]:
    import qtp_exhibit

    markup = qtp_exhibit.exhibit_html(root=root, autoplay=False, idle_seconds=2.0)
    manager, browser, executable = _browser()
    try:
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.set_content(
            "<!doctype html><html><head><style>html,body{margin:0;padding:0;overflow:hidden}</style></head>"
            f"<body>{markup}</body></html>",
            wait_until="load",
        )
        checks, errors = _exercise_exhibit_page(page, embedded_media=True)
        return {
            "ok": all(checks.values()),
            "browser": executable,
            "checks": checks,
            "console_errors": errors,
        }
    finally:
        browser.close()
        manager.stop()


def _dependency_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "installed"))
    except Exception:
        return None


def verify_live_exhibit(root: Path, timeout: float = 120.0) -> dict[str, Any]:
    versions = {"voila": _dependency_version("voila")}
    if versions["voila"] is None:
        return {"ok": None, "status": "skipped_missing_dependency", "dependencies": versions}

    kernel_install = subprocess.run(
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
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if kernel_install.returncode != 0:
        return {
            "ok": False,
            "status": "error",
            "dependencies": versions,
            "error": "could not register the QTP kernel",
            "server_output_tail": kernel_install.stdout[-4000:],
        }

    port = _free_port()
    environment = os.environ.copy()
    environment.update(
        {
            "QTP_EXHIBIT_IDLE_SECONDS": "2.0",
            "QTP_EXHIBIT_EMBED_MEDIA": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = [
        sys.executable,
        "-m",
        "voila",
        "qtp_exhibit.ipynb",
        "--no-browser",
        "--Voila.ip=127.0.0.1",
        f"--port={port}",
        "--VoilaConfiguration.strip_sources=True",
        "--VoilaConfiguration.show_tracebacks=False",
    ]
    process = subprocess.Popen(
        command,
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    manager = browser = None
    server_output = ""
    result: dict[str, Any]
    try:
        deadline = time.time() + timeout
        ready = False
        while time.time() < deadline:
            if process.poll() is not None:
                break
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    ready = True
                    break
            except OSError:
                time.sleep(0.2)
        if not ready:
            raise RuntimeError("the live exhibit server did not become available")

        url = f"http://127.0.0.1:{port}"
        manager, browser, executable = _browser()
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(url, wait_until="domcontentloaded", timeout=int(timeout * 1000))
        checks, console_errors = _exercise_exhibit_page(page, embedded_media=True)
        result = {
            "ok": all(checks.values()),
            "status": "pass" if all(checks.values()) else "fail",
            "browser": executable,
            "dependencies": versions,
            "checks": checks,
            "console_errors": console_errors,
        }
    except Exception as exc:
        result = {
            "ok": False,
            "status": "error",
            "dependencies": versions,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if browser is not None:
            browser.close()
        if manager is not None:
            manager.stop()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        if process.stdout:
            try:
                server_output = process.stdout.read() or ""
            except Exception:
                server_output = ""

    if not result.get("ok") and server_output:
        result["server_output_tail"] = server_output[-4000:]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--with-live-exhibit",
        action="store_true",
        help="also test the live Voilà exhibit",
    )
    parser.add_argument(
        "--require-live-exhibit",
        action="store_true",
        help="require the live Voilà exhibit test and fail if it cannot run",
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "release_version": (root / "VERSION").read_text().strip(),
        "research_explorer": verify_research_explorer(),
        "exhibit_player": verify_exhibit_player(root),
    }
    if args.with_live_exhibit or args.require_live_exhibit:
        report["voila_exhibit"] = verify_live_exhibit(root)
    else:
        report["voila_exhibit"] = {"ok": None, "status": "not_requested"}

    ok = bool(report["research_explorer"]["ok"] and report["exhibit_player"]["ok"])
    if args.require_live_exhibit:
        ok = ok and report["voila_exhibit"].get("ok") is True
    elif args.with_live_exhibit and report["voila_exhibit"].get("ok") is False:
        ok = False
    report["ok"] = ok

    report_path = args.report or (root / "verification_runtime" / "browser_interface_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("wrote", report_path)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
