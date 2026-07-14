from __future__ import annotations

import base64
import html
import json
import os
import uuid
from pathlib import Path
from typing import Any

from IPython.display import HTML, display

CASES = (
    ("spot", "Spot"),
    ("labyrinth", "Labyrinth"),
    ("stripe", "Stripe"),
)


def _video_source(path: Path, *, embed_media: bool, root: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing exhibit movie: {path}. Run make_display_assets.py before launching the exhibit."
        )
    if embed_media:
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:video/mp4;base64,{payload}"
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _case_metadata(display_dir: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for key, label in CASES:
        summary_path = display_dir / f"display_{key}_pattern_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Missing exhibit metadata: {summary_path}. Regenerate the display assets."
            )
        data = json.loads(summary_path.read_text())
        plan = data.get("movie_plan") or {}
        frame_times = [float(value) for value in plan.get("frame_times") or []]
        if not frame_times:
            final_time = float(
                plan.get("expected_final_time") or data.get("parameters", {}).get("T") or 0.0
            )
            n_frames = int(plan.get("n_frames_expected") or 61)
            frame_times = [final_time * index / max(n_frames - 1, 1) for index in range(n_frames)]
        metadata[key] = {
            "label": label,
            "frame_times": frame_times,
            "physical_final_time": float(frame_times[-1]),
            "n_frames": len(frame_times),
        }
    return metadata


def exhibit_html(
    root: str | Path | None = None,
    *,
    autoplay: bool = True,
    embed_media: bool = True,
    idle_seconds: float | None = None,
    playback_rate: float = 1.0,
) -> str:
    """Build the museum exhibit player.

    The player uses the precomputed reference movies. Case switching, replay,
    pause, physical-time seeking, looping, and idle reset remain responsive
    without recomputing the simulation.

    ``embed_media=True`` creates a self-contained notebook display. ``embed_media=False``
    emits relative movie URLs for an exhibit server that exposes package files.
    """
    root_path = Path(root) if root is not None else Path(__file__).resolve().parent
    display_dir = root_path / "display"
    metadata = _case_metadata(display_dir)
    sources = {
        key: _video_source(
            display_dir / f"display_{key}_pattern.mp4",
            embed_media=bool(embed_media),
            root=root_path,
        )
        for key, _label in CASES
    }
    if idle_seconds is None:
        idle_seconds = float(os.environ.get("QTP_EXHIBIT_IDLE_SECONDS", "300"))
    idle_seconds = max(float(idle_seconds), 0.0)
    playback_rate = max(float(playback_rate), 0.05)

    uid = "qtp_exhibit_" + uuid.uuid4().hex
    sources_json = json.dumps(sources)
    metadata_json = json.dumps(metadata)
    autoplay_js = "true" if autoplay else "false"
    idle_ms = int(round(idle_seconds * 1000.0))

    case_buttons = "".join(
        f'<button type="button" class="qtp-case" data-case="{html.escape(key)}" '
        f'aria-pressed="false">{html.escape(label)}</button>'
        for key, label in CASES
    )

    return f'''<div id="{uid}" class="qtp-exhibit" data-qtp-exhibit="true">
<style>
#{uid}.qtp-exhibit {{
  width:100vw; height:100vh;
  margin:0; padding:2px; box-sizing:border-box; overflow:hidden;
  display:flex; flex-direction:column; gap:4px;
  font-family:"Times New Roman",Times,"TeX Gyre Termes","Nimbus Roman",serif; color:#171717;
}}
#{uid} .qtp-toolbar {{
  flex:0 0 auto; display:grid;
  grid-template-columns:auto auto auto auto auto minmax(260px,1fr);
  align-items:center; justify-content:center; gap:8px;
  padding:5px 8px; border:1px solid #d6d6d6; border-radius:10px;
  background:#fafafa;
}}
#{uid} button {{
  min-height:42px; min-width:108px; border:2px solid #777; border-radius:9px;
  background:white; padding:5px 12px; font-size:clamp(15px,1vw,19px);
  font-weight:650; cursor:pointer; touch-action:manipulation;
}}
#{uid} button:hover, #{uid} button:focus-visible {{ outline:3px solid #b9d7ff; outline-offset:2px; }}
#{uid} .qtp-case[aria-pressed="true"] {{ border-width:4px; border-color:#111; box-shadow:0 0 0 3px #d9e8ff inset; }}
#{uid} .qtp-case[aria-pressed="true"]::before {{ content:"✓ "; }}
#{uid} .qtp-range-wrap {{
  display:grid; grid-template-columns:auto minmax(170px,1fr) minmax(128px,auto);
  align-items:center; gap:10px; min-width:0;
  font-size:clamp(15px,1vw,19px); font-weight:650;
}}
#{uid} input[type="range"] {{
  -webkit-appearance:none; appearance:none; width:100%; height:10px;
  min-height:40px; margin:0; background:transparent; cursor:pointer; touch-action:pan-x;
}}
#{uid} input[type="range"]::-webkit-slider-runnable-track {{ height:12px; border-radius:999px; background:#c8c8c8; }}
#{uid} input[type="range"]::-webkit-slider-thumb {{
  -webkit-appearance:none; width:32px; height:32px; margin-top:-10px;
  border:2px solid #111; border-radius:50%; background:#fff;
  box-shadow:0 1px 4px rgba(0,0,0,.25);
}}
#{uid} input[type="range"]::-moz-range-track {{ height:12px; border-radius:999px; background:#c8c8c8; }}
#{uid} input[type="range"]::-moz-range-thumb {{
  width:32px; height:32px; border:2px solid #111; border-radius:50%; background:#fff;
}}
#{uid} .qtp-time {{ min-width:128px; text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
#{uid} .qtp-video-wrap {{
  flex:1 1 auto; min-height:0; display:flex; align-items:center; justify-content:center;
  overflow:hidden; border-radius:8px; background:#fff;
}}
#{uid} video {{ width:100%; height:100%; object-fit:contain; display:block; background:#fff; }}
@media (max-width:1180px) {{
  #{uid} .qtp-toolbar {{ grid-template-columns:repeat(5,minmax(105px,1fr)); }}
  #{uid} .qtp-range-wrap {{ grid-column:1/-1; }}
  #{uid} button {{ min-width:0; }}
}}
@media (max-width:760px) {{
  #{uid} .qtp-toolbar {{ grid-template-columns:repeat(3,1fr); gap:6px; padding:6px; }}
  #{uid} .qtp-toolbar button[data-action="replay"], #{uid} .qtp-toolbar button[data-action="pause"] {{ grid-column:auto; }}
  #{uid} .qtp-range-wrap {{ grid-column:1/-1; grid-template-columns:auto minmax(120px,1fr); }}
  #{uid} .qtp-time {{ grid-column:1/-1; text-align:center; }}
  #{uid} button {{ min-height:46px; padding:5px 8px; }}
}}
</style>
<div class="qtp-toolbar" role="group" aria-label="Quantum Turing pattern playback controls">
  <button type="button" data-action="replay">Replay formation</button>
  <button type="button" data-action="pause">Pause</button>
  {case_buttons}
  <label class="qtp-range-wrap">Evolution
    <input data-role="slider" type="range" min="0" max="1000" value="0" aria-label="Evolution time">
    <span class="qtp-time" data-role="time">t = 0.0 / 0.0</span>
  </label>
</div>
<div class="qtp-video-wrap">
  <video data-role="video" muted playsinline preload="auto" aria-label="Quantum Turing pattern formation movie"></video>
</div>
<script>(function(){{
  const root=document.getElementById('{uid}');
  const sources={sources_json};
  const metadata={metadata_json};
  const shouldAutoplay={autoplay_js};
  const idleMilliseconds={idle_ms};
  const defaultPlaybackRate={playback_rate!r};
  const video=root.querySelector('[data-role=video]');
  const slider=root.querySelector('[data-role=slider]');
  const timeLabel=root.querySelector('[data-role=time]');
  const caseButtons=[...root.querySelectorAll('[data-case]')];
  let selected='spot'; let seeking=false; let idleTimer=null;

  function selectedMetadata(){{ return metadata[selected] || {{frame_times:[0],physical_final_time:0,label:selected}}; }}
  function physicalTimeFromFraction(fraction){{
    const frameTimes=selectedMetadata().frame_times || [0];
    const clipped=Math.min(1,Math.max(0,Number.isFinite(fraction)?fraction:0));
    const index=Math.min(frameTimes.length-1,Math.round(clipped*Math.max(frameTimes.length-1,0)));
    return Number(frameTimes[index] || 0);
  }}
  function updateCaseButtons(){{
    caseButtons.forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.case===selected)));
    video.setAttribute('aria-label',`${{selectedMetadata().label}} quantum Turing pattern formation movie`);
  }}
  function updateTime(){{
    const duration=Number.isFinite(video.duration)&&video.duration>0?video.duration:0;
    const current=Number.isFinite(video.currentTime)?video.currentTime:0;
    const fraction=duration>0?current/duration:Number(slider.value)/1000;
    if(!seeking && duration>0) slider.value=String(Math.round(1000*Math.min(1,Math.max(0,fraction))));
    const physical=physicalTimeFromFraction(fraction);
    const finalTime=Number(selectedMetadata().physical_final_time || 0);
    timeLabel.textContent=`t = ${{physical.toFixed(1)}} / ${{finalTime.toFixed(1)}}`;
  }}
  async function playQuietly(){{ try{{ await video.play(); }}catch(_err){{ /* user gesture may be required */ }} }}
  function resetIdleTimer(){{
    if(idleTimer!==null) window.clearTimeout(idleTimer);
    if(idleMilliseconds>0) idleTimer=window.setTimeout(resetExhibit,idleMilliseconds);
  }}
  function selectCase(name,autoPlay=true){{
    if(!(name in sources)) return;
    video.pause(); selected=name; updateCaseButtons();
    video.src=sources[name]; video.load(); video.playbackRate=defaultPlaybackRate;
    slider.value='0'; updateTime(); resetIdleTimer();
    const startSelectedCase=()=>{{
      video.currentTime=0; slider.value='0'; updateTime();
      if(autoPlay) playQuietly();
    }};
    if(video.readyState>=1) startSelectedCase();
    else video.addEventListener('loadedmetadata',startSelectedCase,{{once:true}});
  }}
  function resetExhibit(){{
    selectCase('spot',true);
  }}
  caseButtons.forEach(button=>button.addEventListener('click',()=>selectCase(button.dataset.case,true)));
  root.querySelector('[data-action=replay]').addEventListener('click',()=>{{
    video.currentTime=0; slider.value='0'; updateTime(); playQuietly(); resetIdleTimer();
  }});
  root.querySelector('[data-action=pause]').addEventListener('click',()=>{{video.pause();resetIdleTimer();}});
  slider.addEventListener('pointerdown',()=>{{seeking=true;resetIdleTimer();}});
  slider.addEventListener('touchstart',resetIdleTimer,{{passive:true}});
  slider.addEventListener('input',()=>{{
    if(Number.isFinite(video.duration)&&video.duration>0) video.currentTime=video.duration*Number(slider.value)/1000;
    updateTime(); resetIdleTimer();
  }});
  slider.addEventListener('change',()=>{{seeking=false;updateTime();resetIdleTimer();}});
  root.addEventListener('pointerdown',resetIdleTimer,{{passive:true}});
  root.addEventListener('touchstart',resetIdleTimer,{{passive:true}});
  video.addEventListener('timeupdate',updateTime);
  video.addEventListener('loadedmetadata',()=>{{video.playbackRate=defaultPlaybackRate;updateTime();}});
  video.addEventListener('ended',()=>{{video.currentTime=0;playQuietly();}});
  selectCase('spot',shouldAutoplay);
  window['{uid}']={{selectCase,resetExhibit,video,slider,root,metadata}};
}})();</script>
</div>'''


def show_exhibit(
    root: str | Path | None = None,
    *,
    autoplay: bool = True,
    embed_media: bool | None = None,
    idle_seconds: float | None = None,
) -> None:
    """Display the museum exhibit."""
    if embed_media is None:
        raw = os.environ.get("QTP_EXHIBIT_EMBED_MEDIA", "1").strip().lower()
        embed_media = raw not in {"0", "false", "no", "off"}
    display(
        HTML(
            exhibit_html(
                root=root,
                autoplay=autoplay,
                embed_media=bool(embed_media),
                idle_seconds=idle_seconds,
            )
        )
    )
