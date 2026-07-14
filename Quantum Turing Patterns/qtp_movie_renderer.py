"""Movie rendering for the research and exhibit interfaces."""

from __future__ import annotations

import io

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import colormaps, font_manager
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import qtp_kernels as q

CMAP_FIELD = q.CMAP_FIELD
CMAP_ENT = q.CMAP_ENT
VMIN, VMAX = q.VMIN, q.VMAX
EXHIBIT_Y_LABEL_X_OFFSET_PX = 76
EXHIBIT_AXIS_LABEL_FONT_SIZE = 19
EXHIBIT_AXIS_LABEL_SCALE_REFERENCE = r"$x/L$"
TIMES_FAMILIES = ["Times New Roman", "Times", "TeX Gyre Termes", "Nimbus Roman", "Liberation Serif"]

# All rendered scientific figures and movie labels use a Times-family serif.
# Matplotlib resolves the first installed member of the stack; mathtext uses
# STIX, whose metrics and visual language are compatible with Times.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": TIMES_FAMILIES,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)


def _movie_font(size=24, bold=False):
    """Resolve a Times-family font without bundling proprietary font files."""
    try:
        prop = font_manager.FontProperties(
            family=TIMES_FAMILIES,
            weight="bold" if bold else "normal",
        )
        path = font_manager.findfont(prop, fallback_to_default=True)
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


_FONT_TITLE = _movie_font(34, True)
_FONT_LABEL = _movie_font(24, False)
_FONT_TINY = _movie_font(16, False)
_FONT_EXHIBIT_TITLE = _movie_font(42, True)
_FONT_EXHIBIT_LABEL = _movie_font(30, False)
_FONT_EXHIBIT_TICK = _movie_font(22, False)
_FIELD_CMAP_OBJ = colormaps[CMAP_FIELD]
_ENT_CMAP_OBJ = colormaps[CMAP_ENT]


def _arr_to_rgb(arr, cmap, vmin, vmax):
    x = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    return (cmap(x)[:, :, :3] * 255).astype(np.uint8)


def _resize_scalar_panel(arr, panel_size, *, resample=Image.Resampling.BICUBIC):
    """Resize a native scalar field before applying its colormap.

    Resizing the scalar data, rather than an already-colored low-resolution
    image, preserves the full native-grid diagnostic information in exhibit
    frames while remaining a display-only interpolation.
    """
    src = Image.fromarray(np.asarray(arr, dtype=np.float32), mode="F")
    return np.asarray(src.resize(panel_size, resample), dtype=float)


def _nice_ceil(x):
    if x <= 0:
        return 1.0
    exp = np.floor(np.log10(x))
    base = x / (10**exp)
    for c in [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]:
        if base <= c + 1e-12:
            return float(c * (10**exp))
    return float(10 ** (exp + 1))


def _movie_field_ticks(vmin, vmax):
    m = max(abs(vmin), abs(vmax))
    m = 0.2 if m <= 0.24 + 1e-12 else _nice_ceil(m)
    return [-m, 0.0, m]


def _movie_ent_ticks(vmax):
    top = _nice_ceil(vmax)
    return [0.0, top / 2.0, top]


def _turing_selection_stage(res, R):
    """Return a lightweight frame-level spectral-selection label.

    This is a display diagnostic only. The pass/fail certification remains in
    the JSON summary, which is computed from the raw arrays.
    """
    try:
        final_R = np.asarray(res["R_frames"][-1])
        kr, ps, _ = q.radial_spectrum(final_R)
        j = int(np.argmax(ps[1:]) + 1)
        k_dom = float(kr[j])
        shell_width = q.radial_bin_width(final_R)
        threshold = {"stripe": 0.95, "spot": 0.50, "labyrinth": 0.50}.get(res.get("kind"), 0.50)
        shell_t = float(q.shell_concentration(np.asarray(R), k_dom, shell_width))
        if shell_t < 0.25:
            stage = "not yet selected"
        elif shell_t < threshold:
            stage = "developing"
        else:
            stage = "selected"
        return stage, shell_t
    except Exception:
        return "not evaluated", float("nan")


def _draw_colorbar(draw, x, y, h, cmap, vmin, vmax, label, ticks, *, tick_font=None):
    w = 22
    tick_font = tick_font or _FONT_TINY
    for j in range(h):
        frac = 1.0 - j / max(h - 1, 1)
        color = tuple((np.array(cmap(frac)[:3]) * 255).astype(np.uint8).tolist())
        draw.line((x, y + j, x + w, y + j), fill=color)
    draw.rectangle((x, y, x + w, y + h), outline=(80, 80, 80), width=1)
    for tick in ticks:
        frac = (tick - vmin) / (vmax - vmin) if vmax > vmin else 0
        yy = y + h - int(frac * h)
        draw.line((x + w, yy, x + w + 5, yy), fill=(0, 0, 0), width=1)
        draw.text(
            (x + w + 8, yy - tick_font.size // 2), f"{tick:g}", fill=(0, 0, 0), font=tick_font
        )
    if label:
        draw.text((x + w + 8, y + h + 8), label, fill=(0, 0, 0), font=tick_font)


_MATH_LABEL_CACHE = {}


def _raw_math_label_image(expr, fontsize):
    """Rasterize and crop a math label without changing its scale."""
    fig = plt.figure(figsize=(0.1, 0.1), dpi=200)
    fig.patch.set_alpha(0.0)
    fig.text(0.0, 0.0, expr, fontsize=fontsize, ha="left", va="bottom")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.01, transparent=True)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    bbox = img.getbbox()
    return img.crop(bbox) if bbox is not None else img


def _math_label_image(expr, fontsize=24, rotate=0, scale_reference=None):
    key = (expr, fontsize, rotate, scale_reference)
    if key in _MATH_LABEL_CACHE:
        return _MATH_LABEL_CACHE[key]
    img = _raw_math_label_image(expr, fontsize)
    reference = _raw_math_label_image(scale_reference or expr, fontsize)
    target_reference_height = max(8, int(fontsize * 1.10))
    scale = target_reference_height / max(reference.height, 1)
    target_size = (
        max(1, int(round(img.width * scale))),
        max(1, int(round(img.height * scale))),
    )
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    if rotate:
        img = img.rotate(rotate, expand=True, resample=Image.Resampling.BICUBIC)
    _MATH_LABEL_CACHE[key] = img
    return img


def _paste_math(canvas, expr, xy, fontsize=24, anchor="lt", rotate=0, scale_reference=None):
    img = _math_label_image(
        expr,
        fontsize=fontsize,
        rotate=rotate,
        scale_reference=scale_reference,
    )
    x, y = xy
    if "m" in anchor:
        x -= img.width // 2
    if "r" in anchor:
        x -= img.width
    if "a" in anchor:
        y -= img.height
    if "c" in anchor:
        y -= img.height // 2
    canvas.paste(img, (int(x), int(y)), img)


def movie_frame(res, R, E, t, diagnostic_color_vmax, case_final_diagnostic_max=None):
    """Render a fixed 1280x720 scientific movie frame."""
    canvas = Image.new("RGB", (1280, 720), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    raw_L = int(np.asarray(R).shape[0])
    panel_size = (400, 400)
    Rdisp = _resize_scalar_panel(R, panel_size)
    Edisp = _resize_scalar_panel(E, panel_size)
    ent_vmax_raw = max(float(1000 * diagnostic_color_vmax), 1e-9)
    ent_vmax = _nice_ceil(ent_vmax_raw)
    field_ticks = _movie_field_ticks(VMIN, VMAX)
    ent_ticks = _movie_ent_ticks(ent_vmax)
    left = Image.fromarray(_arr_to_rgb(Rdisp, _FIELD_CMAP_OBJ, VMIN, VMAX))
    right = Image.fromarray(_arr_to_rgb(1000 * Edisp, _ENT_CMAP_OBJ, 0, ent_vmax))
    lx, ly = 72, 142
    rx, ry = 700, 142
    canvas.paste(left, (lx, ly))
    canvas.paste(right, (rx, ry))
    draw.text(
        (640, 34), f"{res['title']},  t = {t:.1f}", anchor="ma", fill=(0, 0, 0), font=_FONT_TITLE
    )
    draw.text((lx, 104), "pattern field", fill=(0, 0, 0), font=_FONT_LABEL)
    draw.text((rx, 104), "excess Gaussian diagnostic", fill=(0, 0, 0), font=_FONT_LABEL)
    draw.rectangle(
        (lx, ly, lx + panel_size[0], ly + panel_size[1]), outline=(100, 100, 100), width=1
    )
    draw.rectangle(
        (rx, ry, rx + panel_size[0], ry + panel_size[1]), outline=(100, 100, 100), width=1
    )
    for x0, y0 in [(lx, ly), (rx, ry)]:
        for frac, lab in [(0, "0"), (0.5, str(raw_L // 2)), (1, str(raw_L))]:
            xx = x0 + int(frac * panel_size[0])
            yy = y0 + panel_size[1]
            draw.line((xx, yy, xx, yy + 5), fill=(0, 0, 0))
            draw.text((xx - 8, yy + 8), lab, fill=(0, 0, 0), font=_FONT_TINY)
            yy2 = y0 + panel_size[1] - int(frac * panel_size[1])
            draw.line((x0 - 5, yy2, x0, yy2), fill=(0, 0, 0))
            draw.text((x0 - 40, yy2 - 8), lab, fill=(0, 0, 0), font=_FONT_TINY)
        _paste_math(
            canvas,
            r"$x$",
            (x0 + panel_size[0] // 2, y0 + panel_size[1] + 45),
            fontsize=12,
            anchor="ma",
        )
        _paste_math(
            canvas, r"$y$", (x0 - 61, y0 + panel_size[1] // 2), fontsize=12, anchor="mc", rotate=90
        )
    _draw_colorbar(
        draw,
        lx + panel_size[0] + 26,
        ly + 12,
        panel_size[1] - 24,
        _FIELD_CMAP_OBJ,
        VMIN,
        VMAX,
        "",
        field_ticks,
    )
    _draw_colorbar(
        draw,
        rx + panel_size[0] + 26,
        ry + 12,
        panel_size[1] - 24,
        _ENT_CMAP_OBJ,
        0,
        ent_vmax,
        "",
        ent_ticks,
    )
    _paste_math(
        canvas, r"$R^{\rm pat}$", (lx + panel_size[0] + 37, ly - 16), fontsize=12, anchor="mc"
    )
    _paste_math(
        canvas,
        r"$10^3\Delta E_{N,G}^{\rm loc}$",
        (rx + panel_size[0] + 37, ry - 16),
        fontsize=12,
        anchor="mc",
    )

    max_e = float(np.nanmax(E)) if np.size(E) else 0.0
    temporal_ref = max(
        float(
            case_final_diagnostic_max
            if case_final_diagnostic_max is not None
            else diagnostic_color_vmax
        ),
        1e-12,
    )
    if max_e < 0.02 * temporal_ref:
        stage = "negligible"
    elif max_e < 0.70 * temporal_ref:
        stage = "developing"
    else:
        stage = "high"
    turing_stage, shell_t = _turing_selection_stage(res, R)
    min_nupt = min_phys = None
    try:
        e_times = np.asarray(res.get("E_times", []), dtype=float)
        ii = int(np.argmin(np.abs(e_times - float(t)))) if e_times.size else None
        if ii is not None:
            if "min_nupt" in res:
                min_nupt = float(np.asarray(res["min_nupt"])[ii])
            if "min_nu_phys" in res:
                min_phys = float(np.asarray(res["min_nu_phys"])[ii])
            elif "min_nu_physical" in res:
                min_phys = float(np.asarray(res["min_nu_physical"])[ii])
    except Exception:
        min_nupt = min_phys = None

    strip_y, strip_bottom = 570, 704
    draw.rounded_rectangle(
        (72, strip_y, 1208, strip_bottom),
        radius=14,
        fill=(248, 248, 248),
        outline=(210, 210, 210),
        width=1,
    )
    badge = (
        "Gaussian diagnostic: NPT"
        if (min_nupt is not None and min_nupt < 0.5)
        else "Gaussian diagnostic: tracking"
    )
    draw.text((92, strip_y + 14), badge, fill=(0, 0, 0), font=_FONT_LABEL)
    scale = str(res.get("movie_diagnostic_scale", "per_case_notebook_movie"))
    scale_text = (
        "common diagnostic scale for the reference movies"
        if scale == "bundled_reference_movies"
        else "per-case diagnostic color scale"
    )
    draw.text(
        (92, strip_y + 48), f"Local Gaussian excess: {stage}", fill=(45, 45, 45), font=_FONT_LABEL
    )
    draw.text(
        (92, strip_y + 80),
        f"Turing spectral selection: {turing_stage} (shell={shell_t:.3f})",
        fill=(45, 45, 45),
        font=_FONT_LABEL,
    )
    draw.text((92, strip_y + 108), scale_text, fill=(80, 80, 80), font=_FONT_TINY)
    math_x = 880
    if min_nupt is not None:
        _paste_math(
            canvas,
            rf"$\min\,\widetilde{{\nu}}_-={min_nupt:.3f}$",
            (math_x, strip_y + 22),
            fontsize=13,
            anchor="lt",
        )
    if min_phys is not None:
        _paste_math(
            canvas,
            rf"$\min\,\nu_{{\rm phys}}={min_phys:.3f}$",
            (math_x, strip_y + 64),
            fontsize=13,
            anchor="lt",
        )
    return np.array(canvas)


def exhibit_frame(res, R, E, t, pattern_color_vmax=VMAX, diagnostic_color_vmax=0.025):
    """Render a 1440×800 frame for the museum exhibit.

    The frame uses normalized axes and fixed reference scales.
    """
    canvas = Image.new("RGB", (1440, 800), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    pattern_vmax = max(abs(float(pattern_color_vmax)), 1e-12)
    diagnostic_vmax_raw = max(float(diagnostic_color_vmax), 1e-12)
    diagnostic_vmax_display = _nice_ceil(1000.0 * diagnostic_vmax_raw)
    panel_size = (500, 500)
    lx, ly = 92, 132
    # Add 40 px (about 1 cm at 96 dpi) between the two heatmap regions.
    rx, ry = 786, 132

    # Render directly from the native diagnostic grid.
    Rdisp = _resize_scalar_panel(R, panel_size)
    Edisp = _resize_scalar_panel(E, panel_size)
    left = Image.fromarray(_arr_to_rgb(Rdisp, _FIELD_CMAP_OBJ, -pattern_vmax, pattern_vmax))
    right = Image.fromarray(
        _arr_to_rgb(1000.0 * Edisp, _ENT_CMAP_OBJ, 0.0, diagnostic_vmax_display)
    )
    canvas.paste(left, (lx, ly))
    canvas.paste(right, (rx, ry))

    case_name = str(res.get("kind", res.get("title", "pattern"))).capitalize()
    draw.text(
        (720, 28),
        f"Quantum Turing Patterns — {case_name} — t = {float(t):.1f}",
        anchor="ma",
        fill=(0, 0, 0),
        font=_FONT_EXHIBIT_TITLE,
    )
    draw.text((lx, 88), "Pattern field", fill=(0, 0, 0), font=_FONT_EXHIBIT_LABEL)
    draw.text((rx, 88), "Entanglement diagnostic", fill=(0, 0, 0), font=_FONT_EXHIBIT_LABEL)

    for x0, y0 in [(lx, ly), (rx, ry)]:
        draw.rectangle(
            (x0, y0, x0 + panel_size[0], y0 + panel_size[1]),
            outline=(80, 80, 80),
            width=2,
        )
        for frac, lab in [(0.0, "0"), (0.5, "0.5"), (1.0, "1")]:
            xx = x0 + int(frac * panel_size[0])
            yy = y0 + panel_size[1]
            draw.line((xx, yy, xx, yy + 8), fill=(0, 0, 0), width=2)
            tw = draw.textbbox((0, 0), lab, font=_FONT_EXHIBIT_TICK)[2]
            draw.text((xx - tw // 2, yy + 10), lab, fill=(0, 0, 0), font=_FONT_EXHIBIT_TICK)
            yy2 = y0 + panel_size[1] - int(frac * panel_size[1])
            draw.line((x0 - 8, yy2, x0, yy2), fill=(0, 0, 0), width=2)
            th = draw.textbbox((0, 0), lab, font=_FONT_EXHIBIT_TICK)[3]
            draw.text((x0 - 58, yy2 - th // 2), lab, fill=(0, 0, 0), font=_FONT_EXHIBIT_TICK)
        _paste_math(
            canvas,
            r"$x/L$",
            (x0 + panel_size[0] // 2, y0 + panel_size[1] + 67),
            fontsize=EXHIBIT_AXIS_LABEL_FONT_SIZE,
            anchor="ma",
            scale_reference=EXHIBIT_AXIS_LABEL_SCALE_REFERENCE,
        )
        _paste_math(
            canvas,
            r"$y/L$",
            (x0 - EXHIBIT_Y_LABEL_X_OFFSET_PX, y0 + panel_size[1] // 2),
            fontsize=EXHIBIT_AXIS_LABEL_FONT_SIZE,
            anchor="mc",
            rotate=90,
            scale_reference=EXHIBIT_AXIS_LABEL_SCALE_REFERENCE,
        )

    field_ticks = [-pattern_vmax, 0.0, pattern_vmax]
    diagnostic_ticks = _movie_ent_ticks(diagnostic_vmax_display)
    _draw_colorbar(
        draw,
        lx + panel_size[0] + 28,
        ly + 10,
        panel_size[1] - 20,
        _FIELD_CMAP_OBJ,
        -pattern_vmax,
        pattern_vmax,
        "",
        field_ticks,
        tick_font=_FONT_EXHIBIT_TICK,
    )
    _draw_colorbar(
        draw,
        rx + panel_size[0] + 28,
        ry + 10,
        panel_size[1] - 20,
        _ENT_CMAP_OBJ,
        0.0,
        diagnostic_vmax_display,
        "",
        diagnostic_ticks,
        tick_font=_FONT_EXHIBIT_TICK,
    )
    _paste_math(
        canvas,
        r"$R^{\rm pat}$",
        (lx + panel_size[0] + 39, ly - 25),
        fontsize=19,
        anchor="mc",
    )
    _paste_math(
        canvas,
        r"$10^3\Delta E_{N,G}^{\rm loc}$",
        (rx + panel_size[0] + 39, ry - 25),
        fontsize=19,
        anchor="mc",
    )
    return np.asarray(canvas)
