from __future__ import annotations

import numpy as np
from matplotlib import colormaps, font_manager
from PIL import Image, ImageDraw, ImageFont

import qtp_kernels as q

CMAP_FIELD = q.CMAP_FIELD
VMIN, VMAX = q.VMIN, q.VMAX
FRAME_SIZE = 1024
PANEL_SIZE = 820
PANEL_LEFT = 92
PANEL_TOP = 88
COLORBAR_LEFT = 930
COLORBAR_WIDTH = 24
AXIS_LABEL_FONT_SIZE = 24
AXIS_LABEL_SCALE_REFERENCE = "x/L"
TIMES_FAMILIES = [
    "Times New Roman",
    "Times",
    "TeX Gyre Termes",
    "Nimbus Roman",
    "Liberation Serif",
]
_FIELD_CMAP = colormaps[CMAP_FIELD]


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Resolve a Times-family font without bundling font files."""
    try:
        prop = font_manager.FontProperties(
            family=TIMES_FAMILIES,
            weight="bold" if bold else "normal",
        )
        path = font_manager.findfont(prop, fallback_to_default=True)
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


_FONT_TITLE = _font(38, True)
_FONT_LABEL = _font(28)
_FONT_TICK = _font(21)
_FONT_TIME = _font(28, True)
_FONT_SMALL = _font(19)


def _arr_to_rgb(array: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    scale = max(float(vmax) - float(vmin), np.finfo(float).eps)
    normalized = np.clip((values - float(vmin)) / scale, 0.0, 1.0)
    return (_FIELD_CMAP(normalized)[..., :3] * 255).astype(np.uint8)


def _resize_scalar(array: np.ndarray, size: int) -> np.ndarray:
    source = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")
    return np.asarray(
        source.resize((size, size), Image.Resampling.BICUBIC),
        dtype=float,
    )


def _centered_x(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return int((FRAME_SIZE - (box[2] - box[0])) / 2)


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    *,
    left: int,
    top: int,
    size: int,
    grid_size: int,
) -> Image.Image:
    bottom = top + size
    right = left + size
    draw.rectangle((left, top, right, bottom), outline=(35, 35, 35), width=2)
    ticks = (0, grid_size // 2, grid_size)
    for value in ticks:
        fraction = float(value) / max(float(grid_size), 1.0)
        x = left + int(round(fraction * size))
        y = bottom - int(round(fraction * size))
        draw.line((x, bottom, x, bottom + 8), fill=(20, 20, 20), width=2)
        draw.line((left - 8, y, left, y), fill=(20, 20, 20), width=2)

        x_text = f"{value / max(grid_size, 1):.1f}"
        box = draw.textbbox((0, 0), x_text, font=_FONT_TICK)
        draw.text(
            (x - (box[2] - box[0]) / 2, bottom + 10),
            x_text,
            fill=(20, 20, 20),
            font=_FONT_TICK,
        )

        y_text = f"{value / max(grid_size, 1):.1f}"
        box = draw.textbbox((0, 0), y_text, font=_FONT_TICK)
        draw.text(
            (left - 14 - (box[2] - box[0]), y - (box[3] - box[1]) / 2),
            y_text,
            fill=(20, 20, 20),
            font=_FONT_TICK,
        )

    x_label = "x/L"
    box = draw.textbbox((0, 0), x_label, font=_FONT_LABEL)
    draw.text(
        (left + (size - (box[2] - box[0])) / 2, bottom + 41),
        x_label,
        fill=(15, 15, 15),
        font=_FONT_LABEL,
    )

    y_label = Image.new("RGBA", (60, 120), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text((4, 2), "y/L", fill=(15, 15, 15, 255), font=_FONT_LABEL)
    y_label = y_label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    return y_label


def _draw_colorbar(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    top: int,
    height: int,
    vmin: float,
    vmax: float,
) -> None:
    gradient = np.linspace(vmax, vmin, height, dtype=float)[:, None]
    gradient = np.repeat(gradient, COLORBAR_WIDTH, axis=1)
    bar = Image.fromarray(_arr_to_rgb(gradient, vmin, vmax), mode="RGB")
    canvas.paste(bar, (COLORBAR_LEFT, top))
    draw.rectangle(
        (
            COLORBAR_LEFT,
            top,
            COLORBAR_LEFT + COLORBAR_WIDTH,
            top + height,
        ),
        outline=(35, 35, 35),
        width=2,
    )
    for value in (vmax, 0.0, vmin):
        fraction = (float(vmax) - float(value)) / max(float(vmax) - float(vmin), 1e-12)
        y = top + int(round(fraction * height))
        draw.line(
            (COLORBAR_LEFT + COLORBAR_WIDTH, y, COLORBAR_LEFT + COLORBAR_WIDTH + 7, y),
            fill=(20, 20, 20),
            width=2,
        )
        draw.text(
            (COLORBAR_LEFT + COLORBAR_WIDTH + 10, y - 11),
            f"{value:.2f}",
            fill=(20, 20, 20),
            font=_FONT_SMALL,
        )
    draw.text(
        (COLORBAR_LEFT - 2, top + height + 36),
        "R pattern",
        fill=(20, 20, 20),
        font=_FONT_SMALL,
    )


def pattern_frame(
    result: dict,
    field: np.ndarray,
    physical_time: float,
    *,
    pattern_color_vmax: float = VMAX,
) -> np.ndarray:
    """Render one full-screen pattern frame."""
    field = np.asarray(field, dtype=float)
    grid_size = int(field.shape[-1])
    vmax = abs(float(pattern_color_vmax))
    vmin = -vmax

    canvas = Image.new("RGB", (FRAME_SIZE, FRAME_SIZE), "white")
    draw = ImageDraw.Draw(canvas)
    label = str(result.get("kind", "pattern")).capitalize()
    title = f"{label} pattern"
    draw.text(
        (_centered_x(draw, title, _FONT_TITLE), 20),
        title,
        fill=(12, 12, 12),
        font=_FONT_TITLE,
    )

    resized = _resize_scalar(field, PANEL_SIZE)
    panel = Image.fromarray(_arr_to_rgb(resized, vmin, vmax), mode="RGB")
    canvas.paste(panel, (PANEL_LEFT, PANEL_TOP))
    y_label = _draw_axes(
        draw,
        left=PANEL_LEFT,
        top=PANEL_TOP,
        size=PANEL_SIZE,
        grid_size=grid_size,
    )
    canvas.paste(
        y_label,
        (8, PANEL_TOP + PANEL_SIZE // 2 - y_label.height // 2),
        y_label,
    )
    _draw_colorbar(
        canvas,
        draw,
        top=PANEL_TOP,
        height=PANEL_SIZE,
        vmin=vmin,
        vmax=vmax,
    )

    final_time = float(np.asarray(result.get("times", [physical_time]), dtype=float)[-1])
    time_text = f"t = {float(physical_time):.1f} / {final_time:.1f}"
    draw.text(
        (_centered_x(draw, time_text, _FONT_TIME), 978),
        time_text,
        fill=(12, 12, 12),
        font=_FONT_TIME,
    )
    return np.asarray(canvas, dtype=np.uint8)


def movie_frame(
    result: dict,
    field: np.ndarray,
    physical_time: float,
    pattern_color_vmax: float = VMAX,
) -> np.ndarray:
    """Render one pattern movie frame."""
    return pattern_frame(
        result,
        field,
        physical_time,
        pattern_color_vmax=pattern_color_vmax,
    )
