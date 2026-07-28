#!/usr/bin/env python3
"""Portable writers for generated numerical data."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
import os
import zipfile

import numpy as np

_FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def save_npz(path: Path, /, **arrays: object) -> None:
    """Write a compressed NumPy archive with deterministic container fields."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            archive.comment = b""
            for name in sorted(arrays):
                buffer = BytesIO()
                np.lib.format.write_array(
                    buffer,
                    np.asanyarray(arrays[name]),
                    allow_pickle=False,
                )
                info = zipfile.ZipInfo(f"{name}.npy", date_time=_FIXED_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 0
                info.external_attr = 0x20
                info.comment = b""
                info.extra = b""
                archive.writestr(
                    info,
                    buffer.getvalue(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
