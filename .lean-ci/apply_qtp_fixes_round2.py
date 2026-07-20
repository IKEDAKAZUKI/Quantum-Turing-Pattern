#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
path = root / "QuantumTuringPatterns/Basic.lean"
old = """  simpa [explicitOmega0] using
    Real.sq_sqrt (mul_nonneg (by norm_num) sqrt3_nonneg)
"""
new = """  simpa [explicitOmega0] using
    Real.sq_sqrt (mul_nonneg (show (0 : ℝ) ≤ 2 by norm_num) sqrt3_nonneg)
"""
text = path.read_text(encoding="utf-8")
if text.count(old) != 1:
    raise RuntimeError(f"unexpected match count: {text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")

paths = [
    root / "QuantumTuringPatterns.lean",
    root / "lakefile.toml",
    root / "lean-toolchain",
    root / "scripts/static_audit.py",
    *sorted((root / "QuantumTuringPatterns").glob("*.lean")),
]
h = hashlib.sha256()
for item in paths:
    rel = item.relative_to(root).as_posix().encode()
    data = item.read_bytes()
    h.update(len(rel).to_bytes(8, "big")); h.update(rel)
    h.update(len(data).to_bytes(8, "big")); h.update(data)
actual = h.hexdigest()
expected = "1ee71f39c6dc9b6258225fa048c9626bda4ee1c5d9c4bf1bca261e526bb4f028"
print(f"TREE_SHA256={actual}")
if actual != expected:
    raise RuntimeError(f"tree digest mismatch: expected {expected}, got {actual}")
