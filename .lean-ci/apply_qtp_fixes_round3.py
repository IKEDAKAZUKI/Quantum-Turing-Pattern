#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
path = root / "QuantumTuringPatterns/ExplicitStripe.lean"
text = path.read_text(encoding="utf-8")
replacements = [
    (
        """    reducedStationaryFactor lam ν B = 0 := by
  rw [hB]
  unfold reducedStationaryFactor leadingAmplitudeSq
""",
        """    reducedStationaryFactor lam ν B = 0 := by
  unfold reducedStationaryFactor
  rw [hB]
  unfold leadingAmplitudeSq
""",
    ),
    (
        """    reducedRadialDerivative lam ν B = -lam / 2 := by
  rw [hB]
  unfold reducedRadialDerivative leadingAmplitudeSq
""",
        """    reducedRadialDerivative lam ν B = -lam / 2 := by
  unfold reducedRadialDerivative
  rw [hB]
  unfold leadingAmplitudeSq
""",
    ),
]
for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"unexpected match count {count} for replacement beginning {old[:60]!r}")
    text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")

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
expected = "2fe60153a1205110e7e30ea2aaaa97188a38fa52854d3e464972ecc49780a2cf"
print(f"TREE_SHA256={actual}")
if actual != expected:
    raise RuntimeError(f"tree digest mismatch: expected {expected}, got {actual}")
