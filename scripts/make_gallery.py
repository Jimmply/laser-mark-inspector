"""Render one mark per fault with its grade and predicted cause -> docs/img/fault_gallery.png."""

import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grader import grade_image  # noqa: E402
from mark_generator import FAULTS, render_mark, sample_fault  # noqa: E402
from root_cause import FEATURES, MISSING  # noqa: E402

model = joblib.load(ROOT / "models/root_cause.joblib")
rng = np.random.default_rng(11)
tiles = []
for fault in FAULTS:
    img = render_mark("LSR-004521-AL6061", sample_fault(fault, rng), rng=rng)
    rep = grade_image(img)
    x = pd.DataFrame([{f: getattr(rep, f) if f != "decoded" else int(rep.decoded) for f in FEATURES}]).astype(float).fillna(MISSING)
    pred = model.predict(x)[0]
    tile = cv2.copyMakeBorder(cv2.resize(img, (240, 240)), 0, 62, 0, 0, cv2.BORDER_CONSTANT, value=255)
    for i, line in enumerate([f"true: {fault}", f"grade: {rep.overall}", f"model: {pred}"]):
        cv2.putText(tile, line, (6, 258 + 19 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
    tiles.append(cv2.copyMakeBorder(tile, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255))
cv2.imwrite(str(ROOT / "docs/img/fault_gallery.png"), np.hstack(tiles))
print("saved docs/img/fault_gallery.png")
