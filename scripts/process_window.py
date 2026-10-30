"""Sweep power x speed: where does the mark grade A vs. where does the model call it ok? -> docs/img/process_window.png."""

import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grader import grade_image  # noqa: E402
from mark_generator import ProcessConfig, random_serial, render_mark  # noqa: E402
from root_cause import FEATURES, MISSING  # noqa: E402

POWERS = np.arange(5, 46, 2)
SPEEDS = np.arange(400, 1201, 50)
REPS = 2
CELL = 22
COLORS = {"A": (90, 180, 90), "not A": (80, 80, 210), "ok": (90, 180, 90),
          "under_power": (200, 140, 60), "over_power": (60, 150, 240)}  # BGR


def panel(grid: list[list[str]], title: str) -> np.ndarray:
    h, w = len(grid) * CELL, len(grid[0]) * CELL
    img = np.full((h + 70, w + 60, 3), 255, np.uint8)
    for i, row in enumerate(grid):
        for j, label in enumerate(row):
            y = h - (i + 1) * CELL + 30
            img[y:y + CELL - 1, 50 + j * CELL:50 + (j + 1) * CELL - 1] = COLORS.get(label, (160, 160, 160))
    ni, nj = list(POWERS).index(21), list(SPEEDS).index(800)
    cv2.circle(img, (50 + nj * CELL + CELL // 2, h - ni * CELL - CELL // 2 + 30), 6, (0, 0, 0), 2)
    cv2.putText(img, title, (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1, cv2.LINE_AA)
    for j in (0, len(SPEEDS) // 2, len(SPEEDS) - 1):
        cv2.putText(img, str(SPEEDS[j]), (42 + j * CELL, h + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1, cv2.LINE_AA)
    cv2.putText(img, "scan speed (mm/s)", (50 + w // 2 - 70, h + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1, cv2.LINE_AA)
    for i in (0, len(POWERS) // 2, len(POWERS) - 1):
        cv2.putText(img, f"{POWERS[i]}W", (8, h - i * CELL + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1, cv2.LINE_AA)
    return img


def main() -> None:
    model = joblib.load(ROOT / "models/root_cause.joblib")
    rng = np.random.default_rng(7)
    rows = []
    for p in POWERS:
        for v in SPEEDS:
            for _ in range(REPS):
                rep = grade_image(render_mark(random_serial(rng), ProcessConfig(power_w=float(p), speed_mm_s=float(v)), rng=rng))
                x = pd.DataFrame([{f: getattr(rep, f) if f != "decoded" else int(rep.decoded) for f in FEATURES}])
                rows.append({"p": p, "v": v, "A": rep.overall == "A", "pred": model.predict(x.astype(float).fillna(MISSING))[0]})
    df = pd.DataFrame(rows)
    cells = df.groupby(["p", "v"]).agg(A=("A", "mean"), pred=("pred", lambda s: s.mode()[0]))

    grade_grid = [["A" if cells.loc[(p, v), "A"] >= 0.5 else "not A" for v in SPEEDS] for p in POWERS]
    model_grid = [[cells.loc[(p, v), "pred"] for v in SPEEDS] for p in POWERS]
    out = np.hstack([panel(grade_grid, "Verifier: grades A (green)"),
                     panel(model_grid, "Model: ok / under (blue) / over (orange)")])
    cv2.imwrite(str(ROOT / "docs/img/process_window.png"), out)

    summary = {
        "settings": len(cells),
        "graded_a": round(float((cells["A"] >= 0.5).mean()), 3),
        "model_ok": round(float((cells["pred"] == "ok").mean()), 3),
        "graded_a_but_model_flags": round(float(((cells["A"] >= 0.5) & (cells["pred"] != "ok")).mean()), 3),
    }
    print(json.dumps(summary, indent=2))
    (ROOT / "models/process_window.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
