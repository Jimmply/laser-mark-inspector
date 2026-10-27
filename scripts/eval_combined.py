"""Stress test: grade + classify marks with TWO simultaneous faults (never seen in training)."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grader import grade_image  # noqa: E402
from mark_generator import FAULTS, random_serial, render_mark, sample_fault  # noqa: E402
from root_cause import FEATURES, MISSING  # noqa: E402

FIELD = {
    "under_power": "power_w", "over_power": "power_w", "defocus": "defocus_mm",
    "galvo_scale": "galvo_scale_err", "pulse_dropout": "dropout_rate",
}
PER_PAIR = 40


def main() -> None:
    model = joblib.load(ROOT / "models/root_cause.joblib")
    rng = np.random.default_rng(2026)
    pairs = [p for p in itertools.combinations(FIELD, 2) if FIELD[p[0]] != FIELD[p[1]]]

    rows = []
    for a, b in pairs:
        for _ in range(PER_PAIR):
            proc = sample_fault(a, rng)
            setattr(proc, FIELD[b], getattr(sample_fault(b, rng), FIELD[b]))
            rep = grade_image(render_mark(random_serial(rng), proc, rng=rng))
            x = pd.DataFrame([{f: getattr(rep, f) if f != "decoded" else int(rep.decoded) for f in FEATURES}])
            pred = model.predict(x.astype(float).fillna(MISSING))[0]
            rows.append({"pair": f"{a} + {b}", "grade": rep.overall, "pred": pred, "hit": pred in (a, b)})

    df = pd.DataFrame(rows)
    by_pair = df.groupby("pair").agg(
        graded_a=("grade", lambda g: (g == "A").mean()),
        flagged=("pred", lambda p: (p != "ok").mean()),
        names_a_true_fault=("hit", "mean"),
    ).round(2)
    print(by_pair.to_string())

    summary = {
        "marks": len(df),
        "pairs": len(pairs),
        "graded_a": round(float((df["grade"] == "A").mean()), 3),
        "flagged_faulty": round(float((df["pred"] != "ok").mean()), 3),
        "names_a_true_fault": round(float(df["hit"].mean()), 3),
    }
    print(json.dumps(summary, indent=2))
    (ROOT / "models/combined_eval.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
