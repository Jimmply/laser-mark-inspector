"""Streamlit: mark a DataMatrix on black anodize, grade it, and suggest the likely process fault."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grader import AN_BANDS, CC_BANDS, UEC_BANDS, grade_image  # noqa: E402
from mark_generator import ProcessConfig, render_mark  # noqa: E402
from root_cause import FEATURES, MISSING  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parents[1] / "models/root_cause.joblib"
ACTIONS = {
    "ok": "No action. Process in spec.",
    "under_power": "Raise power or slow the scan; check laser output with a power meter.",
    "over_power": "Lower power or speed up; burn-through exposes bare Al and hurts readability.",
    "defocus": "Re-check focal height / Z offset and part fixturing.",
    "galvo_scale": "Re-run galvo field calibration; X and Y scale disagree.",
    "pulse_dropout": "Inspect Q-switch / seed laser and the fiber connection for missed pulses.",
}

st.set_page_config(page_title="Laser Mark Inspector", page_icon="🔎", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


with st.sidebar:
    st.header("Source")
    upload = st.file_uploader("Grade your own mark photo (PNG/JPG)", type=["png", "jpg", "jpeg"])
    st.header("Simulated laser settings")
    serial = st.text_input("Serial to mark", "LSR-004521-AL6061")
    power = st.slider("Power (W)", 5.0, 45.0, 20.0, 0.5)
    speed = st.slider("Scan speed (mm/s)", 400.0, 1200.0, 800.0, 10.0)
    defocus = st.slider("Defocus (mm)", -1.6, 1.6, 0.0, 0.05)
    galvo = st.slider("Galvo X-scale error", -0.15, 0.15, 0.0, 0.005)
    dropout = st.slider("Pulse dropout rate", 0.0, 0.2, 0.0, 0.01)
    seed = st.number_input("Random seed", 0, 9999, 7)

if upload is not None:
    img = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
else:
    proc = ProcessConfig(power, speed, defocus, galvo, dropout)
    img = render_mark(serial, proc, rng=np.random.default_rng(int(seed)))

rep = grade_image(img)

st.title("🔎 Laser Mark Inspector")
st.caption("DataMatrix on black-anodized aluminum, graded with ISO/IEC TR 29158 (AIM DPM) bands.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall grade", rep.overall)
c2.metric("Decoded", rep.text or "no read")
c3.metric("Cell contrast", f"{rep.cell_contrast:.2f}" if rep.cell_contrast is not None else "–")
c4.metric("Axial non-uniformity", f"{rep.axial_nonuniformity:.3f}" if rep.axial_nonuniformity is not None else "–")

left, right = st.columns([1, 1])
with left:
    st.image(img, caption="Mark as seen by the verifier camera", width="stretch")
with right:
    rows = [
        ("Decode", "pass" if rep.decoded else "fail", "A or F", rep.grades.get("decode", "F")),
        ("Cell contrast", rep.cell_contrast, f"A ≥ {CC_BANDS[0]}", rep.grades.get("cell_contrast", "–")),
        ("Axial non-uniformity", rep.axial_nonuniformity, f"A ≤ {AN_BANDS[0]}", rep.grades.get("axial_nonuniformity", "–")),
        ("Unused error correction", rep.unused_ec, f"A ≥ {UEC_BANDS[0]}", rep.grades.get("unused_ec", "–")),
    ]
    table = pd.DataFrame(rows, columns=["Parameter", "Value", "A-grade threshold", "Grade"])
    table["Value"] = table["Value"].map(lambda v: v if isinstance(v, str) else ("–" if v is None else f"{v:.3f}"))
    st.dataframe(table, hide_index=True, width="stretch")

    model = load_model()
    if model is None:
        st.info("Root-cause model not trained yet. Run `python scripts/generate_data.py && python scripts/train.py`.")
    else:
        x = pd.DataFrame([{f: getattr(rep, f) if f != "decoded" else int(rep.decoded) for f in FEATURES}]).astype(float).fillna(MISSING)
        probs = pd.Series(model.predict_proba(x)[0], index=model.classes_).sort_values(ascending=False)
        top = probs.index[0]
        st.subheader(f"Likely cause: {top.replace('_', ' ')} ({probs.iloc[0]:.0%})")
        st.write(ACTIONS[top])
        st.bar_chart(probs)
