"""Render N marks across fault scenarios, grade each one, save features to data/marks.csv."""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grader import grade_image  # noqa: E402
from mark_generator import FAULTS, PhysicsConfig, random_serial, render_mark, sample_fault  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def physics_from_config(cfg: dict) -> PhysicsConfig:
    return PhysicsConfig(
        power_nom=cfg["process_nominal"]["power_w"],
        speed_nom=cfg["process_nominal"]["speed_mm_s"],
        cell_px=cfg["render"]["cell_px"],
        quiet_zone_modules=cfg["render"]["quiet_zone_modules"],
        **cfg["physics"],
    )


def generate(n: int, phys: PhysicsConfig, random_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows = []
    for i in range(n):
        fault = FAULTS[i % len(FAULTS)]
        proc = sample_fault(fault, rng)
        rep = grade_image(render_mark(random_serial(rng), proc, phys, rng))
        row = {"fault": fault, **{f"proc_{k}": v for k, v in asdict(proc).items()}}
        row.update({k: v for k, v in asdict(rep).items() if k not in ("grades", "text")})
        row["decoded"] = int(rep.decoded)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfg = yaml.safe_load((ROOT / "config/settings.yaml").read_text())
    df = generate(cfg["dataset"]["n_samples"], physics_from_config(cfg), cfg["dataset"]["random_seed"])
    out = ROOT / "data/marks.csv"
    df.to_csv(out, index=False)
    logger.info("wrote %d rows -> %s", len(df), out)
    logger.info("overall grade by fault:\n%s", pd.crosstab(df["fault"], df["overall"]))
