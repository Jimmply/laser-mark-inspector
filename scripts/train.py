"""Train the root-cause classifier on data/marks.csv and save it to models/."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from root_cause import train_root_cause  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    cfg = yaml.safe_load((ROOT / "config/settings.yaml").read_text())["model"]
    df = pd.read_csv(ROOT / "data/marks.csv")
    model, res = train_root_cause(df, cfg["test_size"], cfg["random_seed"], cfg["n_estimators"])

    joblib.dump(model, ROOT / "models/root_cause.joblib")
    metrics = {
        "accuracy": round(res.accuracy, 3),
        "grade_only_baseline": round(res.grade_only_baseline, 3),
        "majority_baseline": round(res.majority_baseline, 3),
        "hidden_faults_in_test": res.n_hidden_faults,
        "hidden_fault_recall": round(res.hidden_fault_recall, 3),
        "feature_importances": res.feature_importances.round(3).to_dict(),
    }
    (ROOT / "models/metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("confusion (rows=true, cols=pred, order=%s):\n%s", res.labels, res.confusion)
    logger.info("top features:\n%s", res.feature_importances.round(3).to_string())
