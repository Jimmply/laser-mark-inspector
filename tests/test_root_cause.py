import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_data import generate  # noqa: E402
from mark_generator import PhysicsConfig  # noqa: E402
from root_cause import FEATURES, train_root_cause  # noqa: E402


def test_features_never_use_laser_settings_or_label():
    assert not any(f.startswith("proc_") or f == "fault" for f in FEATURES)


def test_classifier_beats_grade_only_baseline():
    df = generate(240, PhysicsConfig(), random_seed=0)
    _, res = train_root_cause(df, n_estimators=50)
    assert res.accuracy > res.grade_only_baseline > res.majority_baseline
