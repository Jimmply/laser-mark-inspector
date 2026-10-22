import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mark_generator import FAULTS, PhysicsConfig, ProcessConfig, dose, render_mark, sample_fault  # noqa: E402


def test_render_is_reproducible_with_seed():
    a = render_mark("SN-1", ProcessConfig(), rng=np.random.default_rng(3))
    b = render_mark("SN-1", ProcessConfig(), rng=np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_render_returns_uint8_grayscale():
    img = render_mark("SN-1", ProcessConfig(), rng=np.random.default_rng(0))
    assert img.dtype == np.uint8 and img.ndim == 2


def test_nominal_settings_give_unit_dose():
    assert dose(ProcessConfig(), PhysicsConfig()) == pytest.approx(1.0)


def test_galvo_scale_error_stretches_width():
    base = render_mark("SN-1", ProcessConfig(), rng=np.random.default_rng(0))
    wide = render_mark("SN-1", ProcessConfig(galvo_scale_err=0.1), rng=np.random.default_rng(0))
    assert wide.shape[1] > base.shape[1]


def test_more_power_gives_brighter_marks_on_anodize():
    lo = render_mark("SN-1", ProcessConfig(power_w=9), rng=np.random.default_rng(0)).astype(float)
    hi = render_mark("SN-1", ProcessConfig(power_w=20), rng=np.random.default_rng(0)).astype(float)
    assert np.percentile(hi, 95) > np.percentile(lo, 95)


@pytest.mark.parametrize("fault", FAULTS)
def test_every_fault_is_sampleable(fault):
    proc = sample_fault(fault, np.random.default_rng(0))
    assert proc.power_w > 0 and proc.speed_mm_s > 0


def test_unknown_fault_raises():
    with pytest.raises(ValueError):
        sample_fault("laser_on_fire", np.random.default_rng(0))
