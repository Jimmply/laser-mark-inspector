import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grader import AN_BANDS, CC_BANDS, grade_higher, grade_image, grade_lower  # noqa: E402
from mark_generator import ProcessConfig, render_mark  # noqa: E402


def _grade(**kw):
    return grade_image(render_mark("LSR-000001-AL6061", ProcessConfig(**kw), rng=np.random.default_rng(1)))


def test_grade_bands_match_standard_boundaries():
    assert grade_higher(0.30, CC_BANDS) == "A"
    assert grade_higher(0.299, CC_BANDS) == "B"
    assert grade_higher(0.14, CC_BANDS) == "F"
    assert grade_lower(0.06, AN_BANDS) == "A"
    assert grade_lower(0.061, AN_BANDS) == "B"
    assert grade_lower(0.13, AN_BANDS) == "F"


def test_nominal_mark_decodes_and_grades_a():
    rep = _grade()
    assert rep.decoded and rep.text == "LSR-000001-AL6061"
    assert rep.overall == "A"
    assert rep.module_agreement > 0.95


def test_overall_is_worst_parameter():
    rep = _grade(galvo_scale_err=0.11)
    assert rep.grades["axial_nonuniformity"] in "CDF"
    assert rep.overall == max(rep.grades.values(), key="ABCDF".index)


def test_galvo_error_raises_axial_nonuniformity():
    assert _grade(galvo_scale_err=0.12).axial_nonuniformity > _grade().axial_nonuniformity + 0.05


def test_defocus_lowers_sharpness():
    assert _grade(defocus_mm=1.2).sharpness < _grade().sharpness


def test_blank_image_fails_decode():
    rep = grade_image(np.full((120, 120), 15, np.uint8))
    assert not rep.decoded and rep.overall == "F"
