"""DPM verification of a DataMatrix image, following ISO/IEC TR 29158 (AIM DPM) grade bands.

Graded: Decode, Cell Contrast, Axial Non-uniformity, Unused Error Correction.
Not implemented: Cell Modulation, Fixed Pattern Damage, Grid Non-uniformity,
Minimum Reflectance. This is a teaching/portfolio verifier, not a calibrated one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
import zxingcpp

from mark_generator import module_matrix

logger = logging.getLogger(__name__)

GRADES = "ABCDF"
CC_BANDS = (0.30, 0.25, 0.20, 0.15)          # higher is better
UEC_BANDS = (0.62, 0.50, 0.37, 0.25)         # higher is better
AN_BANDS = (0.06, 0.08, 0.10, 0.12)          # lower is better
SAMPLE_PX = 12


@dataclass
class GradeReport:
    """Verification result for one mark."""
    decoded: bool
    text: str | None = None
    symbol_size: str | None = None
    cell_contrast: float | None = None
    axial_nonuniformity: float | None = None
    unused_ec: float | None = None
    module_agreement: float | None = None
    light_mean: float | None = None
    dark_mean: float | None = None
    light_cv: float | None = None
    dark_cv: float | None = None
    sharpness: float | None = None
    dot_growth: float | None = None
    grades: dict[str, str] = field(default_factory=dict)
    overall: str = "F"


def grade_higher(value: float, bands: tuple[float, ...]) -> str:
    for g, t in zip(GRADES, bands):
        if value >= t:
            return g
    return "F"


def grade_lower(value: float, bands: tuple[float, ...]) -> str:
    for g, t in zip(GRADES, bands):
        if value <= t:
            return g
    return "F"


def _corners(result) -> np.ndarray:
    p = result.position
    return np.array([[c.x, c.y] for c in (p.top_left, p.top_right, p.bottom_right, p.bottom_left)], np.float32)


def _axial_nonuniformity(corners: np.ndarray) -> float:
    tl, tr, br, bl = corners
    x = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    y = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return float(abs(x - y) / ((x + y) / 2))


def _sample_modules(warped: np.ndarray, n: int) -> np.ndarray:
    c = SAMPLE_PX
    lo, hi = int(c * 0.3), int(c * 0.7)
    blocks = warped[: n * c, : n * c].reshape(n, c, n, c)[:, lo:hi, :, lo:hi]
    return blocks.mean(axis=(1, 3))


def _fallback_stats(img: np.ndarray) -> dict:
    t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    light, dark = img[img > t].astype(float), img[img <= t].astype(float)
    if light.size == 0 or dark.size == 0:
        return {}
    lm, dm = light.mean(), dark.mean()
    return {
        "cell_contrast": (lm - dm) / lm if lm else 0.0,
        "light_mean": lm, "dark_mean": dm,
        "light_cv": light.std() / lm if lm else 0.0,
        "dark_cv": dark.std() / dm if dm else 0.0,
        "sharpness": float(cv2.Laplacian(img, cv2.CV_64F).var() / max(lm - dm, 1) ** 2),
    }


def grade_image(img: np.ndarray) -> GradeReport:
    """Decode and grade a grayscale uint8 image of a DataMatrix."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found = zxingcpp.read_barcodes(img, formats=zxingcpp.BarcodeFormat.DataMatrix)
    if not found:
        rep = GradeReport(decoded=False, **_fallback_stats(img))
        rep.grades = {"decode": "F"}
        return rep

    res = found[0]
    ideal = module_matrix(res.text)
    n = ideal.shape[0]
    size = res.extra.get("Version")

    side = n * SAMPLE_PX
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], np.float32)
    corners = _corners(res)
    warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(corners, dst), (side, side)).astype(float)
    samples = _sample_modules(warped, n)

    best = None
    for k in range(4):
        cand = np.rot90(ideal, k)
        on, off = samples[cand == 1], samples[cand == 0]
        light_is_on = on.mean() >= off.mean()
        light, dark = (on, off) if light_is_on else (off, on)
        gt = (light.mean() + dark.mean()) / 2
        read = (samples > gt) if light_is_on else (samples <= gt)
        agree = float((read == (cand == 1)).mean())
        if best is None or agree > best[0]:
            best = (agree, light, dark, gt, cand, light_is_on)
    agree, light, dark, gt, cand, light_is_on = best

    lm, dm = float(light.mean()), float(dark.mean())
    marked_frac = float((cand == 1).mean() if light_is_on else (cand == 0).mean())
    lit_frac = float((warped > gt).mean())

    rep = GradeReport(
        decoded=True,
        text=res.text,
        symbol_size=size,
        cell_contrast=(lm - dm) / lm if lm else 0.0,
        axial_nonuniformity=_axial_nonuniformity(corners),
        unused_ec=float(res.extra["UEC"]) if "UEC" in res.extra else None,
        module_agreement=agree,
        light_mean=lm,
        dark_mean=dm,
        light_cv=float(light.std() / lm) if lm else 0.0,
        dark_cv=float(dark.std() / dm) if dm else 0.0,
        sharpness=float(cv2.Laplacian(warped, cv2.CV_64F).var() / max(lm - dm, 1) ** 2),
        dot_growth=lit_frac / marked_frac - 1 if marked_frac else 0.0,
    )
    if size and size != f"{n}x{n}":
        logger.warning("decoder size %s != re-encoded %dx%d; grid-based measures may be off", size, n, n)

    rep.grades = {
        "decode": "A",
        "cell_contrast": grade_higher(rep.cell_contrast, CC_BANDS),
        "axial_nonuniformity": grade_lower(rep.axial_nonuniformity, AN_BANDS),
    }
    if rep.unused_ec is not None:
        rep.grades["unused_ec"] = grade_higher(rep.unused_ec, UEC_BANDS)
    rep.overall = max(rep.grades.values(), key=GRADES.index)
    return rep
