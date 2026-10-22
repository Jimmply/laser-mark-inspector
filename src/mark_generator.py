"""Synthetic laser-marked DataMatrix images on black-anodized aluminum."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from ppf.datamatrix import DataMatrix

FAULTS = ["ok", "under_power", "over_power", "defocus", "galvo_scale", "pulse_dropout"]


@dataclass
class ProcessConfig:
    """Laser settings and machine condition for one mark."""
    power_w: float = 20.0
    speed_mm_s: float = 800.0
    defocus_mm: float = 0.0
    galvo_scale_err: float = 0.0
    dropout_rate: float = 0.0


@dataclass
class PhysicsConfig:
    """Anodize response to laser dose."""
    power_nom: float = 20.0
    speed_nom: float = 800.0
    bleach_threshold: float = 0.35
    bleach_scale: float = 0.25
    burn_through: float = 1.7
    r_anodize: float = 0.06
    r_bleached: float = 0.70
    r_bare_al: float = 0.30
    beam_sigma_px: float = 0.6
    defocus_sigma_px_per_mm: float = 3.0
    cell_px: int = 10
    quiet_zone_modules: int = 2


def module_matrix(text: str) -> np.ndarray:
    return np.array(DataMatrix(text).matrix, dtype=np.uint8)


def dose(proc: ProcessConfig, phys: PhysicsConfig) -> float:
    return (proc.power_w / phys.power_nom) * (phys.speed_nom / proc.speed_mm_s)


def reflectance(d: np.ndarray, phys: PhysicsConfig, rng: np.random.Generator) -> np.ndarray:
    dye_batch = rng.uniform(0.85, 1.15)   # anodize dye / seal varies batch to batch
    bleach = np.clip(1 - np.exp(-(d - phys.bleach_threshold) / phys.bleach_scale), 0, 1)
    r = phys.r_anodize * dye_batch + bleach * (phys.r_bleached * dye_batch - phys.r_anodize * dye_batch)
    burn = np.clip((d - phys.burn_through) / 0.3, 0, 1)
    bare_al = phys.r_bare_al + rng.normal(0, 0.10, d.shape)   # specular grain of exposed Al
    return (1 - burn) * r + burn * bare_al


def render_mark(
    text: str,
    proc: ProcessConfig,
    phys: PhysicsConfig | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a uint8 grayscale image of the marked symbol."""
    phys = phys or PhysicsConfig()
    rng = rng or np.random.default_rng(0)

    m = module_matrix(text).astype(float)
    if proc.dropout_rate > 0:
        dropped = (rng.random(m.shape) < proc.dropout_rate) & (m > 0)
        m[dropped] *= rng.uniform(0.0, 0.3, dropped.sum())

    q = phys.quiet_zone_modules
    d = np.kron(np.pad(m, q), np.ones((phys.cell_px, phys.cell_px))) * dose(proc, phys)

    sigma = phys.beam_sigma_px + phys.defocus_sigma_px_per_mm * abs(proc.defocus_mm)
    d = cv2.GaussianBlur(d, (0, 0), sigma)

    r = reflectance(d, phys, rng)

    h, w = r.shape
    if proc.galvo_scale_err:
        r = cv2.resize(r, (int(round(w * (1 + proc.galvo_scale_err))), h), interpolation=cv2.INTER_LINEAR)
        h, w = r.shape

    angle = rng.uniform(-3, 3)
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    r = cv2.warpAffine(r, rot, (w, h), borderValue=phys.r_anodize)

    mottle = cv2.GaussianBlur(rng.normal(0, 1, r.shape), (0, 0), 6) * 0.04
    gradient = np.linspace(-1, 1, w)[None, :] * rng.uniform(0, 0.03)
    r = r + mottle + gradient + rng.normal(0, 0.01, r.shape)

    exposure = rng.uniform(0.75, 1.25)   # camera / lighting varies shot to shot, not a laser fault
    return np.clip(r * exposure / 0.8 * 255, 0, 255).astype(np.uint8)


def sample_fault(fault: str, rng: np.random.Generator) -> ProcessConfig:
    """Draw process settings for a scenario: in-spec drift plus one dominant fault."""
    proc = ProcessConfig(
        power_w=rng.uniform(18, 22),
        speed_mm_s=rng.uniform(750, 850),
        defocus_mm=rng.uniform(-0.1, 0.1),
        galvo_scale_err=rng.uniform(-0.01, 0.01),
    )
    sign = rng.choice([-1, 1])
    # Severity ranges start just outside the in-spec window, so mild faults
    # look almost like normal variation (as they do on a real line).
    if fault == "under_power":
        proc.power_w = rng.uniform(7, 16)
    elif fault == "over_power":
        proc.power_w = rng.uniform(25, 45)
    elif fault == "defocus":
        proc.defocus_mm = sign * rng.uniform(0.25, 1.6)
    elif fault == "galvo_scale":
        proc.galvo_scale_err = sign * rng.uniform(0.02, 0.14)
    elif fault == "pulse_dropout":
        proc.dropout_rate = rng.uniform(0.01, 0.18)
    elif fault != "ok":
        raise ValueError(f"unknown fault {fault!r}")
    return proc


def random_serial(rng: np.random.Generator) -> str:
    return f"LSR-{rng.integers(0, 10**6):06d}-{rng.choice(['AL6061', 'AL7075', 'AL2024'])}"
