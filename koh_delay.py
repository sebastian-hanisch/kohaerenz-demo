"""Verzögerungsschätzer aus dem Kreuzspektrum (verallgemeinerte Kreuzkorrelation, Knapp und Carter 1976) auf einem feinen Gitter.

Gesucht ist d mit maximalem Re sum_k s_k exp(i 2 pi f_k d); die Gewichtung s_k: Kreuzspektrum selbst (CC), nur die Phase (PHAT) oder Phase mal c / (1 - c) (ML, kohärenzgewichtet).
Das Gitter hat die Schrittweite DELAY_STEP; Fehler unter dieser Größenordnung sind nicht aussagekräftig.
"""

import functools

import numpy as np

import koh_constants as C
from koh_signals import frac_delay

METHODS = ("CC", "PHAT", "ML")
METHOD_LABELS = {"CC": "Kreuzkorrelation", "PHAT": "nur Phase (PHAT)", "ML": "kohärenzgewichtet (ML)"}


def delay_grid(half: float = C.DELAY_RANGE, step: float = C.DELAY_STEP) -> np.ndarray:
    n = int(round(2 * half / step))
    return -half + step * np.arange(n + 1)


@functools.lru_cache(maxsize=8)
def _basis(seg: int, half: float, step: float):
    f = np.arange(seg // 2 + 1) / seg
    grid = delay_grid(half, step)
    return grid, np.exp(2j * np.pi * np.outer(f, grid))


def weights(sxy, coh, method: str) -> np.ndarray:
    """Gewichtetes Kreuzspektrum je Bin (letzte Achse): CC unverändert, PHAT auf Betrag 1, ML mit c / (1 - c) (c auf [1e-6, 1 - 1e-6] begrenzt)."""
    sxy = np.asarray(sxy)
    if method == "CC":
        return sxy
    unit = sxy / np.maximum(np.abs(sxy), 1e-300)
    if method == "PHAT":
        return unit
    if method == "ML":
        c = np.clip(np.asarray(coh, float), 1e-6, 1.0 - 1e-6)
        return unit * c / (1.0 - c)
    raise ValueError(f"unbekannter Schätzer: {method}")


def estimate(sxy, coh, seg: int, method: str, half: float = C.DELAY_RANGE, step: float = C.DELAY_STEP):
    """Verzögerung (Abtastwerte) je Zeile von sxy (…, Bins): Maximum des gewichteten Kreuzspektrums über das Gitter."""
    grid, basis = _basis(seg, float(half), float(step))
    s = weights(sxy, coh, method)
    obj = np.real(s @ basis)
    return grid[np.argmax(obj, axis=-1)]


def align(y, d: float) -> np.ndarray:
    """y um d zurückschieben (zirkulär, auch gebrochen): hebt eine bekannte Verzögerung auf."""
    return frac_delay(y, -d)
