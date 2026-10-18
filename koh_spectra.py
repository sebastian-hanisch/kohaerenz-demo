"""Kreuzspektrum nach Welch: Segmente mit Hann-Fenster, S_xx, S_yy, S_xy = E[conj(X) Y], Kohärenz, Phase und die Übertragungsfunktionen H1 und H2.

Konvention: Die Phase von S_xy ist die von H = S_xy / S_xx; für y[t] = x[t - d] ist sie -2 pi f d. Frequenzen f = k / seg, k = 0 .. seg / 2.
"""

import numpy as np


def hann(L: int) -> np.ndarray:
    """Hann-Fenster 0,5 - 0,5 cos(2 pi (m + 1/2) / L), symmetrisch um die Mitte."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(L) + 0.5) / L)


def n_segments(n: int, seg: int, hop: int) -> int:
    """Zahl der vollständigen Segmente der Länge seg im Abstand hop in einer Folge der Länge n."""
    if seg > n or hop < 1:
        raise ValueError("Segment länger als die Folge oder Schrittweite < 1")
    return (n - seg) // hop + 1


def hop_for(seg: int, overlap: float) -> int:
    """Schrittweite in Abtastwerten für den Überlappungsanteil overlap in [0, 1)."""
    return max(1, int(round(seg * (1.0 - overlap))))


def record_length(seg: int, hop: int, k: int) -> int:
    """Länge der Folge, die genau k Segmente im Abstand hop enthält."""
    return seg + (k - 1) * hop


def cross_spectra(x, y, seg: int, hop: int, w=None):
    """(S_xx, S_yy, S_xy, K): Mittel der gefensterten Segment-Spektren, Skalierung |X|^2 / sum w^2, rfft-Bins 0 .. seg/2; x und y mit gleicher Form (…, Zeit)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = hann(seg) if w is None else np.asarray(w, float)
    if len(w) != seg:
        raise ValueError("Fensterlänge ungleich Segmentlänge")
    xs = np.lib.stride_tricks.sliding_window_view(x, seg, axis=-1)[..., ::hop, :] * w
    ys = np.lib.stride_tricks.sliding_window_view(y, seg, axis=-1)[..., ::hop, :] * w
    X, Y = np.fft.rfft(xs, axis=-1), np.fft.rfft(ys, axis=-1)
    norm = float((w ** 2).sum())
    return (np.abs(X) ** 2).mean(-2) / norm, (np.abs(Y) ** 2).mean(-2) / norm, (np.conj(X) * Y).mean(-2) / norm, X.shape[-2]


def coherence(sxx, syy, sxy) -> np.ndarray:
    """Betragsquadrat-Kohärenz |S_xy|^2 / (S_xx S_yy); 0, wo ein Nenner 0 ist."""
    den = np.asarray(sxx) * np.asarray(syy)
    num = np.abs(sxy) ** 2
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def phase(sxy) -> np.ndarray:
    return np.angle(sxy)


def transfer_h1(sxx, sxy) -> np.ndarray:
    """H1 = S_xy / S_xx (Rauschen am Ausgang)."""
    return np.asarray(sxy) / np.asarray(sxx)


def transfer_h2(syy, sxy) -> np.ndarray:
    """H2 = S_yy / S_yx mit S_yx = conj(S_xy) (Rauschen am Eingang)."""
    return np.asarray(syy) / np.conj(sxy)


def freqs(seg: int) -> np.ndarray:
    return np.arange(seg // 2 + 1) / seg
