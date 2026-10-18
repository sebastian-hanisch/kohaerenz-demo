"""Wavelet-Kohärenz nach Torrence und Webster (1999) mit dem analytischen Morlet-Wavelet.

W(s, n) = sum_m x[m] conj(psi((m - n) / s)) / sqrt(s), psi(t) = pi^(-1/4) exp(i w0 t) exp(-t^2 / 2), Frequenz f = w0 / (2 pi s) (Abtastrate 1).
Kohärenz R^2 = |S(conj(W_x) W_y)|^2 / (S(|W_x|^2) S(|W_y|^2)) mit der Glättung S in Zeit (Gauß, Std = tw mal Skala) und Skala (Kasten über ns benachbarte Frequenzen); ohne Glättung ist R^2 = 1.
"""

import math

import numpy as np

import koh_constants as C

TWO_PI = 2.0 * math.pi
PI14 = math.pi ** -0.25


def scale_of(f, w0: float = C.W0):
    """Skala s = w0 / (2 pi f)."""
    return w0 / (TWO_PI * np.asarray(f, dtype=float))


def freq_grid(fmin: float = C.FMIN, fmax: float = C.FMAX, voices: int = C.VOICES) -> np.ndarray:
    """Logarithmisches Frequenzgitter, aufsteigend: fmax * 2^(-k / voices), k = 0 .. K mit fmin unterschritten oder erreicht."""
    if not 0.0 < fmin < fmax or voices < 1:
        raise ValueError("Frequenzgitter braucht 0 < fmin < fmax und voices >= 1")
    k = int(math.ceil(voices * math.log2(fmax / fmin)))
    return (fmax * 2.0 ** (-np.arange(k + 1) / voices))[::-1]


def cwt(x, freqs, w0: float = C.W0) -> np.ndarray:
    """Transformation per FFT (nur positive Frequenzen, Nullen bis zur Zweierpotenz >= 2n angehängt): Form (…, Frequenzen, Zeit), komplex."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    size = 1 << int(math.ceil(math.log2(2 * n)))
    spec = np.fft.fft(x, size, axis=-1)
    w = TWO_PI * np.fft.fftfreq(size)
    out = np.empty(x.shape[:-1] + (len(freqs), n), dtype=complex)
    for i, f in enumerate(freqs):
        s = float(scale_of(f, w0))
        psi = np.where(w > 0, PI14 * math.sqrt(TWO_PI * s) * np.exp(-0.5 * (s * w - w0) ** 2), 0.0)
        out[..., i, :] = np.fft.ifft(spec * psi, axis=-1)[..., :n]
    return out


def cwt_direct(x, f: float, w0: float = C.W0) -> np.ndarray:
    """Definition als Summe (Gegenprobe), eine Frequenz, eine Reihe: W(n) = sum_m x[m] conj(psi((m - n) / s)) / sqrt(s) mit dem abgetasteten Wavelet."""
    x = np.asarray(x, float)
    n = len(x)
    s = float(scale_of(f, w0))
    m = np.arange(n)
    out = np.empty(n, complex)
    for k in range(n):
        t = (m - k) / s
        out[k] = np.sum(x * np.conj(PI14 * np.exp(1j * w0 * t) * np.exp(-0.5 * t * t))) / math.sqrt(s)
    return out


def smooth(P, freqs, tw: float, ns: int, w0: float = C.W0) -> np.ndarray:
    """Glättung von (…, Frequenz, Zeit): Gauß in der Zeit mit Std tw * s (s = Skala in Abtastwerten, Nullen am Rand), dann Kasten über ns benachbarte Frequenzen (Randzeilen wiederholt)."""
    P = np.asarray(P)
    n = P.shape[-1]
    out = np.empty_like(P)
    for i, f in enumerate(freqs):
        s = float(scale_of(f, w0))
        if tw > 0:
            std = tw * s
            K = int(math.ceil(4 * std))
            u = np.arange(-K, K + 1)
            g = np.exp(-u ** 2 / (2 * std ** 2))
            g /= g.sum()
            size = 1 << int(math.ceil(math.log2(n + 2 * K)))
            full = np.fft.ifft(np.fft.fft(P[..., i, :], size, axis=-1) * np.fft.fft(g, size), axis=-1)[..., K:K + n]
            out[..., i, :] = full if np.iscomplexobj(P) else full.real
        else:
            out[..., i, :] = P[..., i, :]
    if ns > 1:
        pad = ns // 2
        padded = np.concatenate([out[..., :1, :].repeat(pad, axis=-2), out, out[..., -1:, :].repeat(ns - 1 - pad, axis=-2)], axis=-2)
        res = np.zeros_like(out)
        for j in range(ns):
            res += padded[..., j:j + out.shape[-2], :] / ns
        out = res
    return out


def coherence_from(Wx, Wy, freqs, tw: float, ns: int, w0: float = C.W0):
    """(R^2, Phase) aus fertigen Transformierten; R^2 = 0, wo ein Nenner 0 ist."""
    sxx = smooth(np.abs(Wx) ** 2, freqs, tw, ns, w0)
    syy = smooth(np.abs(Wy) ** 2, freqs, tw, ns, w0)
    sxy = smooth(np.conj(Wx) * Wy, freqs, tw, ns, w0)
    den = sxx * syy
    r2 = np.where(den > 0, np.abs(sxy) ** 2 / np.where(den > 0, den, 1.0), 0.0)
    return r2, np.angle(sxy)


def wavelet_coherence(x, y, freqs, tw: float, ns: int, w0: float = C.W0):
    """(R^2, Phase): geglättete Wavelet-Kohärenz und Phase des geglätteten Kreuzspektrums (Phase von y gegen x), jeweils (…, Frequenz, Zeit)."""
    return coherence_from(cwt(x, freqs, w0), cwt(y, freqs, w0), freqs, tw, ns, w0)
