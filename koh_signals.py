"""Testsignale: weißes Rauschen, Verzögerung (ganz und gebrochen), FIR-System, Bandrauschen und die zeitweilig gekoppelte Reihe.

Alle Funktionen arbeiten auf der letzten Achse (Zeit); `runs` Läufe werden aus einem Zug von runs * n Normalwerten gebildet. Abtastrate 1, Frequenzen in Zyklen je Abtastwert.
"""

import numpy as np

import koh_constants as C


def noise(rng, runs: int | None, n: int) -> np.ndarray:
    """Weißes Gauß-Rauschen (Einheitsvarianz): Form (n,) bei runs = None, sonst (runs, n)."""
    if runs is None:
        return rng.normal(n)
    return rng.normal(runs * n).reshape(runs, n)


def frac_delay(x, d: float) -> np.ndarray:
    """Zirkuläre Verzögerung um d Abtastwerte (auch gebrochen) per FFT: y[t] = x[t - d]. Bei gebrochenem d geht der (reelle) Nyquist-Anteil teilweise verloren; ganzzahliges d ist exakt."""
    x = np.asarray(x, float)
    f = np.fft.fftfreq(x.shape[-1])
    return np.real(np.fft.ifft(np.fft.fft(x, axis=-1) * np.exp(-2j * np.pi * f * d), axis=-1))


def delayed_pair(x, e, d: float, g2: float):
    """y = x um d verzögert plus Rauschen, dessen Leistung die wahre Kohärenz g2 = 1 / (1 + Rauschleistung) einstellt (Eingangsleistung 1); g2 = 0: y = e (unabhängig)."""
    if g2 <= 0.0:
        return np.asarray(e, float)
    if g2 >= 1.0:
        return frac_delay(x, d)
    return frac_delay(x, d) + np.sqrt((1.0 - g2) / g2) * np.asarray(e, float)


def fir(x, h=C.FIR) -> np.ndarray:
    """Lineare Faltung mit h, auf die Länge von x gekürzt (kausal)."""
    x = np.asarray(x, float)
    h = np.asarray(h, float)
    flat = x.reshape(-1, x.shape[-1])
    out = np.stack([np.convolve(r, h)[: x.shape[-1]] for r in flat])
    return out.reshape(x.shape)


def fir_response(f, h=C.FIR) -> np.ndarray:
    """Frequenzgang H(f) = sum_k h_k exp(-i 2 pi f k)."""
    h = np.asarray(h, float)
    return np.sum(h[:, None] * np.exp(-2j * np.pi * np.outer(np.arange(len(h)), np.atleast_1d(f))), axis=0)


def band_noise(rng, runs: int | None, n: int, f0: float = C.F0, half: float = C.F_HALF) -> np.ndarray:
    """Weißes Rauschen auf das Band f0 ± half beschränkt (FFT-Maske), je Lauf auf Standardabweichung 1 normiert."""
    z = np.fft.rfft(noise(rng, runs, n), axis=-1)
    f = np.fft.rfftfreq(n)
    z = z * (np.abs(f - f0) <= half + 1e-9)                                      # Toleranz: die Bandkanten liegen auf dem Gitter und sollen nicht an Rundung scheitern
    b = np.fft.irfft(z, n, axis=-1)
    return b / b.std(axis=-1, keepdims=True)


def analytic_rotated(c, phase: float) -> np.ndarray:
    """Realteil des um `phase` gedrehten analytischen Signals: die Reihe c mit um phase verschobener Phase (nur positive Frequenzen, doppelt gewichtet)."""
    c = np.asarray(c, float)
    n = c.shape[-1]
    w = np.where(np.fft.fftfreq(n) > 0, 2.0, 0.0)
    return np.real(np.fft.ifft(np.fft.fft(c, axis=-1) * w, axis=-1) * np.exp(1j * phase))


def coupling_mask(n: int, start: float, end: float, edge: float = 10.0) -> np.ndarray:
    """g(t) = 1 im Intervall [start, end), an den Rändern mit tanh(./edge) geglättet."""
    t = np.arange(n)
    return 0.5 * (np.tanh((t - start) / edge) - np.tanh((t - end) / edge))


def coupled_pair(rng, runs: int | None, n: int, start: float, end: float, amp: float, phase: float = C.PHASE0):
    """Zeitweilig gekoppeltes Paar: x = c + amp b1, y = g c(phase) + (1 - g) b3 + amp b2 (c, b1, b2, b3 unabhängiges Bandrauschen, Leistung 1);
    in der Kopplung ist die wahre Kohärenz im Band 1 / (1 + amp^2)^2. Liefert (x, y, g, c)."""
    c, b1, b2, b3 = (band_noise(rng, runs, n) for _ in range(4))
    g = coupling_mask(n, start, end)
    return c + amp * b1, g * analytic_rotated(c, phase) + (1.0 - g) * b3 + amp * b2, g, c
