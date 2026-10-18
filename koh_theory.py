"""Theorie der Kohärenzschätzung aus K unabhängigen Segmenten (komplexe Gauß-Spektren): Schwelle, Erwartungswert, Phasenstreuung, Fensterkorrelation, äquivalente Segmentzahl.

Dichte der geschätzten Kohärenz c bei wahrer Kohärenz g2 und K Segmenten (Carter, Knapp und Nuttall 1973):
p(c) = (K - 1) (1 - g2)^K (1 - c)^(K - 2) 2F1(K, K; 1; g2 c); nach Eulers Umformung ist 2F1(K, K; 1; x) = (1 - x)^(1 - 2K) sum_m C(K - 1, m)^2 x^m (endliche Summe), das braucht keine Bibliothek.
"""

import math

import numpy as np

import koh_constants as C


def threshold(k: float, alpha: float = C.ALPHA) -> float:
    """Schwelle der geschätzten Kohärenz bei wahrer Kohärenz 0: P(c > s) = (1 - s)^(K - 1) = alpha, also s = 1 - alpha^(1/(K - 1)); bei K <= 1 gibt es keine (1)."""
    if k <= 1:
        return 1.0
    return 1.0 - alpha ** (1.0 / (k - 1.0))


def expected_first_order(g2: float, k: float) -> float:
    """Erwartungswert der geschätzten Kohärenz, erste Ordnung in 1/K: g2 + (1 - g2)^2 / K."""
    return g2 + (1.0 - g2) ** 2 / k


def _log_binomial_squares(k: int) -> np.ndarray:
    """log C(K - 1, m)^2 für m = 0 .. K - 1."""
    m = np.arange(k)
    lg = np.array([math.lgamma(i + 1) for i in range(k + 1)])
    return 2.0 * (lg[k - 1] - lg[m] - lg[k - 1 - m])


def grid_points(g2: float, base: int) -> int:
    """Zahl der Gitterpunkte der Integration: mindestens base, bei g2 -> 1 feiner (die Dichte ist dann bei c = 1 scharf gipflig), höchstens 200001."""
    return max(base, min(200001, int(60.0 / (1.0 - g2))))


def expected_coherence(g2: float, k: int, points: int = 4001) -> float:
    """Exakter Erwartungswert der geschätzten Kohärenz für K >= 2 unabhängige Segmente (numerische Integration der Dichte, Trapezregel); K = 1 gibt 1."""
    if k <= 1:
        return 1.0
    if g2 <= 0.0:
        return 1.0 / k                                              # Beta(1, K - 1)
    points = grid_points(g2, points)
    c = np.linspace(0.0, 1.0, points)[1:]                          # ohne 0 (dort ist c * p(c) null), mit 1 (bei K = 2 ist die Dichte dort endlich)
    logc = _log_binomial_squares(k)[None, :] + np.arange(k)[None, :] * np.log(g2 * c)[:, None]
    mx = logc.max(axis=1)
    log_poly = mx + np.log(np.exp(logc - mx[:, None]).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        tail = (k - 2) * np.log1p(-c) if k > 2 else np.zeros_like(c)     # (1 - c)^(K - 2); bei c = 1 ist das für K > 2 null
    log_pdf = math.log(k - 1) + k * math.log1p(-g2) + tail + (1 - 2 * k) * np.log1p(-g2 * c) + log_poly
    pdf = np.exp(log_pdf)
    cc = np.concatenate([[0.0], c])
    pp = np.concatenate([[0.0], pdf])
    return float(np.sum(0.5 * (cc[1:] * pp[1:] + cc[:-1] * pp[:-1]) * np.diff(cc)))


def phase_std(g2: float, k: float) -> float:
    """Streuung der Phase von S_xy (rad) für große K: sqrt((1 - g2) / (2 K g2)); bei g2 = 0 unendlich."""
    if g2 <= 0.0:
        return math.inf
    return math.sqrt((1.0 - g2) / (2.0 * k * g2))


def window_correlation(w, d: float) -> float:
    """rho(d) = sum_m w[m] w[m + d] / sum w^2 für ganzzahliges d (gebrochenes d: lineare Interpolation zwischen den Nachbarn); 0 ab d >= Fensterlänge."""
    w = np.asarray(w, float)
    L = len(w)
    d = abs(float(d))

    def rho(i: int) -> float:
        return float(np.sum(w[: L - i] * w[i:]) / np.sum(w ** 2)) if i < L else 0.0
    lo = int(math.floor(d))
    frac = d - lo
    return rho(lo) if frac == 0.0 else (1.0 - frac) * rho(lo) + frac * rho(lo + 1)


def variance_factor(w, hop: int, k: int) -> float:
    """Streufaktor des Segmentmittels bei Überlappung: 1 + 2 sum_{j=1}^{K-1} (1 - j/K) rho(j hop)^2 (Periodogramme zweier Segmente korrelieren mit rho^2)."""
    total = 1.0
    for j in range(1, k):
        total += 2.0 * (1.0 - j / k) * window_correlation(w, j * hop) ** 2
    return total


def k_effective(w, hop: int, k: int) -> float:
    """Äquivalente Zahl unabhängiger Segmente K / Streufaktor."""
    return k / variance_factor(w, hop, k)


def h1_factor(r: float) -> float:
    """Erwartung von |H1| / |H| bei Eingangsrauschen mit Leistung r gegen die Eingangsleistung: 1 / (1 + r)."""
    return 1.0 / (1.0 + r)


def h2_factor(h_abs, out_noise: float) -> np.ndarray:
    """Erwartung von |H2| / |H| bei Ausgangsrauschen (Leistung out_noise, Eingangsleistung 1): 1 + out_noise / |H|^2."""
    return 1.0 + out_noise / np.asarray(h_abs, float) ** 2


def true_coherence_fir(h_abs, out_noise: float, in_noise: float) -> np.ndarray:
    """Wahre Kohärenz zwischen dem verrauschten Eingang und dem Ausgang: |H|^2 / ((1 + r) (|H|^2 + N))."""
    a = np.asarray(h_abs, float) ** 2
    return a / ((1.0 + in_noise) * (a + out_noise))
