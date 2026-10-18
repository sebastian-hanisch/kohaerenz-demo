"""Theorie der Kohärenzschätzung: Schwelle, exakter Erwartungswert, Phasenstreuung, Fensterkorrelation, äquivalente Segmentzahl (scipy nur als Orakel)."""

import math

import numpy as np
import pytest
import scipy.special as sp
import scipy.stats as st

import koh_signals as S
import koh_spectra as SP
import koh_theory as T
import signal_generator as G


def test_threshold_by_hand_and_against_the_beta_distribution():
    assert T.threshold(2) == pytest.approx(0.95) and T.threshold(8) == pytest.approx(1 - 0.05 ** (1 / 7)) and T.threshold(1) == 1.0 and T.threshold(0.5) == 1.0
    for k in (2, 3, 8, 32):
        assert st.beta(1, k - 1).sf(T.threshold(k)) == pytest.approx(0.05)
    assert st.beta(1, 7).sf(T.threshold(8, 0.01)) == pytest.approx(0.01)


def _k2_closed_form(g):
    """E[c] für K = 2 aus der Stammfunktion: (2 (1-g)^2 + 1 - 3 (1-g) - (1-g)^2 ln(1-g)) / g^2."""
    u = 1 - g
    return (2 * u ** 2 + 1 - 3 * u - u ** 2 * math.log(u)) / g ** 2


def test_expected_coherence_closed_forms():
    assert T.expected_coherence(0.0, 8) == pytest.approx(0.125) and T.expected_coherence(0.7, 1) == 1.0 and T.expected_coherence(0.0, 2) == 0.5
    assert T.expected_coherence(0.5, 2) == pytest.approx(math.log(2.0), abs=2e-4)                      # bei g^2 = 1/2 ist die K = 2 Formel genau ln 2
    for g in (0.25, 0.5, 0.75, 0.9):
        assert T.expected_coherence(g, 2) == pytest.approx(_k2_closed_form(g), abs=3e-4)


def test_expected_coherence_against_scipy_integration_of_the_density():
    for k, g in ((4, 0.5), (16, 0.25), (16, 0.9), (32, 0.5)):
        c = np.linspace(0, 1, 20001)[1:-1]
        pdf = (k - 1) * (1 - g) ** k * (1 - c) ** (k - 2) * sp.hyp2f1(k, k, 1, g * c)
        assert float(np.sum(c * pdf) * (c[1] - c[0])) == pytest.approx(T.expected_coherence(g, k), abs=5e-4), (k, g)
    assert T.expected_coherence(0.5, 8, 4001) == pytest.approx(T.expected_coherence(0.5, 8, 1001), abs=1e-3)


def test_expected_coherence_against_simulation_and_first_order():
    rng = G.SplitMix64(11)
    seg, K, g2 = 32, 4, 0.5
    x, e = S.noise(rng, 300, K * seg), S.noise(rng, 300, K * seg)
    y = S.delayed_pair(x, e, 0.0, g2)
    c = SP.coherence(*SP.cross_spectra(x, y, seg, seg)[:3])[:, 2:15]
    assert c.mean() == pytest.approx(T.expected_coherence(g2, K), abs=0.012)
    assert T.expected_first_order(0.25, 32) == pytest.approx(0.25 + 0.75 ** 2 / 32) and abs(T.expected_first_order(0.9, 64) - T.expected_coherence(0.9, 64)) < 1e-4
    assert T.expected_coherence(0.5, 4) > 0.5 and T.expected_coherence(0.2, 8) < T.expected_coherence(0.6, 8) < T.expected_coherence(0.9, 8)


def test_false_alarm_rate_at_the_threshold_by_simulation():
    rng = G.SplitMix64(12)
    seg, K = 32, 8
    x, e = S.noise(rng, 400, K * seg), S.noise(rng, 400, K * seg)
    c = SP.coherence(*SP.cross_spectra(x, e, seg, seg)[:3])[:, 2:15]
    assert np.mean(c > T.threshold(K)) == pytest.approx(0.05, abs=0.012)


def test_phase_std_formula_and_by_simulation():
    assert T.phase_std(0.5, 32) == pytest.approx(math.sqrt(0.5 / (2 * 32 * 0.5))) and T.phase_std(0.0, 8) == math.inf and T.phase_std(1.0, 8) == 0.0
    rng = G.SplitMix64(13)
    seg, K, g2 = 32, 64, 0.5
    x, e = S.noise(rng, 300, K * seg), S.noise(rng, 300, K * seg)
    y = S.delayed_pair(x, e, 0.0, g2)
    ph = np.angle(SP.cross_spectra(x, y, seg, seg)[2])[:, 2:15]
    assert ph.std() == pytest.approx(T.phase_std(g2, K), rel=0.06)


def test_window_correlation_by_definition():
    w = SP.hann(16)
    assert T.window_correlation(w, 0) == pytest.approx(1.0) and T.window_correlation(w, 16) == 0.0 and T.window_correlation(w, 20) == 0.0
    assert T.window_correlation(w, 3) == pytest.approx(float(np.sum(w[:13] * w[3:]) / np.sum(w ** 2))) and T.window_correlation(w, -3) == T.window_correlation(w, 3)
    r3, r4 = T.window_correlation(w, 3), T.window_correlation(w, 4)
    assert T.window_correlation(w, 3.25) == pytest.approx(0.75 * r3 + 0.25 * r4) and r3 > r4 > 0
    assert T.window_correlation(np.ones(8), 2) == pytest.approx(6.0 / 8.0)


def test_variance_factor_and_effective_segment_count():
    w = SP.hann(64)
    assert T.variance_factor(w, 32, 1) == 1.0 and T.variance_factor(w, 64, 10) == 1.0 and T.k_effective(w, 64, 10) == 10.0
    vf = T.variance_factor(w, 32, 127)
    assert vf == pytest.approx(1 + 2 * (1 - 1 / 127) * T.window_correlation(w, 32) ** 2 + 2 * (1 - 2 / 127) * T.window_correlation(w, 64) ** 2)
    assert T.k_effective(w, 32, 127) == pytest.approx(127 / vf) and T.k_effective(w, 32, 127) == pytest.approx(120.3657, abs=1e-3) and T.k_effective(w, 16, 253) == pytest.approx(131.7185, abs=1e-3)
    assert T.k_effective(w, 8, 50) < T.k_effective(w, 32, 50) < 50


def test_transfer_factors_and_the_fir_coherence():
    assert T.h1_factor(0.0) == 1.0 and T.h1_factor(1.0) == 0.5 and T.h1_factor(0.25) == 0.8
    assert T.h2_factor([1.0, 0.5], 0.5).tolist() == pytest.approx([1.5, 3.0])
    assert T.true_coherence_fir([1.0], 1.0, 0.0).tolist() == pytest.approx([0.5]) and T.true_coherence_fir([1.0], 1.0, 1.0).tolist() == pytest.approx([0.25]) and T.true_coherence_fir([2.0], 0.0, 0.0).tolist() == pytest.approx([1.0])


def test_expected_coherence_is_finite_at_extreme_values():
    for g, k in ((0.99, 128), (0.99, 2), (0.999, 64), (0.001, 128), (0.5, 3)):
        e = T.expected_coherence(g, k)
        assert math.isfinite(e) and g <= e <= 1.0
    assert T.expected_coherence(0.99, 128) == pytest.approx(0.99 + 0.01 ** 2 / 128, abs=2e-4)
    assert 0.5 + 0.25 / 3 < T.expected_coherence(0.5, 3) < 0.65                                   # die erste Ordnung unterschätzt bei K = 3


def test_variance_factor_for_two_and_three_segments_includes_the_last_lag():
    w = SP.hann(64)
    r = T.window_correlation(w, 32) ** 2
    assert T.variance_factor(w, 32, 2) == pytest.approx(1 + 2 * 0.5 * r) and T.variance_factor(w, 16, 3) == pytest.approx(
        1 + 2 * (2 / 3) * T.window_correlation(w, 16) ** 2 + 2 * (1 / 3) * T.window_correlation(w, 32) ** 2)
    assert T.k_effective(w, 32, 2) == pytest.approx(2 / (1 + r))


def test_integration_grid_gets_finer_towards_full_coherence_and_is_capped():
    assert T.grid_points(0.0, 1001) == 1001 and T.grid_points(0.9, 1001) == 1001 and T.grid_points(0.9, 100) == 600 and T.grid_points(0.5, 100) == 120
    assert T.grid_points(0.99, 1001) == 5999 and T.grid_points(0.999, 1001) == 59999 and T.grid_points(0.99999, 1001) == 200001 and T.grid_points(0.9999999, 5) == 200001
