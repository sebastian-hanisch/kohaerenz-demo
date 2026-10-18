"""Wavelet-Kohärenz: Morlet-Transformation gegen die direkte Summe, Glättung gegen direkte Faltung, Kohärenz ohne und mit Glättung, Phase, Skalierung."""

import math

import numpy as np
import pytest

import koh_constants as C
import koh_signals as S
import koh_wavelet as W
import signal_generator as G


def test_frequency_grid_and_scale():
    fr = W.freq_grid()
    assert len(fr) == 41 and fr[0] == pytest.approx(0.2 * 2 ** (-40 / 12))
    assert (np.diff(fr) > 0).all() and fr[-1] == pytest.approx(0.2) and fr[1] / fr[0] == pytest.approx(2 ** (1 / 12))
    assert W.scale_of(0.05) == pytest.approx(6.0 / (2 * math.pi * 0.05)) and W.scale_of(np.array([0.1, 0.2])).tolist() == pytest.approx([6 / (0.2 * math.pi), 6 / (0.4 * math.pi)])
    with pytest.raises(ValueError):
        W.freq_grid(0.3, 0.2)
    with pytest.raises(ValueError):
        W.freq_grid(0.1, 0.2, 0)


def test_cwt_equals_the_direct_sum():
    rng = G.SplitMix64(31)
    x = rng.normal(160)
    fr = np.array([0.04, 0.08, 0.15])
    out = W.cwt(x, fr)
    assert out.shape == (3, 160)
    for i, f in enumerate(fr):
        assert out[i] == pytest.approx(W.cwt_direct(x, f), abs=2e-5), f
    assert W.cwt(rng.normal(2 * 64).reshape(2, 64), fr).shape == (2, 3, 64)


def test_cwt_of_a_pure_oscillation_peaks_at_its_frequency():
    n = 512
    x = np.cos(2 * np.pi * 0.05 * np.arange(n))
    fr = W.freq_grid()
    p = np.abs(W.cwt(x, fr)[:, 200:300]).mean(axis=1)
    assert fr[int(np.argmax(p))] == pytest.approx(0.05, rel=0.07)


def test_smoothing_equals_direct_convolution_in_time_and_scale():
    rng = G.SplitMix64(32)
    fr = np.array([0.05, 0.06, 0.07, 0.08, 0.09])
    P = rng.normal(5 * 120).reshape(5, 120)
    tw = 0.5
    got = W.smooth(P, fr, tw, 1)
    for i, f in enumerate(fr):
        std = tw * float(W.scale_of(f))
        K = int(math.ceil(4 * std))
        u = np.arange(-K, K + 1)
        g = np.exp(-u ** 2 / (2 * std ** 2))
        g /= g.sum()
        assert got[i] == pytest.approx(np.convolve(P[i], g, mode="same"), abs=1e-12)
    box = W.smooth(P, fr, 0.0, 3)                                             # Kasten über 3 Frequenzen, Randzeilen wiederholt
    padded = np.vstack([P[:1], P, P[-1:]])
    assert box == pytest.approx((padded[:-2] + padded[1:-1] + padded[2:]) / 3, abs=1e-12)
    assert W.smooth(P, fr, 0.0, 1) == pytest.approx(P) and W.smooth(1j * P, fr, 0.5, 3) == pytest.approx(1j * W.smooth(P, fr, 0.5, 3))
    both = W.smooth(P, fr, 0.5, 3)
    assert both == pytest.approx(W.smooth(W.smooth(P, fr, 0.5, 1), fr, 0.0, 3))


def test_smoothing_keeps_constants_away_from_the_edges():
    fr = W.freq_grid()[10:14]
    P = np.ones((4, 400))
    out = W.smooth(P, fr, 1.0, 3)
    assert out[:, 150:250] == pytest.approx(np.ones((4, 100)), abs=1e-9)


def test_coherence_without_smoothing_is_exactly_one():
    rng = G.SplitMix64(33)
    x, y = rng.normal(256), rng.normal(256)
    fr = W.freq_grid()
    r2, ph = W.wavelet_coherence(x, y, fr, 0.0, 1)
    assert r2.shape == ph.shape == (41, 256) and r2 == pytest.approx(np.ones((41, 256)), abs=1e-9)


def test_coherence_of_a_series_with_itself_or_a_scaled_copy_is_one_with_smoothing():
    rng = G.SplitMix64(34)
    x = rng.normal(256)
    fr = W.freq_grid()
    r2, ph = W.wavelet_coherence(x, -3.0 * x, fr, 1.0, 7)
    assert r2[:, 40:200] == pytest.approx(np.ones((41, 160)), abs=1e-9) and np.abs(np.abs(ph[:, 40:200]) - math.pi).max() < 1e-6           # y = -3 x: Phase pi
    y2 = rng.normal(256)
    assert W.wavelet_coherence(5.0 * x, 0.2 * y2, fr, 1.0, 7)[0] == pytest.approx(W.wavelet_coherence(x, y2, fr, 1.0, 7)[0], abs=1e-9)          # Skalierung ändert die Kohärenz nicht


def test_smoothed_coherence_of_independent_noise_is_biased_upwards_and_in_range():
    rng = G.SplitMix64(35)
    a, b = S.noise(rng, 30, 512), S.noise(rng, 30, 512)
    fr = W.freq_grid()
    r_none = W.wavelet_coherence(a, b, fr, 0.0, 1)[0]
    r_t1 = W.wavelet_coherence(a, b, fr, 1.0, 1)[0]
    r_t2 = W.wavelet_coherence(a, b, fr, 2.0, 7)[0]
    assert (r_t1 >= 0).all() and (r_t1 <= 1 + 1e-9).all()
    assert r_none[:, 10, 100:400].mean() == pytest.approx(1.0) and 0.35 < r_t1[:, 20, 100:400].mean() < 0.7 and r_t2[:, 20, 100:400].mean() < r_t1[:, 20, 100:400].mean()


def test_phase_of_a_rotated_copy_and_the_interval_structure():
    rng = G.SplitMix64(36)
    n = 1024
    x, y, g, c = S.coupled_pair(rng, None, n, 300, 700, 0.1)
    fr = W.freq_grid()
    i0 = int(np.argmin(np.abs(fr - C.F0)))
    r2, ph = W.wavelet_coherence(x, y, fr, 1.0, 7)
    inside = r2[i0, 400:600].mean()
    outside = np.concatenate([r2[i0, 80:250], r2[i0, 750:950]]).mean()
    assert inside > 0.9 and outside < 0.7 and abs(np.angle(np.exp(1j * ph[i0, 400:600]).mean()) - C.PHASE0) < 0.15


def test_coherence_from_equals_wavelet_coherence():
    rng = G.SplitMix64(37)
    x, y = rng.normal(128), rng.normal(128)
    fr = W.freq_grid()[5:15]
    a = W.wavelet_coherence(x, y, fr, 0.5, 3)
    b = W.coherence_from(W.cwt(x, fr), W.cwt(y, fr), fr, 0.5, 3)
    assert a[0] == pytest.approx(b[0]) and a[1] == pytest.approx(b[1])
    zero = W.coherence_from(np.zeros((2, 5, 64), complex), np.zeros((2, 5, 64), complex), W.freq_grid()[:5], 0.0, 1)
    assert (zero[0] == 0).all()


def test_frequency_grid_validation_boundaries():
    for bad in ((0.0, 0.2, 12), (0.2, 0.2, 12), (0.3, 0.2, 12), (0.02, 0.2, 0)):
        with pytest.raises(ValueError):
            W.freq_grid(*bad)
    g = W.freq_grid(0.1, 0.2, 1)
    assert g.tolist() == pytest.approx([0.1, 0.2]) and len(W.freq_grid(0.1, 0.2, 2)) == 3


def test_scale_smoothing_with_two_frequencies_averages_the_neighbour_below():
    P = np.arange(4 * 6, dtype=float).reshape(4, 6) ** 1.5
    fr = np.array([0.05, 0.06, 0.07, 0.08])
    out = W.smooth(P, fr, 0.0, 2)
    padded = np.vstack([P[:1], P])                                                    # ns = 2: eine Zeile davor, keine danach
    assert out == pytest.approx((padded[:-1] + padded[1:]) / 2)
