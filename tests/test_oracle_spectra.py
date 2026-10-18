"""Kreuzspektrum nach Welch gegen die Definition (Handrechnung, direkte Summe) und gegen scipy (nur als Orakel)."""

import numpy as np
import pytest
import scipy.signal as sps

import koh_signals as S
import koh_spectra as SP
import signal_generator as G


def test_mini_instance_by_hand(mini_xy):
    x, y = mini_xy
    sxx, syy, sxy, k = SP.cross_spectra(x, y, 2, 2, np.ones(2))
    assert k == 2
    assert (sxx * 2).tolist() == pytest.approx([29.0, 1.0]) and (syy * 2).tolist() == pytest.approx([36.5, 2.5]) and (sxy * 2).real.tolist() == pytest.approx([32.5, 0.5]) and (sxy.imag == 0).all()
    assert SP.coherence(sxx, syy, sxy).tolist() == pytest.approx([32.5 ** 2 / (29.0 * 36.5), 0.1])


def _direct(x, y, w, hop):
    seg = len(w)
    segs = [(x[i:i + seg] * w, y[i:i + seg] * w) for i in range(0, len(x) - seg + 1, hop)]
    K = len(segs)
    norm = float(np.sum(w ** 2))
    bins = seg // 2 + 1
    sxx, syy, sxy = np.zeros(bins), np.zeros(bins), np.zeros(bins, complex)
    for xs, ys in segs:
        for kk in range(bins):
            e = np.exp(-2j * np.pi * kk * np.arange(seg) / seg)
            X, Y = np.sum(xs * e), np.sum(ys * e)
            sxx[kk] += abs(X) ** 2 / K / norm
            syy[kk] += abs(Y) ** 2 / K / norm
            sxy[kk] += np.conj(X) * Y / K / norm
    return sxx, syy, sxy, K


def test_cross_spectra_equal_the_direct_double_sum():
    rng = G.SplitMix64(7)
    x, y = rng.normal(40), rng.normal(40)
    w = SP.hann(8)
    got = SP.cross_spectra(x, y, 8, 4, w)
    want = _direct(x, y, w, 4)
    assert got[3] == want[3] == 9 and got[0] == pytest.approx(want[0]) and got[1] == pytest.approx(want[1]) and got[2] == pytest.approx(want[2])


def test_against_scipy_csd_and_coherence():
    rng = G.SplitMix64(8)
    x = rng.normal(2000)
    y = 0.7 * np.convolve(x, [0.0, 1.0, 0.5])[:2000] + rng.normal(2000)
    seg, hop = 64, 32
    w = SP.hann(seg)
    sxx, syy, sxy, k = SP.cross_spectra(x, y, seg, hop, w)
    f, pxy = sps.csd(x, y, fs=1.0, window=w, nperseg=seg, noverlap=seg - hop, detrend=False, scaling="density")
    one_sided = np.where((np.arange(len(f)) == 0) | (np.arange(len(f)) == seg // 2), 1.0, 2.0)
    assert f == pytest.approx(SP.freqs(seg)) and pxy == pytest.approx(sxy * one_sided, rel=1e-9)
    _f, cxy = sps.coherence(x, y, fs=1.0, window=w, nperseg=seg, noverlap=seg - hop, detrend=False)
    assert cxy == pytest.approx(SP.coherence(sxx, syy, sxy), rel=1e-9) and k == (2000 - seg) // hop + 1


def test_one_segment_gives_coherence_one_even_for_independent_series():
    rng = G.SplitMix64(1)
    s = SP.cross_spectra(rng.normal(64), rng.normal(64), 64, 64)
    assert s[3] == 1 and SP.coherence(*s[:3]) == pytest.approx(np.ones(33), abs=1e-12)


def test_coherence_properties():
    rng = G.SplitMix64(2)
    x = rng.normal(512)
    y = np.convolve(x, [0.3, 0.8, 0.1])[:512] + rng.normal(512)
    sxx, syy, sxy, k = SP.cross_spectra(x, y, 32, 16)
    c = SP.coherence(sxx, syy, sxy)
    assert (c >= 0).all() and (c <= 1 + 1e-12).all()                                              # Cauchy-Schwarz
    s2 = SP.cross_spectra(3.0 * x, -2.0 * y, 32, 16)
    assert SP.coherence(*s2[:3]) == pytest.approx(c) and s2[2] == pytest.approx(-6.0 * sxy)       # Skalierung ändert die Kohärenz nicht, S_xy skaliert
    sw = SP.cross_spectra(y, x, 32, 16)
    assert sw[2] == pytest.approx(np.conj(sxy)) and SP.coherence(*sw[:3]) == pytest.approx(c)       # Vertauschen konjugiert S_xy
    assert c == pytest.approx(np.abs(SP.transfer_h1(sxx, sxy) / SP.transfer_h2(syy, sxy)))        # gamma^2 = H1 / H2


def test_phase_of_a_delay_has_the_right_sign_and_slope():
    rng = G.SplitMix64(3)
    x = rng.normal(8192)
    d, seg = 3, 128
    y = np.roll(x, d)
    sxx, syy, sxy, k = SP.cross_spectra(x, y, seg, seg // 2)
    f = SP.freqs(seg)
    err = np.angle(sxy * np.exp(2j * np.pi * f * d))
    assert np.abs(err[1:50]).max() < 0.06 and np.abs(np.angle(sxy[10]) - np.angle(np.exp(-2j * np.pi * f[10] * d))) < 0.06
    assert SP.coherence(sxx, syy, sxy)[1:50].min() > 0.98 and SP.phase(sxy) == pytest.approx(np.angle(sxy))


def test_transfer_functions_of_a_pure_gain():
    rng = G.SplitMix64(4)
    x = rng.normal(4096)
    y = 2.5 * x
    sxx, syy, sxy, k = SP.cross_spectra(x, y, 64, 64)
    assert SP.transfer_h1(sxx, sxy) == pytest.approx(2.5 * np.ones(33)) and SP.transfer_h2(syy, sxy) == pytest.approx(2.5 * np.ones(33))


def test_segment_bookkeeping():
    assert SP.n_segments(100, 20, 10) == 9 and SP.n_segments(20, 20, 10) == 1 and SP.hop_for(64, 0.0) == 64 and SP.hop_for(64, 0.5) == 32 and SP.hop_for(5, 0.99) == 1
    for seg, hop, k in ((64, 64, 7), (64, 32, 15), (128, 32, 3), (32, 1, 1)):
        assert SP.n_segments(SP.record_length(seg, hop, k), seg, hop) == k
    with pytest.raises(ValueError):
        SP.n_segments(10, 20, 5)
    with pytest.raises(ValueError):
        SP.n_segments(100, 20, 0)
    with pytest.raises(ValueError):
        SP.cross_spectra(np.zeros(100), np.zeros(100), 20, 10, np.ones(19))


def test_hann_and_frequency_axis():
    w = SP.hann(8)
    assert w == pytest.approx(0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(8) + 0.5) / 8)) and w == pytest.approx(w[::-1]) and w.max() < 1.0 and np.sum(w ** 2) == pytest.approx(3.0)
    assert SP.freqs(8).tolist() == [0.0, 0.125, 0.25, 0.375, 0.5]
    assert SP.coherence(np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([0.0, 0.5])).tolist() == [0.0, 0.25]            # Nenner null gibt 0


def test_fractional_delay_is_exact_for_band_limited_signals():
    n = 256
    t = np.arange(n)
    x = np.cos(2 * np.pi * 10 * t / n + 0.4)
    assert S.frac_delay(x, 0.5) == pytest.approx(np.cos(2 * np.pi * 10 * (t - 0.5) / n + 0.4), abs=1e-12)
    assert S.frac_delay(x, 3) == pytest.approx(np.roll(x, 3), abs=1e-12) and S.frac_delay(S.frac_delay(x, 2.3), -2.3) == pytest.approx(x, abs=1e-12)


def test_coherence_is_zero_where_the_denominator_is_zero_even_for_inconsistent_inputs():
    assert SP.coherence(np.array([0.0, 2.0]), np.array([1.0, 2.0]), np.array([0.5, 1.0])).tolist() == [0.0, 0.25]
    assert SP.coherence(np.array([1.0]), np.array([0.0]), np.array([0.5 + 0.5j])).tolist() == [0.0]
