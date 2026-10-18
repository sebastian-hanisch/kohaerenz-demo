"""Testsignale: weißes Rauschen, Verzögerung, FIR-System, Bandrauschen und das zeitweilig gekoppelte Paar gegen ihre Definition."""

import numpy as np
import pytest

import koh_constants as C
import koh_signals as S
import signal_generator as G


def test_noise_shapes_and_reproducibility():
    a = S.noise(G.SplitMix64(1), None, 50)
    b = S.noise(G.SplitMix64(1), 3, 50)
    assert a.shape == (50,) and b.shape == (3, 50) and a.tolist() == G.SplitMix64(1).normal(50).tolist() and b[0].tolist() == G.SplitMix64(1).normal(150)[:50].tolist()
    assert abs(S.noise(G.SplitMix64(2), None, 20000).std() - 1.0) < 0.03


def test_delayed_pair_sets_the_true_coherence_through_the_noise_power():
    rng = G.SplitMix64(3)
    x, e = S.noise(rng, None, 100000), S.noise(rng, None, 100000)
    y = S.delayed_pair(x, e, 0.0, 0.5)
    assert y.var() == pytest.approx(2.0, rel=0.03) and np.corrcoef(x, y)[0, 1] ** 2 == pytest.approx(0.5, abs=0.01)
    y9 = S.delayed_pair(x, e, 0.0, 0.9)
    assert np.corrcoef(x, y9)[0, 1] ** 2 == pytest.approx(0.9, abs=0.01)
    assert S.delayed_pair(x, e, 3.0, 0.0).tolist() == e.tolist() and S.delayed_pair(x, e, 3.0, 1.0) == pytest.approx(np.roll(x, 3), abs=1e-9) and S.delayed_pair(x, e, 3.0, 2.0) == pytest.approx(np.roll(x, 3), abs=1e-9)
    assert S.delayed_pair(x, e, 3.0, 0.5) == pytest.approx(np.roll(x, 3) + e, abs=1e-9)                              # g2 = 1/2: Rauschleistung 1


def test_fir_equals_the_causal_convolution_and_its_response():
    rng = G.SplitMix64(4)
    x = S.noise(rng, None, 64)
    y = S.fir(x)
    h = C.FIR
    assert y == pytest.approx(np.convolve(x, h)[:64]) and y[0] == 0.0 and y[1] == pytest.approx(0.6 * x[0])
    assert S.fir(x.reshape(2, 32)).shape == (2, 32) and S.fir(np.stack([x, 2 * x]))[1] == pytest.approx(2 * y)
    f = np.array([0.0, 0.25, 0.5])
    H = S.fir_response(f)
    assert H[0] == pytest.approx(sum(h)) and H[1] == pytest.approx(0.6 * (-1j) + 0.3 * (-1) + 0.1 * (1j)) and H[2] == pytest.approx(0.6 * -1 + 0.3 * 1 + 0.1 * -1)
    assert S.fir_response(0.1) == pytest.approx(S.fir_response(np.array([0.1])))
    assert S.fir_response(0.2, [1.0, 0.0, 0.5]) == pytest.approx(1 + 0.5 * np.exp(-2j * np.pi * 0.2 * 2))


def test_band_noise_lives_in_the_band_with_unit_standard_deviation():
    rng = G.SplitMix64(5)
    b = S.band_noise(rng, 4, 1024)
    assert b.shape == (4, 1024) and b.std(axis=-1) == pytest.approx(np.ones(4)) and S.band_noise(rng, None, 512).shape == (512,)
    spec = np.abs(np.fft.rfft(b, axis=-1)) ** 2
    f = np.fft.rfftfreq(1024)
    out = spec[:, np.abs(f - C.F0) > C.F_HALF + 1e-9].sum() / spec.sum()
    assert out < 1e-20


def test_rotation_and_the_coupling_mask():
    rng = G.SplitMix64(6)
    c = S.band_noise(rng, None, 1024)
    r0 = S.analytic_rotated(c, 0.0)
    assert r0 == pytest.approx(c, abs=1e-9)                                                 # Drehung 0 gibt die Reihe (bis auf Gleich- und Nyquist-Anteil, hier ohne)
    z = np.fft.ifft(np.fft.fft(c) * np.where(np.fft.fftfreq(1024) > 0, 2.0, 0.0))
    assert S.analytic_rotated(c, 0.7) == pytest.approx(np.real(z * np.exp(0.7j))) and S.analytic_rotated(c, np.pi) == pytest.approx(-c, abs=1e-9)
    g = S.coupling_mask(1024, 300, 700)
    assert g.shape == (1024,) and g[500] == pytest.approx(1.0) and g[50] == pytest.approx(0.0, abs=1e-9) and g[950] == pytest.approx(0.0, abs=1e-9) and g[300] == pytest.approx(0.5 * (np.tanh(0.0) - np.tanh(-400 / 10.0)), abs=1e-12) and g[700] == pytest.approx(0.5 * (np.tanh(40.0) - np.tanh(0.0)), abs=1e-12)
    assert S.coupling_mask(100, 20, 80, edge=2.0)[20] == pytest.approx(0.5 * (0.0 - np.tanh(-60 / 2.0)))


def test_coupled_pair_structure_and_equal_power_inside_and_outside():
    rng = G.SplitMix64(7)
    n = 1024
    x, y, g, c = S.coupled_pair(rng, 40, n, 300, 700, 0.5)
    assert x.shape == y.shape == c.shape == (40, n) and g.shape == (n,)
    inside, outside = slice(400, 600), np.r_[80:250, 750:950]
    px, py = x.var(), y.var()
    assert x[:, inside].var() == pytest.approx(1.25, rel=0.2) and y[:, inside].var() == pytest.approx(1.25, rel=0.2) and y[:, outside].var() == pytest.approx(1.25, rel=0.2) and px == pytest.approx(py, rel=0.2)
    # Kopplung nur im Intervall: Korrelation von x und y im Intervall deutlich größer als außerhalb
    def corr(sl):
        a, b = x[:, sl].ravel(), y[:, sl].ravel()
        return float(np.corrcoef(a, b)[0, 1])
    assert abs(corr(inside)) > 0.3 > abs(corr(outside)) or abs(corr(inside)) > 3 * abs(corr(outside))
    x1, y1, g1, c1 = S.coupled_pair(G.SplitMix64(7), None, n, 100, 900, 0.0, 0.0)
    assert x1 == pytest.approx(c1) and y1[300:700] == pytest.approx(S.analytic_rotated(c1, 0.0)[300:700], abs=1e-6)         # ohne Rauschen und Drehung: y = c im Intervall


def test_rotation_drops_the_constant_component_and_the_band_edges_are_inside():
    assert S.analytic_rotated(np.ones(16), 0.3) == pytest.approx(np.zeros(16), abs=1e-12)                 # der Gleichanteil gehört nicht zu den positiven Frequenzen
    rng = G.SplitMix64(8)
    b = S.band_noise(rng, None, 1000)                                                                    # f0 ± half = 0,04 und 0,06 liegen auf dem Gitter
    spec = np.abs(np.fft.rfft(b))
    assert spec[40] > 1e-3 and spec[60] > 1e-3 and spec[39] < 1e-9 and spec[61] < 1e-9 and spec[41:60].min() > 1e-3
