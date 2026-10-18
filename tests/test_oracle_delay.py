"""Verzögerungsschätzer aus dem Kreuzspektrum: Gitter, Gewichte, Schätzung ganzer und gebrochener Verzögerungen, Vorzeichen, Ausrichten."""

import numpy as np
import pytest

import koh_constants as C
import koh_delay as D
import koh_signals as S
import koh_spectra as SP
import signal_generator as G


def test_delay_grid():
    g = D.delay_grid(2.0, 0.5)
    assert g.tolist() == [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0] and len(D.delay_grid()) == 3201 and D.delay_grid()[0] == -8.0 and D.delay_grid()[-1] == pytest.approx(8.0)
    assert D.delay_grid()[1] - D.delay_grid()[0] == pytest.approx(C.DELAY_STEP)


def test_weights_by_definition():
    s = np.array([3 + 4j, -2j, 0.5 + 0j])
    c = np.array([0.5, 0.8, 0.0])
    assert D.weights(s, c, "CC").tolist() == s.tolist()
    unit = s / np.abs(s)
    assert D.weights(s, c, "PHAT") == pytest.approx(unit)
    assert D.weights(s, c, "ML") == pytest.approx(unit * np.array([1.0, 4.0, 1e-6 / (1 - 1e-6)]))              # c / (1 - c), c auf [1e-6, 1 - 1e-6] begrenzt
    assert abs(D.weights(np.array([1.0 + 0j]), np.array([1.0]), "ML")[0]) == pytest.approx((1 - 1e-6) / 1e-6)
    assert D.weights(np.array([0j]), np.array([0.5]), "PHAT")[0] == 0
    with pytest.raises(ValueError):
        D.weights(s, c, "GCC")


@pytest.mark.parametrize("method", list(D.METHODS))
def test_noise_free_delays_are_recovered_with_the_right_sign(method):
    rng = G.SplitMix64(21)
    seg = 256
    for d in (0.0, 3.0, -2.5, 5.725):
        x = S.noise(rng, None, 4096)
        y = S.frac_delay(x, d)
        sxx, syy, sxy, k = SP.cross_spectra(x, y, seg, seg // 2)
        c = SP.coherence(sxx, syy, sxy)
        assert float(D.estimate(sxy, c, seg, method)) == pytest.approx(d, abs=0.02), (method, d)


def test_batched_estimates_and_the_search_range():
    rng = G.SplitMix64(22)
    x = S.noise(rng, 5, 2048)
    y = S.frac_delay(x, 4.0)
    sxx, syy, sxy, k = SP.cross_spectra(x, y, 128, 64)
    est = D.estimate(sxy, SP.coherence(sxx, syy, sxy), 128, "CC")
    assert est.shape == (5,) and est == pytest.approx(np.full(5, 4.0), abs=0.02)
    wide = D.estimate(sxy[0], SP.coherence(sxx, syy, sxy)[0], 128, "CC", half=20.0)
    assert float(wide) == pytest.approx(4.0, abs=0.02) and abs(float(D.estimate(sxy[0], SP.coherence(sxx, syy, sxy)[0], 128, "CC", half=2.0))) <= 2.0


def test_the_estimators_degrade_gracefully_with_noise_and_ml_beats_phat_at_high_coherence():
    rng = G.SplitMix64(23)
    seg, d, runs = 256, 3.4, 60
    x, e = S.noise(rng, runs, 4096), S.noise(rng, runs, 4096)
    y = S.delayed_pair(x, e, d, 0.9)
    sxx, syy, sxy, k = SP.cross_spectra(x, y, seg, seg // 2)
    c = SP.coherence(sxx, syy, sxy)
    rmse = {m: float(np.sqrt(np.mean((D.estimate(sxy, c, seg, m) - d) ** 2))) for m in D.METHODS}
    assert max(rmse.values()) < 0.02 and rmse["ML"] < rmse["PHAT"]


def test_align_removes_a_known_delay():
    rng = G.SplitMix64(24)
    x = S.noise(rng, None, 512)
    assert D.align(S.frac_delay(x, 3.0), 3.0) == pytest.approx(x, abs=1e-10) and D.align(x, 0.0) == pytest.approx(x, abs=1e-12)
    xb = np.fft.irfft(np.fft.rfft(x) * (np.arange(257) < 256), 512)                      # ohne den Nyquist-Anteil (der sich bei gebrochener Verzögerung nicht erhalten lässt)
    assert D.align(S.frac_delay(xb, 3.7), 3.7) == pytest.approx(xb, abs=1e-10)


def test_method_labels_cover_all_methods():
    assert set(D.METHOD_LABELS) == set(D.METHODS) == {"CC", "PHAT", "ML"}
