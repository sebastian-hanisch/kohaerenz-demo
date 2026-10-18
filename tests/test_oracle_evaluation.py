"""Auswertung: Einstellungen, Messung, Analyse (feste Werte der Presets und einiger Varianten) und die Experimentfunktionen (kleine Instanzen)."""

import pickle

import numpy as np
import pytest

import koh_constants as C
import koh_evaluation as ev
import koh_signals as S
import koh_spectra as SP
import signal_generator as G
from rowsapprox import rows_approx


def _mk(name, **over):
    p = C.PRESETS[name]
    base = dict(scenario=p["scenario"], seg=p["seg"], k=p["k"], overlap=p["overlap"], g2=p["g2"], delay=p["delay"], out=p["out"], inn=p["inn"], start=p["start"], end=p["end"], amp=p["amp"], tw=p["tw"],
                ns=p["ns"], seed=p["seed"])
    return ev.Settings(**{**base, **over})


def _vals(a):
    f = lambda v: -1.0 if v is None else float(v)          # noqa: E731
    if a.settings.welch:
        d = a.delays or {}
        return (a.mean_coh, a.mean_true, a.mean_expected, a.thr, a.share_above, f(a.phase_rmse), a.k_eff, a.k_segments, f(a.rel_h1), f(a.rel_h2), f(d.get("CC")), f(d.get("PHAT")), f(d.get("ML")))
    return (a.w_in, a.w_out, a.w_phase_in, a.welch_coh, a.welch_phase, a.w_true)


P = list(C.PRESETS)


def test_settings_derived_values_and_the_decimal_comma():
    s = ev.Settings()
    assert (s.scenario, s.seg, s.k, s.overlap, s.seed) == ("Verzögerung", 64, 32, "0 %", C.DEFAULT_SEED) and s.welch is True and s.hop == 64 and s.n == 64 + 31 * 64
    assert ev.Settings(overlap="50 %").hop == 32 and ev.Settings(overlap="50 %").n == 64 + 31 * 32 and ev.Settings(seg=128, k=1).n == 128
    w = ev.Settings(scenario="Zeitweilige Kopplung", seg=256, k=99)
    assert w.welch is False and w.n == 1024 and ev.Settings(scenario="FIR-System").welch is True
    assert ev.de(0.12345, 3) == "0,123" and ev.de(2.0) == "2,0" and ev.de(-0.5, 2) == "−0,50" and ev.de(float("nan")) == "–" and ev.band_slice(64) == slice(3, 30) and ev.band_slice(32, 2) == slice(2, 15)


def test_measurement_matches_the_scenario_definition():
    s = ev.Settings(scenario="Verzögerung", seg=32, k=8, g2=0.5, delay=2.5, seed=5)
    m = ev.build_measurement(s)
    rng = G.SplitMix64(5)
    x, e, _v = S.noise(rng, None, s.n), S.noise(rng, None, s.n), S.noise(rng, None, s.n)
    assert m["x"].tolist() == x.tolist() and m["y"] == pytest.approx(S.frac_delay(x, 2.5) + e)
    f = ev.build_measurement(ev.Settings(scenario="FIR-System", seg=32, k=8, out=0.5, inn=0.25, seed=6))
    rng = G.SplitMix64(6)
    x, e, v = S.noise(rng, None, 256), S.noise(rng, None, 256), S.noise(rng, None, 256)
    assert f["x"] == pytest.approx(x + 0.5 * v) and f["y"] == pytest.approx(S.fir(x) + np.sqrt(0.5) * e)
    w = ev.build_measurement(ev.Settings(scenario="Zeitweilige Kopplung", seed=7))
    assert w["x"].shape == w["y"].shape == w["g"].shape == (1024,) and ev.build_measurement(ev.Settings(seed=5))["y"].tolist() != ev.build_measurement(ev.Settings(seed=6))["y"].tolist()


def test_analyse_the_five_presets_golden():
    got = [_vals(ev.analyse(_mk(n))) for n in P]
    assert got[0] == pytest.approx((1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 1, -1.0, -1.0, 0.175, -0.075, -0.075), abs=1e-6)
    assert got[1] == pytest.approx((0.689222, 0.5, 0.693148, 0.95, 0.074074, 0.575327, 2.0, 2, -1.0, -1.0, 0.01, -0.02, 0.02), abs=1e-6)
    assert got[2] == pytest.approx((0.775463, 0.95, 0.773941, 0.046438, 1.0, 0.04764, 64.0, 64, -1.0, -1.0, 16.0, 16.0, 16.0), abs=1e-6)
    assert got[3] == pytest.approx((0.375549, 0.379609, 0.385743, 0.046438, 1.0, 0.121631, 64.0, 64, -0.514486, 0.347288, -1.0, -1.0, -1.0), abs=1e-6)
    assert got[4] == pytest.approx((0.753687, 0.492692, 0.997786, 0.253645, 0.98916, 0.64), abs=1e-6)


def test_analyse_variants_golden():
    assert _vals(ev.analyse(_mk(P[2], overlap="50 %", k=31, delay=2.5))) == pytest.approx((0.946791, 0.95, 0.945165, 0.100049, 1.0, 0.030614, 29.418367, 31, -1.0, -1.0, 2.5, 2.49, 2.505), abs=1e-6)
    assert _vals(ev.analyse(_mk(P[3], seg=32, k=10, out=0.5, inn=0.25, seed=5))) == pytest.approx((0.435518, 0.333161, 0.38159, 0.283129, 0.636364, 0.275321, 10.0, 10, -0.06049, 1.623651, -1.0, -1.0, -1.0), abs=1e-6)
    assert _vals(ev.analyse(_mk(P[0], k=8, g2=0.0, seed=9))) == pytest.approx((0.124472, 0.0, 0.125, 0.348164, 0.037037, 0.925565, 8.0, 8, -1.0, -1.0, -7.38, -7.5, 0.96), abs=1e-6)
    assert _vals(ev.analyse(_mk(P[4], tw=0.0, ns=1))) == pytest.approx((1.0, 1.0, 1.040749, 0.253645, 0.98916, 0.64), abs=1e-6)
    assert _vals(ev.analyse(_mk(P[4], amp=0.1, start=200, end=800))) == pytest.approx((0.98753, 0.270991, 1.010656, 0.429458, 1.054591, 0.980296), abs=1e-6)


def test_structure_of_the_result_per_scenario():
    d = ev.analyse(_mk(P[2]))
    assert d.freqs.shape == d.coh.shape == d.phase.shape == d.true_coh.shape == d.true_phase.shape == d.expected.shape == (65,) and d.h1 is None and d.wcoh is None and set(d.delays) == {"CC", "PHAT", "ML"}
    assert d.true_coh == pytest.approx(np.full(65, 0.95)) and d.true_phase[10] == pytest.approx(np.angle(np.exp(-2j * np.pi * d.freqs[10] * 16.0)))
    fir = ev.analyse(_mk(P[3]))
    assert fir.h_true.shape == fir.h1.shape == fir.h2.shape == (33,) and fir.delays is None and fir.true_phase == pytest.approx(np.angle(fir.h_true)) and fir.true_coh.shape == (33,)
    w = ev.analyse(_mk(P[4]))
    assert w.wcoh.shape == w.wphase.shape == (41, 1024) and w.mask.shape == (1024,) and w.freqs is None and w.coh is None and abs(w.wfreqs[w.i0] - 0.05) < 1e-12 and w.k_segments == 0


def test_share_above_and_phase_error_use_the_threshold_and_the_band():
    a = ev.analyse(_mk(P[2]))
    sl = ev.band_slice(128)
    assert a.share_above == pytest.approx(float(np.mean(a.coh[sl] > a.thr))) and a.mean_coh == pytest.approx(float(a.coh[sl].mean()))
    one = ev.analyse(_mk(P[0]))
    assert one.share_above == 0.0 and one.phase_rmse is None and one.thr == 1.0 and one.k_segments == 1


def test_wavelet_windows():
    assert ev.wavelet_windows(1024, 300, 700) == (slice(400, 600), [slice(80, 250), slice(750, 944)])
    assert ev.wavelet_windows(1024, 300, 360) == (slice(315, 345), [slice(80, 250), slice(410, 944)])
    assert ev.wavelet_windows(1024, 60, 400) == (slice(145, 315), [slice(450, 944)])
    assert ev.wavelet_windows(1024, 60, 960)[1] == []


def test_analysis_is_picklable():
    a = ev.analyse(ev.Settings(seg=32, k=4))
    b = pickle.loads(pickle.dumps(a))
    assert b.mean_coh == a.mean_coh and b.settings == a.settings and (b.coh == a.coh).all()


def test_sweep_bias_small_instance():
    r = ev.sweep_bias(seg=32, runs=20, ks=(2, 4), g2s=(0.0, 0.5), thr_ks=(2, 4), thr_runs=50, hops=(32, 16))
    assert [(x["k"], x["g2"]) for x in r["rows"]] == [(2, 0.0), (2, 0.5), (4, 0.0), (4, 0.5)]
    assert [round(x["sim"], 5) for x in r["rows"]] == pytest.approx([0.47189, 0.72027, 0.24415, 0.60408], abs=1e-5)
    assert [round(x["first"], 5) for x in r["rows"]] == pytest.approx([0.5, 0.625, 0.25, 0.5625]) and [round(x["exact"], 5) for x in r["rows"]] == pytest.approx([0.5, 0.69315, 0.25, 0.57944], abs=1e-5)
    assert [(x["k"], round(x["thr"], 5), round(x["share"], 4), round(x["mean"], 5)) for x in r["thr"]] == rows_approx([(2, 0.95, 0.0415, 0.49743), (4, 0.6316, 0.0554, 0.25278)], abs=1e-4)
    assert [(x["hop"], x["k"], round(x["mean"], 5), round(x["k_eff_measured"], 3), round(x["k_eff_formula"], 3)) for x in r["overlap"]] == rows_approx(
        [(32, 128, 0.0079, 126.513, 128.0), (16, 255, 0.00421, 237.438, 241.629)], abs=1e-3)


def test_sweep_phase_small_instance():
    r = ev.sweep_phase(seg=32, runs=20, ks=(8,), g2s=(0.5,), delay_seg=64, d=1.7, delay_runs=10, cases=((512, 0.9),))
    p = r["phase"][0]
    assert (p["k"], p["g2"]) == (8, 0.5) and p["std"] == pytest.approx(0.32759, abs=1e-5) and p["formula"] == 0.25 and p["mean"] == pytest.approx(-0.00984, abs=1e-5)
    d = r["delay"][0]
    assert (d["n"], d["k"], d["g2"]) == (512, 15, 0.9) and (d["CC"], d["PHAT"], d["ML"]) == pytest.approx((0.02068, 0.0208, 0.01151), abs=1e-5)


def test_sweep_transfer_small_instance():
    r = ev.sweep_transfer(seg=32, k=8, runs=10, outs=(0.1,), ins=(0.0, 1.0))
    assert [round(v, 5) for v in r["h_abs"]] == pytest.approx([0.92474, 0.46306], abs=1e-5)
    a, b = r["rows"]
    assert (a["h1"], a["h1_pred"], a["h2"], a["h2_pred"], a["geo"], a["phase1"]) == pytest.approx((-0.01054, 0.0, 0.3823, 0.36661, 0.15807, -0.00672), abs=1e-5)
    assert (b["h1"], b["h1_pred"], b["h2"], b["geo"], b["phase2"]) == pytest.approx((-0.46018, -0.5, 0.46689, -0.14201, 0.01667), abs=1e-5) and b["phase1"] == pytest.approx(b["phase2"])


def test_sweep_delay_small_instance():
    r = ev.sweep_delay(seg=32, k=8, runs=10, ds=(0, 2, 8))
    assert [(x["d"], round(x["coh"], 5), round(x["rho2"], 5), round(x["aligned"], 5)) for x in r["rows"]] == rows_approx([(0, 1.0, 1.0, 1.0), (2, 0.95059, 0.94988, 1.0), (8, 0.50592, 0.43448, 1.0)], abs=1e-5)


def test_sweep_wavelet_small_instance():
    r = ev.sweep_wavelet(runs=4, smoothings=((0.0, 1), (1.0, 3)), coupled=((1.0, 3),), freqs_check=(0.05,))
    assert [(x["tw"], x["ns"]) for x in r["null"]] == [(0.0, 1), (1.0, 3)] and r["null"][0]["mean"] == pytest.approx(1.0) and r["null"][0]["q95"] == pytest.approx(1.0)
    n1 = r["null"][1]
    assert (n1["mean"], n1["k_eff"], n1["q95"], n1["formula"]) == pytest.approx((0.511354, 1.955593, 0.898600, 0.956498), abs=1e-5)
    c = r["coupled"][0]
    assert (c["inside"], c["outside"], c["phase"]) == pytest.approx((0.762614, 0.514786, 0.694269), abs=1e-5)
    assert r["welch"]["seg"] == 128 and r["welch"]["k"] == 15 and (r["welch"]["coh"], r["welch"]["phase"]) == pytest.approx((0.155867, 0.696459), abs=1e-5) and r["welch"]["share"] == pytest.approx(0.390625)
    assert r["freq"][0]["f"] == pytest.approx(0.05) and r["freq"][0]["mean"] == pytest.approx(0.342843, abs=1e-5) and r["true_in"] == 0.64 and (r["start"], r["end"]) == (300, 700)


def test_one_segment_never_reports_a_significant_bin_whatever_the_rounding():
    for seed in list(range(1, 61)) + [C.DEFAULT_SEED, 41]:
        a = ev.analyse(ev.Settings(k=1, seed=seed))
        assert a.share_above == 0.0 and a.phase_rmse is None, seed


def test_sweep_bias_overlap_fraction_and_the_transfer_bins_for_other_lengths():
    r = ev.sweep_bias(seg=32, runs=4, ks=(2,), g2s=(0.0,), thr_ks=(2,), thr_runs=4, hops=(32, 16, 8))
    assert [x["overlap"] for x in r["overlap"]] == [0.0, 0.5, 0.75]
    t = ev.sweep_transfer(seg=38, k=4, runs=3, outs=(0.1,), ins=(0.0,))                         # 20 Bins: die Bins 3 und 10 liegen drin, 20 und 30 nicht
    assert len(t["h_abs"]) == 2 and ev.sweep_transfer(seg=64, k=4, runs=3, outs=(0.1,), ins=(0.0,))["h_abs"].__len__() == 4


def test_wavelet_windows_at_the_boundaries_of_the_outside_slices():
    assert ev.wavelet_windows(1024, 130, 500)[1] == [slice(550, 944)]                          # erste Außenstrecke leer (80 bis 80)
    assert ev.wavelet_windows(1024, 131, 500)[1] == [slice(80, 81), slice(550, 944)]           # Länge 1 zählt
    assert ev.wavelet_windows(1024, 129, 500)[1] == [slice(550, 944)]                          # Länge -1 zählt nicht
    assert ev.wavelet_windows(1024, 200, 894)[1] == [slice(80, 150)]                           # zweite Außenstrecke 944 bis 944: leer
