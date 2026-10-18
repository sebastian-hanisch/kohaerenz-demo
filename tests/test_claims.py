"""Jede Zahl und Aussage in den Texten der App (Presets, Einleitung, Grenzen, Experiment-Überschriften) und in der README ist hier an Rechnung oder Messung gebunden."""

import math
from pathlib import Path

import numpy as np
import pytest

import koh_constants as C
import koh_delay as D
import koh_evaluation as ev
import koh_spectra as SP
import koh_theory as T
from sweeps_cache import get

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
APP = (ROOT / "app.py").read_text(encoding="utf-8")
de = ev.de
P = list(C.PRESETS)


def preset(name, **over):
    p = C.PRESETS[name]
    base = dict(scenario=p["scenario"], seg=p["seg"], k=p["k"], overlap=p["overlap"], g2=p["g2"], delay=p["delay"], out=p["out"], inn=p["inn"], start=p["start"], end=p["end"], amp=p["amp"], tw=p["tw"],
                ns=p["ns"], seed=p["seed"])
    return ev.analyse(ev.Settings(**{**base, **over}))


# --- Presets ----------------------------------------------------------------------------------------------------------------------------------


def test_preset_one_segment_gives_coherence_one_everywhere():
    a = preset(P[0])
    assert a.k_segments == 1 and a.mean_true == 0.0 and np.allclose(a.coh, 1.0) and a.share_above == 0.0
    h = C.PRESET_HELP[P[0]]
    assert "wahre Kohärenz 0" in h and "nur ein Segment" in h and "genau 1" in h


def test_preset_two_segments_measure_about_0_69():
    a = preset(P[1])
    assert a.k_segments == 2 and a.settings.g2 == 0.5 and de(a.mean_expected, 2) == "0,69" and abs(a.mean_coh - 0.69) < 0.01
    h = C.PRESET_HELP[P[1]]
    assert "Zwei Segmente, wahre Kohärenz 0,5" in h and "bei 0,69" in h and "hier 0,69" in h and "gemessen fast 0,7" in P[1]


def test_preset_delay_window_loss_of_about_20_percent():
    a = preset(P[2])
    rho2 = T.window_correlation(SP.hann(128), 16.0) ** 2
    assert de(rho2, 2) == "0,81" and a.settings.delay == 16.0 and a.settings.seg == 128 and a.settings.g2 == 0.95
    assert 0.15 < 1 - a.mean_coh / a.mean_true < 0.25 and all(abs(v - 16.0) < 0.02 for v in a.delays.values())
    h = C.PRESET_HELP[P[2]]
    assert "Verzögerung 16 Abtastwerte bei Segmentlänge 128" in h and "wahrer Kohärenz 0,95" in h and "etwa 20 % tiefer" in h


def test_preset_input_noise_halves_h1_and_raises_h2():
    a = preset(P[3])
    assert a.settings.inn == 1.0 and 0.45 < -a.rel_h1 < 0.55 and a.rel_h2 > 0.2 and abs(a.phase_rmse) < 0.2
    h = C.PRESET_HELP[P[3]]
    assert "Eingangsrauschen von der Größe des Eingangs" in h and "bei der Hälfte des wahren Betrags" in h and "H2 = S_yy/S_yx darüber" in h


def test_preset_wavelet_coupling_numbers():
    a = preset(P[4])
    assert a.settings.start == 300 and a.settings.end == 700 and de(a.w_in, 2) == "0,75" and de(a.w_out, 2) == "0,49" and de(a.welch_coh, 2) == "0,25" and abs(a.w_phase_in - C.PHASE0) < 0.05
    h = C.PRESET_HELP[P[4]]
    assert "zwischen 300 und 700" in h and "innen 0,75, außen 0,49" in h and "sieht nur 0,25" in h and "bei 0,05" in h and C.F0 == 0.05


# --- README: Befunde --------------------------------------------------------------------------------------------------------------------------


def test_readme_bias():
    b = get("bias")
    sim = {(r["k"], r["g2"]): r for r in b["rows"]}
    assert "Bei einem Segment ist sie bei jeder Frequenz 1, auch bei unabhängigen Reihen." in README
    assert (f"(K = 8: {de(sim[(8, 0.0)]['sim'], 3)} gemessen {de(sim[(8, 0.0)]['exact'], 3)}; K = 32: {de(sim[(32, 0.0)]['exact'], 3)} gemessen {de(sim[(32, 0.0)]['sim'], 3)})") in README
    assert sim[(8, 0.0)]["exact"] == pytest.approx(0.125) and abs(sim[(8, 0.0)]["sim"] - 0.125) < 0.001
    assert (f"liegt er bei K = 2 bei {de(sim[(2, 0.5)]['sim'], 2)} (exakt ln 2 = {de(math.log(2), 3)}), bei K = 8 bei {de(sim[(8, 0.5)]['sim'], 2)} und bei K = 32 bei {de(sim[(32, 0.5)]['sim'], 2)}") in README
    assert sim[(2, 0.5)]["exact"] == pytest.approx(math.log(2), abs=2e-4)
    r = sim[(32, 0.25)]
    assert f"trifft ab K = 32 (g² = 0,25: {de(r['sim'], 3)} gemessen, {de(r['first'], 3)})" in README and abs(r["sim"] - r["first"]) < 0.001
    thr = {r["k"]: r for r in b["thr"]}
    assert f"(K = 2: {de(thr[2]['thr'], 2)}; K = 8: {de(thr[8]['thr'], 2)}; K = 64: {de(thr[64]['thr'], 3)})" in README
    shares = [r["share"] for r in b["thr"]]
    assert f"von {de(100 * min(shares), 1)} % bis {de(100 * max(shares), 1)} % der Bins überschritten" in README and 0.04 < min(shares) and max(shares) < 0.06
    ov = {r["hop"]: r for r in b["overlap"]}
    assert (f"Bei 50 % Überlappung entsprechen {ov[32]['k']} Segmente {de(ov[32]['k_eff_formula'], 0)} unabhängigen (gemessen {de(ov[32]['k_eff_measured'], 0)}), "
            f"bei 75 % entsprechen {ov[16]['k']} Segmente {de(ov[16]['k_eff_formula'], 0)} (gemessen {de(ov[16]['k_eff_measured'], 0)})") in README
    assert f"(120,4 gegen gemessen 121,6 bei 50 %)" in README and de(ov[32]["k_eff_formula"], 1) == "120,4" and de(ov[32]["k_eff_measured"], 1) == "121,6"


def test_readme_phase_and_delay():
    p = get("phase")
    ph = {(r["k"], r["g2"]): r for r in p["phase"]}
    assert max(abs(r["mean"]) for r in p["phase"]) < 0.005 + 1e-9 and "(Mittel im Bereich ± 0,005 rad)" in README
    assert (f"(γ² = 0,5: {de(ph[(32, 0.5)]['std'], 3)} gegen {de(ph[(32, 0.5)]['formula'], 3)}; K = 128: {de(ph[(128, 0.5)]['std'], 3)} gegen {de(ph[(128, 0.5)]['formula'], 3)})") in README
    assert f"bei K = 8 und γ² = 0,25 liegt sie darüber ({de(ph[(8, 0.25)]['std'], 2)} gegen {de(ph[(8, 0.25)]['formula'], 2)})" in README and ph[(8, 0.25)]["std"] > 1.2 * ph[(8, 0.25)]["formula"]
    dl = {(r["n"], r["g2"]): r for r in p["delay"]}
    a, b = dl[(4096, 0.9)], dl[(4096, 0.05)]
    assert f"bei γ² = 0,9 und K = {a['k']} mit einem RMSE von {de(a['CC'], 4)} (Kreuzkorrelation), {de(a['PHAT'], 4)} (nur Phase, PHAT) und {de(a['ML'], 4)} (kohärenzgewichtet)" in README
    assert f"bei γ² = 0,05 sind alle ähnlich schlecht ({de(b['CC'], 3)}, {de(b['PHAT'], 3)} und {de(b['ML'], 3)})" in README and a["ML"] < a["CC"] < a["PHAT"] and max(b["CC"], b["PHAT"], b["ML"]) < 1.2 * min(b["CC"], b["PHAT"], b["ML"])
    assert p["d"] == 3.4 and "Das Gitter ist 0,005 fein." in README and C.DELAY_STEP == 0.005


def test_readme_transfer():
    t = get("transfer")
    rows = {(r["out"], r["r"]): r for r in t["rows"]}
    assert (f"(r = 0,25: {de(100 * rows[(0.1, 0.25)]['h1'], 0)} %; r = 1: {de(100 * rows[(0.1, 1.0)]['h1'], 0)} %)") in README and "(r = 0,25: −20 %; r = 1: −50 %)" in README
    assert f"bei Ausgangsrauschen 0,1 um {de(100 * rows[(0.1, 0.0)]['h2'], 0)} % und bei 0,5 um {de(100 * rows[(0.5, 0.0)]['h2'], 0)} % zu hoch" in README
    geo = [r["geo"] for r in t["rows"]]
    assert f"({de(100 * min(geo), 0)} % bis {de(100 * max(geo), 0)} %)" in README and "(−18 % bis 64 %)" in README
    assert max(abs(r["phase1"]) for r in t["rows"]) < 0.01 and "(unter 0,01 rad)" in README
    for r in t["rows"]:
        assert abs(r["h1"] - r["h1_pred"]) < 0.02 and abs(r["h2"] - r["h2_pred"]) < 0.1 and r["h2"] > r["h2_pred"] - 0.001


def test_readme_delay_versus_window():
    d = {r["d"]: r for r in get("delay")["rows"]}
    assert f"(gemessen d = 16: {de(d[16]['coh'], 3)}; d = 32: {de(d[32]['coh'], 3)}; d = 64: {de(d[64]['coh'], 3)};" in README and " / ".join(de(d[k]['rho2'], 3) for k in (16, 32, 64)) == "0,814 / 0,434 / 0,028" and "ρ(d)² selbst 0,814 / 0,434 / 0,028" in README and d[64]["coh"] > d[64]["rho2"] and "Segmentlänge 128" in README
    assert all(abs(r["aligned"] - 1.0) < 1e-9 for r in d.values()) and "Zurückschieben um d stellt 1,000 her" in README
    assert abs(d[16]["coh"] - d[16]["rho2"]) < 0.002 and abs(d[32]["coh"] - d[32]["rho2"]) < 0.01


def test_readme_wavelet():
    w = get("wavelet")
    null = {(r["tw"], r["ns"]): r for r in w["null"]}
    cp = {(r["tw"], r["ns"]): r for r in w["coupled"]}
    n0, n1, n2 = null[(0.0, 1)], null[(1.0, 1)], null[(2.0, 7)]
    assert f"Ohne Glättung ist sie {de(n0['mean'], 3)} bei Kopplung null; mit Glättung (Zeit 1 Skala) {de(n1['mean'], 2)} (K_eff {de(n1['k_eff'], 1)}), bei Zeit 2 und Skala 7 noch {de(n2['mean'], 2)}" in README
    assert f"das 95-%-Quantil ({de(n1['q95'], 2)}) liegt unter der Beta-Formel mit K_eff ({de(n1['formula'], 2)})" in README and n1["q95"] < n1["formula"]
    c = cp[(1.0, 1)]
    assert (f"liegt sie innen bei {de(c['inside'], 2)} und außen bei {de(c['outside'], 2)}, die Phase innen bei {de(c['phase'], 2)} rad (gesetzt {de(C.PHASE0, 2)}); Welch über die ganze Aufzeichnung sieht {de(w['welch']['coh'], 2)}") in README
    assert "(wahre Kohärenz 0,64)" in README and w["true_in"] == 0.64
    fr = [r["mean"] for r in w["freq"]]
    assert f"({de(min(fr), 2)} bis {de(max(fr), 2)})" in README and "(0,33 bis 0,35)" in README and de(null[(1.0, 7)]["mean"], 2) == "0,45" and "Zeit 1, Skala 7) bei 0,45" in README
    assert (de(min(r["k_eff"] for k, r in null.items() if k != (0.0, 1)), 1), de(max(r["k_eff"] for k, r in null.items() if k != (0.0, 1)), 1)) == ("1,4", "3,5") and "nur 1,4 bis 3,5 äquivalente Segmente" in README and (de(min(r["mean"] for k, r in null.items() if k != (0.0, 1)), 2), de(max(r["mean"] for k, r in null.items() if k != (0.0, 1)), 2)) == ("0,29", "0,73") and "Bias 0,29 bis 0,73 bei Kopplung null" in README


# --- README: Methodik, Modell, Grenzen ---------------------------------------------------------------------------------------------------------


def test_readme_methodology_numbers():
    assert C.EXP_SEED == 2100 and "Seed 2100" in README and C.EXP_BIAS_RUNS == 200 and C.EXP_BIAS_SEG == 64 and "200 Läufe beim Bias (Segmentlänge 64, Bins 2 bis 30)" in README
    assert C.EXP_THR_RUNS == 400 and "400 Läufe bei der Schwelle" in README and C.EXP_PHASE_RUNS == 300 and "300 Läufe bei der Phase" in README
    assert C.EXP_DELAY_RUNS == 150 and C.EXP_DELAY_SEG == 256 and C.EXP_DELAY_D == 3.4 and "150 Läufe bei der Verzögerung (Segmentlänge 256,\nd = 3,4)" in README.replace("(Segmentlänge 256,\nd", "(Segmentlänge 256,\nd")
    assert C.EXP_TRANSFER_RUNS == C.EXP_WINDOW_RUNS == C.EXP_WAVELET_RUNS == 100 and C.EXP_TRANSFER_K == C.EXP_WINDOW_K == 64 and C.EXP_WINDOW_SEG == 128 and C.N_WAVELET == 1024
    assert "100 Läufe bei der Übertragungsfunktion (K = 64)" in README and "(128, K = 64)" in README and "n = 1024, f₀ = 0,05" in README and C.EXP_BIAS_SEG // 2 - 2 == 30


def test_readme_model_numbers():
    assert C.FIR == (0.0, 0.6, 0.3, 0.1) and "h = (0; 0,6; 0,3; 0,1)" in README and C.F0 == 0.05 and C.F_HALF == 0.01 and "bei 0,05 ± 0,01 (Leistung 1)" in README and C.PHASE0 == 1.0 and "Phase φ = 1,0 rad" in README
    assert C.W0 == 6.0 and "(ω₀ = 6)" in README and C.VOICES == 12 and "12 je Oktave" in README and C.DELAY_STEP == 0.005 and "Schrittweite 0,005" in README
    assert abs(T.true_coherence_fir([2.0], 1.0, 1.0)[0] - 4.0 / (2.0 * 5.0)) < 1e-12 and "wahre Kohärenz |H|²/((1 + r)(|H|² + N))" in README
    assert abs(1.0 / (1.0 + 0.5 ** 2) ** 2 - 0.64) < 1e-12 and "1/(1 + a²)²" in README


def test_readme_limit_about_k_eff_and_sections():
    for section in ("## Warum dieses Problem", "## Modell", "## Methodik", "## Befunde (gemessen, keine Behauptungen)", "## Ehrliche Grenzen", "## Verifikation", "## Dateistruktur", "## Bewusst nicht umgesetzt"):
        assert section in README
    for f in ROOT.glob("koh_*.py"):
        assert f"`{f.name}`" in README, f.name
    assert "scipy" in (ROOT / "requirements-dev.txt").read_text() and "scipy" not in (ROOT / "requirements.txt").read_text()


# --- App-Texte --------------------------------------------------------------------------------------------------------------------------------


def test_app_texts():
    assert "dreizehnte" not in APP and "fünfzehnte Demo" in APP and "Stück 15" in APP and "Bei einem Segment ist sie immer 1" in APP
    assert "1 − 0,05^(1/(K − 1))" in APP and T.threshold(8) == pytest.approx(1 - 0.05 ** (1 / 7)) and "Mit K = 1 ist die Kohärenz immer 1" in APP
    assert "wahre Kohärenz im Intervall ist 1/(1 + Amplitude²)²" in APP and "12 je Oktave" in APP and "Mindestens 40" not in APP and "mindestens 40 Abtastwerte" in APP
    assert "Gitter 0,005" in APP and C.DELAY_STEP == 0.005 and "Segmentlänge 128, 50 % Überlappung" in APP and C.WELCH_SEG_WAVELET == 128
    assert "Abtastwert, ab dem y das gemeinsame Band enthält" in APP and "ohne Glättung ... " not in APP and "Kohärenz dann überall 1" in APP
    assert "H_1 = S_{xy}/S_{xx}" in APP and "Hann-Fenster $w$" in APP and "equiv 1" in APP
    assert "außerhalb des Bandes 0,04 bis 0,06" in APP and C.F0 - C.F_HALF == pytest.approx(0.04) and C.F0 + C.F_HALF == pytest.approx(0.06)
    assert set(D.METHODS) == {"CC", "PHAT", "ML"} and "(Carter, Knapp und Nuttall 1973" in APP
