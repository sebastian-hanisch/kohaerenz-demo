"""Verdikte: reine Funktionen an handgesetzten Zeilen; Art, Schwellen und jede Zahl im Text."""

import numpy as np

import koh_evaluation as ev

Z = np.zeros(1)
DELAYS = {"CC": 3.4, "PHAT": 3.41, "ML": 3.399}


def make(scenario="Verzögerung", k=32, delay=0.0, k_eff=32.0, mc=0.55, mt=0.5, me=0.52, thr=0.09, sh=0.9, r1=None, r2=None, dl=None, w_in=0.78, w_out=0.5, wp=1.0, wc=0.17, wt=0.64, tw=1.0, ns=7, seg=64):
    s = ev.Settings(scenario=scenario, k=k, delay=delay, tw=tw, ns=ns, seg=seg)
    return ev.Analysis(s, Z, Z, k_segments=k, k_eff=k_eff, thr=thr, mean_coh=mc, mean_true=mt, mean_expected=me, share_above=sh, rel_h1=r1, rel_h2=r2, delays=dl, w_in=w_in, w_out=w_out, w_phase_in=wp, welch_coh=wc, w_true=wt)


def test_main_verdict_one_segment_is_no_measure():
    kind, text = ev.verdict_main(make(k=1, k_eff=1.0, mc=1.0, mt=0.0, me=1.0, thr=1.0, sh=0.0))
    assert kind == "warning"
    assert text == ("K = 1 Segment (äquivalent 1,0): mittlere Kohärenz im Band 1,00, wahr 0,00, erwartet 1,00; Schwelle 1,00, 0 % der Bins darüber. "
                    "Bei einem Segment ist die Kohärenz bei jeder Frequenz 1, auch bei unabhängigen Reihen: kein Maß.")


def test_main_verdict_bias_upwards_with_the_delay_estimates():
    kind, text = ev.verdict_main(make(k=2, k_eff=2.0, mc=0.75, mt=0.5, me=0.69, thr=0.95, sh=0.07, dl=DELAYS, delay=3.4))
    assert kind == "warning"
    assert text == ("K = 2 Segmente (äquivalent 2,0): mittlere Kohärenz im Band 0,75, wahr 0,50, erwartet 0,69; Schwelle 0,95, 7 % der Bins darüber. Die Kohärenz ist nach oben verzerrt (wenige Segmente). "
                    "Verzögerung gesetzt 3,40, geschätzt Kreuzkorrelation 3,400, nur Phase (PHAT) 3,410, kohärenzgewichtet (ML) 3,399.")
    assert ev.verdict_main(make(mc=0.625, mt=0.5, dl=DELAYS))[0] == "warning" and ev.verdict_main(make(mc=0.59375, mt=0.5, dl=DELAYS))[0] == "success"           # Abstand 0,125 gegen 0,094: Schwelle 0,1


def test_main_verdict_fine_estimates_for_both_welch_scenarios():
    kind, text = ev.verdict_main(make(dl=DELAYS, delay=3.4))
    assert kind == "success" and text == ("K = 32 Segmente (äquivalent 32,0): mittlere Kohärenz im Band 0,55, wahr 0,50, erwartet 0,52; Schwelle 0,09, 90 % der Bins darüber. "
                                          "Verzögerung gesetzt 3,40, geschätzt Kreuzkorrelation 3,400, nur Phase (PHAT) 3,410, kohärenzgewichtet (ML) 3,399.")
    kind, text = ev.verdict_main(make(scenario="FIR-System", mc=0.38, mt=0.38, me=0.39, r1=-0.51, r2=0.35))
    assert kind == "success" and text.endswith("90 % der Bins darüber. Betrag von H1 −51 %, von H2 35 % gegen die wahre Übertragung.")


def test_main_verdict_window_loss_from_the_delay():
    kind, text = ev.verdict_main(make(k=64, k_eff=64.0, seg=128, mc=0.78, mt=0.95, me=0.77, thr=0.05, sh=1.0, dl=DELAYS, delay=16.0))
    assert kind == "warning"
    assert text == ("K = 64 Segmente (äquivalent 64,0): mittlere Kohärenz im Band 0,78, wahr 0,95, erwartet 0,77; Schwelle 0,05, 100 % der Bins darüber. Die Kohärenz ist niedriger als die wahre: "
                    "Das Fenster ist gegen die um 16,0 verzögerte Reihe verschoben (ρ(d)² = 0,81); vor dem Schätzen ausrichten. Verzögerung gesetzt 16,00, geschätzt Kreuzkorrelation 3,400, nur Phase (PHAT) 3,410, "
                    "kohärenzgewichtet (ML) 3,399.")
    assert ev.verdict_main(make(mc=0.5, mt=0.95, dl=DELAYS, delay=0.0))[0] == "success"                              # ohne Verzögerung kein Fensterverlust
    assert ev.verdict_main(make(mc=0.86, mt=0.95, dl=DELAYS, delay=4.0))[0] == "success" and ev.verdict_main(make(mc=0.82, mt=0.95, dl=DELAYS, delay=4.0))[0] == "warning"


def test_main_verdict_wavelet_coupling_visible_or_blurred():
    kind, text = ev.verdict_main(make(scenario="Zeitweilige Kopplung"))
    assert kind == "success"
    assert text == ("Wavelet-Kohärenz bei f = 0,05 (Glättung Zeit 1,00 Skalen, Skala 7): innen 0,78 (wahr 0,64), außen 0,50 (wahr 0); Phase innen 1,00 rad (gesetzt 1,00). Welch über die ganze Aufzeichnung: 0,17. "
                    "Die Kopplung ist als Intervall zu erkennen; die Werte liegen durch die Glättung über der wahren Kohärenz.")
    kind, text = ev.verdict_main(make(scenario="Zeitweilige Kopplung", w_in=0.6, w_out=0.45))
    assert kind == "warning" and text.endswith("Der Unterschied zwischen innen und außen ist klein: Die Glättung verwischt ihn oder die Kopplung ist zu schwach.")
    assert ev.verdict_main(make(scenario="Zeitweilige Kopplung", w_in=0.75, w_out=0.5))[0] == "success" and ev.verdict_main(make(scenario="Zeitweilige Kopplung", w_in=0.625, w_out=0.5))[0] == "warning"      # 0,25 gegen 0,125


def _bias_res(first=(0.4981, 0.0634)):
    rows = []
    for k in (2, 4, 8, 16, 32, 64, 128):
        for g2 in (0.0, 0.25, 0.5):
            rows.append({"k": k, "g2": g2, "sim": g2 + (1 - g2) ** 2 / k + (0.0001 if g2 == 0.25 else 0.0), "first": g2 + (1 - g2) ** 2 / k, "exact": 0.0})
    thr = [{"k": k, "thr": 1 - 0.05 ** (1 / (k - 1)), "share": s, "mean": 0.0, "inv_k": 1 / k} for k, s in ((2, 0.051), (4, 0.052), (8, 0.046), (16, 0.052), (32, 0.047), (64, 0.049))]
    return {"seg": 64, "runs": 200, "rows": rows, "thr": thr, "thr_runs": 400, "overlap": []}


def test_verdict_bias_text():
    kind, text = ev.verdict_bias(_bias_res())
    assert kind == "warning"
    assert text == ("Bei wahrer Kohärenz 0 ist der Mittelwert der Schätzung genau 1/K (0,250 bei K = 4, 0,031 bei K = 32); bei wahrer Kohärenz 0,5 und K = 2 liegt er bei 0,62. "
                    "Die erste Ordnung g² + (1 − g²)²/K trifft ab K = 32 (0,268 gegen 0,268 bei g² = 0,25). "
                    "Die 95-%-Schwelle 1 − 0,05^(1/(K − 1)) (0,35 bei K = 8) wird bei Kopplung null von 4,6 % bis 5,2 % der Bins überschritten.")


def _phase_res():
    phase = [{"k": k, "g2": g, "std": s, "formula": f, "mean": 0.0} for k, g, s, f in ((8, 0.25, 0.56, 0.43), (8, 0.5, 0.29, 0.25), (32, 0.25, 0.225, 0.2165), (32, 0.5, 0.1275, 0.125), (128, 0.5, 0.0632, 0.0625))]
    delay = [{"n": 1024, "k": 7, "g2": 0.9, "CC": 0.0076, "PHAT": 0.0102, "ML": 0.0067}] * 4 + [{"n": 4096, "k": 31, "g2": 0.9, "CC": 0.0044, "PHAT": 0.0082, "ML": 0.0032}] * 4
    return {"seg": 64, "runs": 300, "phase": phase, "delay_seg": 256, "d": 3.4, "delay_runs": 150, "delay": delay}


def test_verdict_phase_text():
    kind, text = ev.verdict_phase(_phase_res())
    assert kind == "info"
    assert text == ("Die Phase ist unverzerrt; ihre Streuung folgt sqrt((1 − g²)/(2 K g²)) ab K = 32 (0,128 gegen 0,125 bei g² = 0,50), bei K = 8 und g² = 0,25 liegt sie darüber (0,56 gegen 0,43). "
                    "Verzögerung 3,4: bei g² = 0,9 und K = 31 sind Kreuzkorrelation, nur Phase und kohärenzgewichtet auf 0,0044, 0,0082 und 0,0032 genau (RMSE, Abtastwerte); das Gitter ist 0,005 fein.")


def _transfer_res():
    rows = []
    for out, r, h1, h2, geo in ((0.1, 0.0, -0.002, 0.361, 0.161), (0.1, 1.0, -0.499, 0.367, -0.179), (0.5, 0.0, 0.004, 1.818, 0.643), (0.5, 1.0, -0.491, 1.863, 0.165)):
        rows.append({"out": out, "r": r, "h1": h1, "h1_pred": 1 / (1 + r) - 1, "h2": h2, "h2_pred": 0.0, "geo": geo, "phase1": 0.0027, "phase2": 0.0027})
    return {"seg": 64, "k": 64, "runs": 100, "h_abs": [0.9, 0.8], "rows": rows}


def test_verdict_transfer_text():
    kind, text = ev.verdict_transfer(_transfer_res())
    assert kind == "warning"
    assert text == ("Eingangsrauschen 1,00 drückt |H1| um 50 % (Näherung 1/(1 + r) − 1: −50 %); |H2| liegt bei Ausgangsrauschen 0,1 36 % und bei 0,5 182 % über dem wahren Betrag. "
                    "Das geometrische Mittel gleicht das nicht aus (−18 % bis 64 %). Die Phase ist in beiden unverzerrt (0,003 rad).")


def _delay_res():
    rows = [{"d": d, "ratio": d / 128, "coh": c, "rho2": r, "aligned": 1.0} for d, c, r in ((0, 1.0, 1.0), (1, 0.9992, 0.9992), (2, 0.9968, 0.9968), (4, 0.9872, 0.9872), (8, 0.9499, 0.9499), (16, 0.8146, 0.8138),
                                                                                       (32, 0.4383, 0.4345), (64, 0.0426, 0.0278))]
    return {"seg": 128, "k": 64, "runs": 100, "rows": rows}


def test_verdict_delay_text():
    kind, text = ev.verdict_delay(_delay_res())
    assert kind == "warning"
    assert text == ("Bei Segmentlänge 128 fällt die Kohärenz einer reinen Verzögerung mit dem Fenster: 0,815 bei d = 16 (ρ(d)² = 0,814), 0,043 bei d = 64 (ρ(d)² = 0,028). Nach dem Zurückschieben um d ist sie wieder 1,000.")


def _wavelet_res():
    null = [{"tw": tw, "ns": ns, "mean": m, "k_eff": 1 / m, "q95": 0.9, "formula": 0.9} for tw, ns, m in ((0.0, 1, 1.0), (0.5, 1, 0.7344), (1.0, 1, 0.5293), (2.0, 1, 0.3287))]
    coupled = [{"tw": tw, "ns": ns, "inside": i, "outside": o, "phase": p} for tw, ns, i, o, p in ((0.5, 1, 0.8546, 0.7374, 1.001), (1.0, 1, 0.7762, 0.5315, 0.991), (2.0, 1, 0.711, 0.3329, 0.979))]
    return {"runs": 100, "n": 1024, "f0": 0.05, "null": null, "coupled": coupled, "welch": {"seg": 128, "k": 15, "f": 0.0469, "coh": 0.1739, "phase": 0.9989, "share": 0.390625}, "freq": [], "true_in": 0.64, "start": 300,
            "end": 700}


def test_verdict_wavelet_text():
    kind, text = ev.verdict_wavelet(_wavelet_res())
    assert kind == "warning"
    assert text == ("Ohne Glättung ist die Wavelet-Kohärenz 1,000 bei Kopplung null; mit Glättung (Zeit 1 Skala) 0,53 (K_eff 1,9). Bei zeitweiliger Kopplung (wahr 0,64) liegt sie innen bei 0,78 und außen bei 0,53, "
                    "die Phase innen bei 0,99 rad (gesetzt 1,00); Welch über die ganze Aufzeichnung sieht 0,17 (Anteil der Kopplungszeit 0,39).")


def test_verdict_thresholds_at_exact_float_equality():
    assert ev.verdict_main(make(scenario="Zeitweilige Kopplung", w_in=0.16, w_out=0.01))[0] == "warning"                       # Abstand genau 0,15 (0,16 - 0,01 = 0,15 in Gleitkomma)
    assert ev.verdict_main(make(scenario="Zeitweilige Kopplung", w_in=0.17, w_out=0.01))[0] == "success"
    assert ev.verdict_main(make(mc=0.11, mt=0.01, dl=DELAYS))[0] == "success"                                                  # genau 0,1 zählt noch nicht als verzerrt
    assert ev.verdict_main(make(mc=0.12, mt=0.01, dl=DELAYS))[0] == "warning"
    assert ev.verdict_main(make(mc=0.01, mt=0.11, dl=DELAYS, delay=4.0))[0] == "success"                                        # Fensterverlust genau 0,1 zählt noch nicht
    assert ev.verdict_main(make(mc=0.01, mt=0.12, dl=DELAYS, delay=4.0))[0] == "warning"
    assert ev.verdict_main(make(mc=0.5, mt=0.95, dl=DELAYS, delay=0.5))[0] == "warning"                                         # schon eine Verzögerung unter 1 gilt
