"""Analyse, Kennzahlen, Experimente (feste Ströme) und Verdikte der Demo. Reine Funktionen; Zufall nur über SplitMix64."""

import math
from dataclasses import dataclass

import numpy as np

import koh_constants as C
import koh_delay as D
import koh_signals as S
import koh_spectra as SP
import koh_theory as T
import koh_wavelet as W
import signal_generator as G


def de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma und Minuszeichen; „–“ für nicht definierte Werte."""
    if x != x:
        return "–"
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


@dataclass(frozen=True)
class Settings:
    scenario: str = C.DEFAULT_SCENARIO
    seg: int = C.DEFAULT_SEG
    k: int = C.DEFAULT_K
    overlap: str = C.DEFAULT_OVERLAP
    g2: float = C.DEFAULT_G2
    delay: float = C.DEFAULT_D
    out: float = C.DEFAULT_OUT
    inn: float = C.DEFAULT_IN
    start: int = C.DEFAULT_START
    end: int = C.DEFAULT_END
    amp: float = C.DEFAULT_AMP
    tw: float = C.DEFAULT_TW
    ns: int = C.DEFAULT_NS
    seed: int = C.DEFAULT_SEED

    @property
    def welch(self) -> bool:
        return self.scenario in C.WELCH_SCENARIOS

    @property
    def hop(self) -> int:
        return SP.hop_for(self.seg, 0.5 if self.overlap == "50 %" else 0.0)

    @property
    def n(self) -> int:
        return SP.record_length(self.seg, self.hop, self.k) if self.welch else C.N_WAVELET


def band_slice(seg: int, edge: int = C.EDGE) -> slice:
    """Bins edge .. seg/2 - edge (einschließlich): ohne Gleichanteil und Nyquist-Rand."""
    return slice(edge, seg // 2 - edge + 1)


def build_measurement(s: Settings) -> dict:
    """Die Reihen des Szenarios: {'x': beobachteter Eingang, 'y': Ausgang, …}; bei der zeitweiligen Kopplung zusätzlich 'g' (Kopplungsmaske)."""
    rng = G.SplitMix64(s.seed)
    n = s.n
    if s.scenario == "Zeitweilige Kopplung":
        x, y, g, _c = S.coupled_pair(rng, None, n, s.start, s.end, s.amp)
        return {"x": x, "y": y, "g": g}
    x, e, v = S.noise(rng, None, n), S.noise(rng, None, n), S.noise(rng, None, n)
    if s.scenario == "Verzögerung":
        return {"x": x, "y": S.delayed_pair(x, e, s.delay, s.g2)}
    return {"x": x + math.sqrt(s.inn) * v, "y": S.fir(x) + math.sqrt(s.out) * e}


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    x: np.ndarray
    y: np.ndarray
    # Welch (nur bei Verzögerung und FIR-System)
    freqs: np.ndarray | None = None
    coh: np.ndarray | None = None
    phase: np.ndarray | None = None
    true_coh: np.ndarray | None = None
    true_phase: np.ndarray | None = None
    expected: np.ndarray | None = None
    k_segments: int = 0
    k_eff: float = 0.0
    thr: float = 1.0
    mean_coh: float = float("nan")
    mean_true: float = float("nan")
    mean_expected: float = float("nan")
    share_above: float = float("nan")
    phase_rmse: float | None = None
    h1: np.ndarray | None = None
    h2: np.ndarray | None = None
    h_true: np.ndarray | None = None
    rel_h1: float | None = None
    rel_h2: float | None = None
    delays: dict | None = None
    # Wavelet-Kohärenz (nur bei der zeitweiligen Kopplung)
    wfreqs: np.ndarray | None = None
    wcoh: np.ndarray | None = None
    wphase: np.ndarray | None = None
    mask: np.ndarray | None = None
    i0: int = 0
    w_in: float = float("nan")
    w_out: float = float("nan")
    w_phase_in: float = float("nan")
    welch_coh: float = float("nan")
    welch_phase: float = float("nan")
    w_true: float = float("nan")


def wavelet_windows(n: int, start: int, end: int) -> tuple[slice, list[slice]]:
    """Zeitfenster der Wavelet-Kennzahlen: innen [start + m, end - m) mit m = min(100, Länge / 4); außen [80, start - 50) und [end + 50, n - 80) (leere entfallen)."""
    m = min(100, (end - start) // 4)
    inside = slice(start + m, end - m)
    outs = [sl for sl in (slice(80, start - 50), slice(end + 50, n - 80)) if sl.stop - sl.start > 0]
    return inside, outs


def _circ_mean(angles) -> float:
    return float(np.angle(np.exp(1j * np.asarray(angles)).mean()))


def analyse(s: Settings) -> Analysis:
    m = build_measurement(s)
    if s.scenario == "Zeitweilige Kopplung":
        return _analyse_wavelet(s, m)
    return _analyse_welch(s, m)


def _analyse_welch(s: Settings, m: dict) -> Analysis:
    x, y = m["x"], m["y"]
    w = SP.hann(s.seg)
    sxx, syy, sxy, k = SP.cross_spectra(x, y, s.seg, s.hop, w)
    coh = SP.coherence(sxx, syy, sxy)
    f = SP.freqs(s.seg)
    k_eff = T.k_effective(w, s.hop, k)
    kk = max(1, int(round(k_eff)))
    sl = band_slice(s.seg)
    thr = T.threshold(k_eff)
    h1 = h2 = h_true = rel1 = rel2 = delays = None
    if s.scenario == "Verzögerung":
        rho = T.window_correlation(w, s.delay)
        true_coh = np.full(len(f), s.g2)
        true_phase = np.angle(np.exp(-2j * np.pi * f * s.delay))
        g_eff = np.full(len(f), s.g2 * rho ** 2)
        half = max(C.DELAY_RANGE, math.ceil(abs(s.delay)) + C.DELAY_RANGE)
        delays = {meth: float(D.estimate(sxy, coh, s.seg, meth, half)) for meth in D.METHODS}
    else:
        h_true = S.fir_response(f)
        true_coh = T.true_coherence_fir(np.abs(h_true), s.out, s.inn)
        true_phase = np.angle(h_true)
        g_eff = true_coh
        h1, h2 = SP.transfer_h1(sxx, sxy), SP.transfer_h2(syy, sxy)
        rel1 = float((np.abs(h1[sl]) / np.abs(h_true[sl]) - 1).mean())
        rel2 = float((np.abs(h2[sl]) / np.abs(h_true[sl]) - 1).mean())
    expected = np.array([T.expected_coherence(float(g), kk, 1001) for g in g_eff])
    ph = SP.phase(sxy)
    sig = (coh > thr) if k_eff > 1.0 else np.zeros(len(f), bool)                # bei einem Segment gibt es keine Schwelle (die Kohärenz ist überall 1)
    sig_band = np.zeros(len(f), bool)
    sig_band[sl] = sig[sl]
    err = np.angle(np.exp(1j * (ph - true_phase)))
    return Analysis(s, x, y, f, coh, ph, true_coh, true_phase, expected, k, k_eff, thr, float(coh[sl].mean()), float(true_coh[sl].mean()), float(expected[sl].mean()), float(sig[sl].mean()),
                    float(math.sqrt(np.mean(err[sig_band] ** 2))) if sig_band.any() else None, h1, h2, h_true, rel1, rel2, delays)


def _analyse_wavelet(s: Settings, m: dict) -> Analysis:
    x, y, g = m["x"], m["y"], m["g"]
    freqs = W.freq_grid()
    r2, ph = W.wavelet_coherence(x, y, freqs, s.tw, s.ns)
    i0 = int(np.argmin(np.abs(freqs - C.F0)))
    inside, outs = wavelet_windows(s.n, s.start, s.end)
    w_in = float(r2[i0, inside].mean()) if inside.stop > inside.start else float("nan")
    w_out = float(np.concatenate([r2[i0, o] for o in outs]).mean()) if outs else float("nan")
    w_phase_in = _circ_mean(ph[i0, inside]) if inside.stop > inside.start else float("nan")
    seg = C.WELCH_SEG_WAVELET
    sxx, syy, sxy, k = SP.cross_spectra(x, y, seg, seg // 2)
    fb = SP.freqs(seg)
    ib = int(np.argmin(np.abs(fb - C.F0)))
    coh = SP.coherence(sxx, syy, sxy)
    return Analysis(s, x, y, wfreqs=freqs, wcoh=r2, wphase=ph, mask=g, i0=i0, w_in=w_in, w_out=w_out, w_phase_in=w_phase_in, welch_coh=float(coh[ib]), welch_phase=float(np.angle(sxy[ib])),
                    w_true=1.0 / (1.0 + s.amp ** 2) ** 2)


# --- Experimente (feste Ströme; Standardwerte aus koh_constants) ------------------------------------------------------------------------------


def _band(seg: int, lo: int = 2, hi: int | None = None) -> slice:
    return slice(lo, (seg // 2 - 2 if hi is None else hi) + 1)


def sweep_bias(seg: int = C.EXP_BIAS_SEG, runs: int = C.EXP_BIAS_RUNS, ks=C.EXP_BIAS_KS, g2s=C.EXP_BIAS_G2S, thr_ks=C.EXP_THR_KS, thr_runs: int = C.EXP_THR_RUNS,
               hops=(64, 48, 32, 16), seed: int = C.EXP_SEED) -> dict:
    """E1: Mittel der geschätzten Kohärenz über K unabhängige Segmente (Simulation, erste Ordnung, exakt), die 95-%-Schwelle gegen die Falschalarmrate bei Kopplung null und die äquivalente Segmentzahl bei Überlappung."""
    rng = G.SplitMix64(seed)
    sl = _band(seg)
    rows = []
    for k in ks:
        for g2 in g2s:
            x, e = S.noise(rng, runs, k * seg), S.noise(rng, runs, k * seg)
            y = S.delayed_pair(x, e, 0.0, g2)
            sxx, syy, sxy, _k = SP.cross_spectra(x, y, seg, seg)
            rows.append({"k": k, "g2": g2, "sim": float(SP.coherence(sxx, syy, sxy)[:, sl].mean()), "first": T.expected_first_order(g2, k), "exact": T.expected_coherence(g2, k)})
    thr = []
    for k in thr_ks:
        x, e = S.noise(rng, thr_runs, k * seg), S.noise(rng, thr_runs, k * seg)
        sxx, syy, sxy, _k = SP.cross_spectra(x, e, seg, seg)
        c = SP.coherence(sxx, syy, sxy)[:, sl]
        t = T.threshold(k)
        thr.append({"k": k, "thr": t, "share": float(np.mean(c > t)), "mean": float(c.mean()), "inv_k": 1.0 / k})
    overlap = []
    w = SP.hann(seg)
    for hop in hops:
        x, e = S.noise(rng, 300, 4096), S.noise(rng, 300, 4096)
        sxx, syy, sxy, k = SP.cross_spectra(x, e, seg, hop)
        m = float(SP.coherence(sxx, syy, sxy)[:, sl].mean())
        overlap.append({"hop": hop, "overlap": 1.0 - hop / seg, "k": k, "mean": m, "k_eff_measured": 1.0 / m, "k_eff_formula": T.k_effective(w, hop, k)})
    return {"seg": seg, "runs": runs, "rows": rows, "thr": thr, "thr_runs": thr_runs, "overlap": overlap}


def sweep_phase(seg: int = C.EXP_PHASE_SEG, runs: int = C.EXP_PHASE_RUNS, ks=C.EXP_PHASE_KS, g2s=C.EXP_PHASE_G2S, delay_seg: int = C.EXP_DELAY_SEG, d: float = C.EXP_DELAY_D,
                delay_runs: int = C.EXP_DELAY_RUNS, cases=C.EXP_DELAY_CASES, seed: int = C.EXP_SEED + 100) -> dict:
    """E2: Streuung der Phase von S_xy gegen sqrt((1 - g2) / (2 K g2)) und RMSE dreier Verzögerungsschätzer (gebrochene Verzögerung d, Hann, 50 % Überlappung)."""
    rng = G.SplitMix64(seed)
    f = SP.freqs(seg)
    sl = _band(seg)
    phase_rows = []
    for k in ks:
        for g2 in g2s:
            x, e = S.noise(rng, runs, k * seg), S.noise(rng, runs, k * seg)
            y = S.delayed_pair(x, e, 1.0, g2)
            _sxx, _syy, sxy, _k = SP.cross_spectra(x, y, seg, seg)
            ph = np.angle(sxy * np.exp(2j * np.pi * f * 1.0))[:, sl]
            phase_rows.append({"k": k, "g2": g2, "std": float(ph.std()), "formula": T.phase_std(g2, k), "mean": float(ph.mean())})
    delay_rows = []
    for n, g2 in cases:
        x, e = S.noise(rng, delay_runs, n), S.noise(rng, delay_runs, n)
        y = S.delayed_pair(x, e, d, g2)
        sxx, syy, sxy, k = SP.cross_spectra(x, y, delay_seg, delay_seg // 2)
        c = SP.coherence(sxx, syy, sxy)
        row = {"n": n, "k": k, "g2": g2}
        for meth in D.METHODS:
            est = D.estimate(sxy, c, delay_seg, meth)
            row[meth] = float(math.sqrt(np.mean((est - d) ** 2)))
        delay_rows.append(row)
    return {"seg": seg, "runs": runs, "phase": phase_rows, "delay_seg": delay_seg, "d": d, "delay_runs": delay_runs, "delay": delay_rows}


def sweep_transfer(seg: int = C.EXP_TRANSFER_SEG, k: int = C.EXP_TRANSFER_K, runs: int = C.EXP_TRANSFER_RUNS, outs=C.EXP_TRANSFER_OUT, ins=C.EXP_TRANSFER_IN,
                   seed: int = C.EXP_SEED + 200) -> dict:
    """E3: H1 und H2 bei Eingangs- und Ausgangsrauschen: mittlere relative Abweichung von |H| (Bins 3 bis 30) gegen die Näherungen 1/(1 + r) - 1 und 1 + N/|H|^2 - 1, Phasenabweichung."""
    rng = G.SplitMix64(seed)
    n = k * seg
    f = SP.freqs(seg)
    h = S.fir_response(f)
    sl = _band(seg, 3)
    rows = []
    for out in outs:
        for r in ins:
            x = S.noise(rng, runs, n)
            xt = S.fir(x)
            y = xt + math.sqrt(out) * S.noise(rng, runs, n)
            xo = x + math.sqrt(r) * S.noise(rng, runs, n)
            sxx, syy, sxy, _k = SP.cross_spectra(xo, y, seg, seg)
            h1, h2 = SP.transfer_h1(sxx, sxy), SP.transfer_h2(syy, sxy)
            a1, a2 = np.abs(h1[:, sl]), np.abs(h2[:, sl])
            rows.append({"out": out, "r": r, "h1": float((a1 / np.abs(h[sl]) - 1).mean()), "h1_pred": T.h1_factor(r) - 1.0, "h2": float((a2 / np.abs(h[sl]) - 1).mean()),
                         "h2_pred": float(np.mean(T.h2_factor(np.abs(h[sl]), out)) - 1.0), "geo": float((np.sqrt(a1 * a2) / np.abs(h[sl]) - 1).mean()),
                         "phase1": float(np.angle(h1[:, sl] / h[sl]).mean()), "phase2": float(np.angle(h2[:, sl] / h[sl]).mean())})
    return {"seg": seg, "k": k, "runs": runs, "h_abs": [float(np.abs(h[i])) for i in (3, 10, 20, 30) if i < len(h)], "rows": rows}


def sweep_delay(seg: int = C.EXP_WINDOW_SEG, k: int = C.EXP_WINDOW_K, runs: int = C.EXP_WINDOW_RUNS, ds=C.EXP_WINDOW_DS, seed: int = C.EXP_SEED + 300) -> dict:
    """E4: Kohärenz bei reiner Verzögerung d (fast ohne Rauschen) gegen rho(d)^2 und nach dem Ausrichten."""
    rng = G.SplitMix64(seed)
    n = k * seg
    w = SP.hann(seg)
    sl = _band(seg, 4, seg // 2 - 4)
    rows = []
    for d in ds:
        x = S.noise(rng, runs, n)
        y = S.frac_delay(x, d)
        sxx, syy, sxy, _k = SP.cross_spectra(x, y, seg, seg)
        c = float(SP.coherence(sxx, syy, sxy)[:, sl].mean())
        sa = SP.cross_spectra(x, D.align(y, d), seg, seg)
        rows.append({"d": d, "ratio": d / seg, "coh": c, "rho2": T.window_correlation(w, d) ** 2, "aligned": float(SP.coherence(*sa[:3])[:, sl].mean())})
    return {"seg": seg, "k": k, "runs": runs, "rows": rows}


def sweep_wavelet(runs: int = C.EXP_WAVELET_RUNS, smoothings=C.EXP_WAVELET_SMOOTH, coupled=C.EXP_WAVELET_COUPLED, freqs_check=C.EXP_WAVELET_FREQS, seed: int = C.EXP_SEED + 400) -> dict:
    """E5: Wavelet-Kohärenz bei Kopplung null (Bias, K_eff, Quantil) und bei zeitweiliger Kopplung (innen, außen, Phase), dazu Welch über die ganze Aufzeichnung."""
    n = C.N_WAVELET
    rng = G.SplitMix64(seed)
    fr = W.freq_grid()
    i0 = int(np.argmin(np.abs(fr - C.F0)))
    a, b = S.band_noise(rng, runs, n), S.band_noise(rng, runs, n)
    Wa, Wb = W.cwt(a, fr), W.cwt(b, fr)
    null = []
    for tw, ns in smoothings:
        r2, _ph = W.coherence_from(Wa, Wb, fr, tw, ns)
        r = r2[:, i0, 200:800]
        keff = 1.0 / float(r.mean())
        null.append({"tw": tw, "ns": ns, "mean": float(r.mean()), "k_eff": keff, "q95": float(np.quantile(r, 0.95)), "formula": T.threshold(keff) if keff > 1.0001 else 1.0})
    start, end, amp = C.DEFAULT_START, C.DEFAULT_END, C.DEFAULT_AMP
    x, y, g, _c = S.coupled_pair(rng, runs, n, start, end, amp)
    Wx, Wy = W.cwt(x, fr), W.cwt(y, fr)
    inside, outs = wavelet_windows(n, start, end)
    coupled_rows = []
    for tw, ns in coupled:
        r2, ph = W.coherence_from(Wx, Wy, fr, tw, ns)
        coupled_rows.append({"tw": tw, "ns": ns, "inside": float(r2[:, i0, inside].mean()), "outside": float(np.concatenate([r2[:, i0, o].ravel() for o in outs]).mean()),
                             "phase": _circ_mean(ph[:, i0, inside])})
    seg = C.WELCH_SEG_WAVELET
    sxx, syy, sxy, k = SP.cross_spectra(x, y, seg, seg // 2)
    fb = SP.freqs(seg)
    ib = int(np.argmin(np.abs(fb - C.F0)))
    welch = {"seg": seg, "k": k, "f": float(fb[ib]), "coh": float(SP.coherence(sxx, syy, sxy)[:, ib].mean()), "phase": _circ_mean(np.angle(sxy[:, ib])), "share": (end - start) / n}
    aw, bw = S.noise(rng, runs, n), S.noise(rng, runs, n)
    r2, _ph = W.coherence_from(W.cwt(aw, fr), W.cwt(bw, fr), fr, 1.0, 7)
    freq_rows = []
    for fi in freqs_check:
        j = int(np.argmin(np.abs(fr - fi)))
        freq_rows.append({"f": float(fr[j]), "mean": float(r2[:, j, 200:800].mean())})
    return {"runs": runs, "n": n, "f0": float(fr[i0]), "null": null, "coupled": coupled_rows, "welch": welch, "freq": freq_rows, "true_in": 1.0 / (1.0 + amp ** 2) ** 2, "start": start, "end": end}


# --- Verdikte (reine Funktionen: (Art, Text)) ----------------------------------------------------------------------------------------------


def verdict_main(a: Analysis) -> tuple[str, str]:
    s = a.settings
    if not s.welch:
        gap = a.w_in - a.w_out
        text = (f"Wavelet-Kohärenz bei f = {de(C.F0, 2)} (Glättung Zeit {de(s.tw, 2)} Skalen, Skala {s.ns}): innen {de(a.w_in, 2)} (wahr {de(a.w_true, 2)}), außen {de(a.w_out, 2)} (wahr 0); "
                f"Phase innen {de(a.w_phase_in, 2)} rad (gesetzt {de(C.PHASE0, 2)}). Welch über die ganze Aufzeichnung: {de(a.welch_coh, 2)}.")
        if gap > 0.15:
            return "success", text + " Die Kopplung ist als Intervall zu erkennen; die Werte liegen durch die Glättung über der wahren Kohärenz."
        return "warning", text + " Der Unterschied zwischen innen und außen ist klein: Die Glättung verwischt ihn oder die Kopplung ist zu schwach."
    base = f"K = {a.k_segments} Segment{'' if a.k_segments == 1 else 'e'} (äquivalent {de(a.k_eff, 1)}): mittlere Kohärenz im Band {de(a.mean_coh, 2)}, wahr {de(a.mean_true, 2)}, erwartet {de(a.mean_expected, 2)}; Schwelle {de(a.thr, 2)}, {de(100 * a.share_above, 0)} % der Bins darüber."
    if a.k_segments == 1:
        return "warning", base + " Bei einem Segment ist die Kohärenz bei jeder Frequenz 1, auch bei unabhängigen Reihen: kein Maß."
    extra = ""
    if s.scenario == "Verzögerung":
        extra = " Verzögerung gesetzt " + de(s.delay, 2) + ", geschätzt " + ", ".join(f"{D.METHOD_LABELS[m]} {de(v, 3)}" for m, v in a.delays.items()) + "."
    else:
        extra = f" Betrag von H1 {de(100 * a.rel_h1, 0)} %, von H2 {de(100 * a.rel_h2, 0)} % gegen die wahre Übertragung."
    if a.mean_coh - a.mean_true > 0.1:
        return "warning", base + " Die Kohärenz ist nach oben verzerrt (wenige Segmente)." + extra
    if s.scenario == "Verzögerung" and s.delay > 0 and a.mean_true - a.mean_coh > 0.1:
        rho2 = T.window_correlation(SP.hann(s.seg), s.delay) ** 2
        return "warning", base + f" Die Kohärenz ist niedriger als die wahre: Das Fenster ist gegen die um {de(s.delay, 1)} verzögerte Reihe verschoben (ρ(d)² = {de(rho2, 2)}); vor dem Schätzen ausrichten." + extra
    return "success", base + extra


def verdict_bias(res: dict) -> tuple[str, str]:
    rows = {(r["k"], r["g2"]): r for r in res["rows"]}
    ks = sorted({r["k"] for r in res["rows"]})
    small, big = ks[1], ks[-3]
    thr = {r["k"]: r for r in res["thr"]}
    kt = sorted(thr)[2]
    lo, hi = min(r["share"] for r in res["thr"]), max(r["share"] for r in res["thr"])
    return "warning", (f"Bei wahrer Kohärenz 0 ist der Mittelwert der Schätzung genau 1/K ({de(rows[(small, 0.0)]['sim'], 3)} bei K = {small}, {de(rows[(big, 0.0)]['sim'], 3)} bei K = {big}); bei wahrer Kohärenz 0,5 und K = {ks[0]} "
                       f"liegt er bei {de(rows[(ks[0], 0.5)]['sim'], 2)}. Die erste Ordnung g² + (1 − g²)²/K trifft ab K = {big} ({de(rows[(big, 0.25)]['sim'], 3)} gegen {de(rows[(big, 0.25)]['first'], 3)} bei g² = 0,25). "
                       f"Die 95-%-Schwelle 1 − 0,05^(1/(K − 1)) ({de(thr[kt]['thr'], 2)} bei K = {kt}) wird bei Kopplung null von {de(100 * lo, 1)} % bis {de(100 * hi, 1)} % der Bins überschritten.")


def verdict_phase(res: dict) -> tuple[str, str]:
    rows = {(r["k"], r["g2"]): r for r in res["phase"]}
    ks = sorted({r["k"] for r in res["phase"]})
    g2s = sorted({r["g2"] for r in res["phase"]})
    small, big = rows[(ks[0], g2s[0])], rows[(ks[-2], g2s[1])]
    last = res["delay"][4]
    return "info", (f"Die Phase ist unverzerrt; ihre Streuung folgt sqrt((1 − g²)/(2 K g²)) ab K = {ks[-2]} ({de(big['std'], 3)} gegen {de(big['formula'], 3)} bei g² = {de(g2s[1], 2)}), bei K = {ks[0]} und g² = {de(g2s[0], 2)} "
                    f"liegt sie darüber ({de(small['std'], 2)} gegen {de(small['formula'], 2)}). Verzögerung {de(res['d'], 1)}: bei g² = {de(last['g2'], 1)} und K = {last['k']} sind Kreuzkorrelation, nur Phase und kohärenzgewichtet auf "
                    f"{de(last['CC'], 4)}, {de(last['PHAT'], 4)} und {de(last['ML'], 4)} genau (RMSE, Abtastwerte); das Gitter ist {de(C.DELAY_STEP, 3)} fein.")


def verdict_transfer(res: dict) -> tuple[str, str]:
    rows = {(r["out"], r["r"]): r for r in res["rows"]}
    outs = sorted({r["out"] for r in res["rows"]})
    ins = sorted({r["r"] for r in res["rows"]})
    a, b = rows[(outs[0], ins[-1])], rows[(outs[0], ins[0])]
    c = rows[(outs[-1], ins[0])]
    return "warning", (f"Eingangsrauschen {de(ins[-1], 2)} drückt |H1| um {de(-100 * a['h1'], 0)} % (Näherung 1/(1 + r) − 1: {de(100 * a['h1_pred'], 0)} %); |H2| liegt bei Ausgangsrauschen {de(outs[0], 1)} {de(100 * b['h2'], 0)} % und bei {de(outs[-1], 1)} "
                       f"{de(100 * c['h2'], 0)} % über dem wahren Betrag. Das geometrische Mittel gleicht das nicht aus ({de(100 * a['geo'], 0)} % bis {de(100 * c['geo'], 0)} %). Die Phase ist in beiden unverzerrt ({de(a['phase1'], 3)} rad).")


def verdict_delay(res: dict) -> tuple[str, str]:
    rows = {r["d"]: r for r in res["rows"]}
    ds = sorted(rows)
    mid, far = rows[ds[5]], rows[ds[-1]]
    return "warning", (f"Bei Segmentlänge {res['seg']} fällt die Kohärenz einer reinen Verzögerung mit dem Fenster: {de(mid['coh'], 3)} bei d = {mid['d']} (ρ(d)² = {de(mid['rho2'], 3)}), {de(far['coh'], 3)} bei d = {far['d']} "
                       f"(ρ(d)² = {de(far['rho2'], 3)}). Nach dem Zurückschieben um d ist sie wieder {de(mid['aligned'], 3)}.")


def verdict_wavelet(res: dict) -> tuple[str, str]:
    null = {(r["tw"], r["ns"]): r for r in res["null"]}
    cp = {(r["tw"], r["ns"]): r for r in res["coupled"]}
    w = res["welch"]
    n0, n1 = null[(0.0, 1)], null[(1.0, 1)]
    c1 = cp[(1.0, 1)]
    return "warning", (f"Ohne Glättung ist die Wavelet-Kohärenz {de(n0['mean'], 3)} bei Kopplung null; mit Glättung (Zeit 1 Skala) {de(n1['mean'], 2)} (K_eff {de(n1['k_eff'], 1)}). Bei zeitweiliger Kopplung (wahr {de(res['true_in'], 2)}) liegt sie innen bei "
                       f"{de(c1['inside'], 2)} und außen bei {de(c1['outside'], 2)}, die Phase innen bei {de(c1['phase'], 2)} rad (gesetzt {de(C.PHASE0, 2)}); Welch über die ganze Aufzeichnung sieht {de(w['coh'], 2)} (Anteil der Kopplungszeit {de(w['share'], 2)}).")
