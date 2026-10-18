"""Randfälle: alle Szenarien, Segmentlängen, Segmentzahlen, Rauschpegel, Verzögerungen, Abbildungen und PDF."""

import numpy as np
import pytest

import koh_constants as C
import koh_evaluation as ev
import koh_visualization as V
from koh_pdf_export import build_pdf


def _ok(a):
    s = a.settings
    if s.welch:
        return (np.isfinite(a.coh).all() and np.isfinite(a.phase).all() and np.isfinite(a.expected).all() and 0.0 <= a.share_above <= 1.0 and np.isfinite(a.mean_coh) and a.k_segments == s.k
                and (a.coh <= 1 + 1e-9).all())
    return np.isfinite(a.wcoh).all() and (a.wcoh <= 1 + 1e-9).all() and np.isfinite(a.w_in) and np.isfinite(a.welch_coh)


@pytest.mark.parametrize("seg", list(C.SEG_CHOICES))
def test_every_segment_length_with_one_few_and_many_segments_and_both_overlaps(seg):
    for scenario in C.WELCH_SCENARIOS:
        for k in (1, 2, C.K_MAX):
            for overlap in C.OVERLAPS:
                a = ev.analyse(ev.Settings(scenario=scenario, seg=seg, k=k, overlap=overlap))
                assert _ok(a) and a.freqs.shape == (seg // 2 + 1,)


def test_extreme_coherence_delay_and_noise_levels_run():
    for g2 in (C.G2_MIN, C.G2_MAX):
        for d in (C.D_MIN, C.D_MAX):
            a = ev.analyse(ev.Settings(seg=256, k=4, g2=g2, delay=d))
            assert _ok(a) and set(a.delays) == {"CC", "PHAT", "ML"} and all(abs(v) <= d + 8.0 for v in a.delays.values())
    for out in (C.OUT_MIN, C.OUT_MAX):
        for inn in (C.IN_MIN, C.IN_MAX):
            a = ev.analyse(ev.Settings(scenario="FIR-System", k=8, out=out, inn=inn))
            assert _ok(a) and np.isfinite(a.rel_h1) and np.isfinite(a.rel_h2)


def test_the_wavelet_scenario_at_the_extremes():
    for start, end in ((C.INTERVAL_MIN, C.INTERVAL_MIN + 40), (C.INTERVAL_MAX - 40, C.INTERVAL_MAX), (C.INTERVAL_MIN, C.INTERVAL_MAX)):
        for tw in (C.TW_MIN, C.TW_MAX):
            for ns in (C.NS_CHOICES[0], C.NS_CHOICES[-1]):
                for amp in (C.AMP_MIN, C.AMP_MAX):
                    a = ev.analyse(ev.Settings(scenario="Zeitweilige Kopplung", start=start, end=end, tw=tw, ns=ns, amp=amp))
                    assert _ok(a) and 0.0 <= a.w_true <= 1.0 and (np.isnan(a.w_out) or 0.0 <= a.w_out <= 1.0 + 1e-9)


def test_a_short_interval_with_no_outside_window_gives_nan_not_an_error():
    a = ev.analyse(ev.Settings(scenario="Zeitweilige Kopplung", start=C.INTERVAL_MIN, end=C.INTERVAL_MAX + 40))
    assert np.isnan(a.w_out) and np.isfinite(a.w_in) and ev.de(a.w_out) == "–"


@pytest.mark.parametrize("scenario", list(C.SCENARIOS))
def test_every_figure_builds_and_the_pdf_is_a_pdf(scenario):
    a = ev.analyse(ev.Settings(scenario=scenario, seg=64, k=8))
    figs = [V.build_signals(a)]
    if a.settings.welch:
        figs += [V.build_coherence(a), V.build_phase(a)]
        assert (V.build_transfer(a) is None) == (scenario != "FIR-System")
        if scenario == "FIR-System":
            figs.append(V.build_transfer(a))
    else:
        figs += [V.build_wavelet(a), V.build_wavelet_row(a)]
    assert all(f.data for f in figs)
    pdf = build_pdf(a)
    assert pdf[:5] == b"%PDF-" and len(pdf) > 700


def test_a_single_segment_figure_and_pdf_work():
    a = ev.analyse(ev.Settings(k=1))
    assert V.build_coherence(a).data and V.build_phase(a).data and build_pdf(a)[:5] == b"%PDF-"
