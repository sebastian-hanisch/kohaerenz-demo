"""Plotly-Figuren. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import koh_constants as C
import koh_delay as D
from koh_evaluation import Analysis, band_slice, wavelet_windows


def _fixed(fig: go.Figure, height: int = 340, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_signals(a: Analysis) -> go.Figure:
    """Die beiden Reihen (höchstens die ersten 600 Abtastwerte); bei der zeitweiligen Kopplung das Kopplungsintervall markiert."""
    n = min(len(a.x), 600)
    t = np.arange(n)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("x (Eingang)", "y (Ausgang)"))
    fig.add_trace(go.Scatter(x=t, y=a.x[:n], mode="lines", line=dict(color=C.COLOR_BLUE, width=1), name="x", hovertemplate="t = %{x}: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=a.y[:n], mode="lines", line=dict(color=C.COLOR_EST, width=1), name="y", hovertemplate="t = %{x}: %{y:.2f}<extra></extra>"), row=2, col=1)
    if not a.settings.welch:
        fig.add_vrect(x0=a.settings.start, x1=a.settings.end, fillcolor=C.COLOR_AUX, opacity=0.12, line_width=0)
    fig.update_xaxes(title_text="Zeit (Abtastwerte)", row=2, col=1)
    fig.update_layout(showlegend=False)
    return _fixed(fig, height=320)


def build_coherence(a: Analysis) -> go.Figure:
    """Geschätzte Kohärenz über der Frequenz gegen die wahre, die erwartete (Bias) und die 95-%-Schwelle."""
    f = a.freqs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=a.coh, mode="lines", line=dict(color=C.COLOR_EST, width=1.6), name="geschätzt", hovertemplate="f = %{x:.3f}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=f, y=a.true_coh, mode="lines", line=dict(color=C.COLOR_TRUE, width=1.6, dash="dash"), name="wahr", hovertemplate="f = %{x:.3f}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=f, y=a.expected, mode="lines", line=dict(color=C.COLOR_AUX, width=1.4, dash="dot"), name="erwartet (mit Bias)", hovertemplate="f = %{x:.3f}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=a.thr, line=dict(color="black", width=1, dash="dot"), annotation_text="95-%-Schwelle bei Kopplung null", annotation_position="top left")
    fig.update_xaxes(title_text="Frequenz (Zyklen je Abtastwert)", range=[0, 0.5])
    fig.update_yaxes(title_text="Kohärenz", range=[0, 1.02])
    return _fixed(fig, height=320)


def build_phase(a: Analysis) -> go.Figure:
    """Phase von S_xy gegen die wahre Phase (rad); Punkte oberhalb der Schwelle kräftig."""
    f = a.freqs
    sig = a.coh > a.thr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=a.true_phase, mode="lines", line=dict(color=C.COLOR_TRUE, width=1.4, dash="dash"), name="wahre Phase", hovertemplate="f = %{x:.3f}: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=f[~sig], y=a.phase[~sig], mode="markers", marker=dict(color=C.COLOR_NOISY, size=5), name="unter der Schwelle", hovertemplate="f = %{x:.3f}: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=f[sig], y=a.phase[sig], mode="markers", marker=dict(color=C.COLOR_EST, size=6), name="über der Schwelle", hovertemplate="f = %{x:.3f}: %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Frequenz (Zyklen je Abtastwert)", range=[0, 0.5])
    fig.update_yaxes(title_text="Phase (rad)", range=[-3.3, 3.3])
    return _fixed(fig, height=300)


def build_transfer(a: Analysis) -> go.Figure | None:
    """|H1|, |H2| und der wahre Betrag |H| (nur beim FIR-System)."""
    if a.h_true is None:
        return None
    f = a.freqs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=np.abs(a.h_true), mode="lines", line=dict(color=C.COLOR_TRUE, width=1.8, dash="dash"), name="|H| wahr", hovertemplate="f = %{x:.3f}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=f, y=np.abs(a.h1), mode="lines", line=dict(color=C.COLOR_BLUE, width=1.4), name="|H1| = |S_xy / S_xx|", hovertemplate="f = %{x:.3f}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=f, y=np.abs(a.h2), mode="lines", line=dict(color=C.COLOR_AUX, width=1.4), name="|H2| = |S_yy / S_yx|", hovertemplate="f = %{x:.3f}: %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Frequenz (Zyklen je Abtastwert)", range=[0, 0.5])
    fig.update_yaxes(title_text="Betrag der Übertragungsfunktion", rangemode="tozero")
    return _fixed(fig, height=300)


def build_wavelet(a: Analysis) -> go.Figure:
    """Wavelet-Kohärenz als Bild (Zeit × Frequenz, logarithmische Frequenz) mit Kopplungsintervall und f0."""
    s = a.settings
    t = np.arange(a.wcoh.shape[1])
    fig = go.Figure(go.Heatmap(x=t, y=a.wfreqs, z=a.wcoh, colorscale="Viridis", zmin=0.0, zmax=1.0, colorbar=dict(title="R²", thickness=12), hovertemplate="t = %{x}, f = %{y:.3f}: %{z:.2f}<extra></extra>"))
    fig.add_vline(x=s.start, line=dict(color=C.COLOR_AUX, width=1.5, dash="dash"))
    fig.add_vline(x=s.end, line=dict(color=C.COLOR_AUX, width=1.5, dash="dash"))
    fig.add_hline(y=C.F0, line=dict(color="white", width=1, dash="dot"))
    fig.update_xaxes(title_text="Zeit (Abtastwerte); orange gestrichelt: Kopplungsintervall")
    fig.update_yaxes(title_text="Frequenz (Zyklen je Abtastwert)", type="log")
    return _fixed(fig, height=380)


def build_wavelet_row(a: Analysis) -> go.Figure:
    """Wavelet-Kohärenz und Phase bei f0 über der Zeit, dazu die Kohärenz von Welch über die ganze Aufzeichnung."""
    s = a.settings
    t = np.arange(a.wcoh.shape[1])
    inside, _outs = wavelet_windows(len(t), s.start, s.end)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Kohärenz bei f₀", "Phase bei f₀ (rad)"))
    fig.add_trace(go.Scatter(x=t, y=a.wcoh[a.i0], mode="lines", line=dict(color=C.COLOR_EST, width=1.6), name="Wavelet-Kohärenz", hovertemplate="t = %{x}: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=a.welch_coh, line=dict(color=C.COLOR_NOISY, width=1.4, dash="dash"), annotation_text="Welch, ganze Aufzeichnung", annotation_position="top left", row=1, col=1)
    fig.add_hline(y=a.w_true, line=dict(color=C.COLOR_TRUE, width=1.2, dash="dot"), annotation_text="wahr im Intervall", annotation_position="bottom right", row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=a.wphase[a.i0], mode="lines", line=dict(color=C.COLOR_BLUE, width=1.2), name="Phase", hovertemplate="t = %{x}: %{y:.2f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=C.PHASE0, line=dict(color=C.COLOR_TRUE, width=1.2, dash="dot"), row=2, col=1)
    fig.add_vrect(x0=inside.start, x1=inside.stop, fillcolor=C.COLOR_AUX, opacity=0.12, line_width=0)
    fig.update_xaxes(title_text="Zeit (Abtastwerte); orange: Fenster der Kennzahl innen", row=2, col=1)
    fig.update_yaxes(range=[0, 1.02], row=1, col=1)
    fig.update_yaxes(range=[-3.3, 3.3], row=2, col=1)
    fig.update_layout(showlegend=False)
    return _fixed(fig, height=420)


# --- Experimente ------------------------------------------------------------------------------------------------------------------------------


def build_bias(res: dict) -> go.Figure:
    fig = go.Figure()
    g2s = sorted({r["g2"] for r in res["rows"]})
    for g2 in g2s:
        rows = [r for r in res["rows"] if r["g2"] == g2]
        fig.add_trace(go.Scatter(x=[r["k"] for r in rows], y=[r["sim"] for r in rows], mode="lines+markers", name=f"wahr {g2:g}".replace(".", ","), hovertemplate=f"wahr {g2:g}, K = %{{x}}: %{{y:.3f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[r["k"] for r in rows], y=[r["exact"] for r in rows], mode="lines", line=dict(color="black", width=0.8, dash="dot"), showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(title_text="Zahl K der Segmente (log); gepunktet: exakte Erwartung", type="log")
    fig.update_yaxes(title_text="mittlere geschätzte Kohärenz", range=[0, 1.02])
    return _fixed(fig, height=340)


def build_phase_exp(res: dict) -> go.Figure:
    rows = res["phase"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["formula"] for r in rows], y=[r["std"] for r in rows], mode="markers", marker=dict(color=C.COLOR_EST, size=9), text=[f"K = {r['k']}, g² = {r['g2']:g}" for r in rows],
                             hovertemplate="%{text}: Formel %{x:.3f}, gemessen %{y:.3f}<extra></extra>", name="gemessen"))
    top = max(max(r["formula"], r["std"]) for r in rows) * 1.05
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color="black", width=1, dash="dot"), name="Gleichheit", hoverinfo="skip"))
    fig.update_xaxes(title_text="Formel sqrt((1 − g²) / (2 K g²)) (rad)")
    fig.update_yaxes(title_text="gemessene Streuung der Phase (rad)")
    return _fixed(fig, height=320)


def build_delay_exp(res: dict) -> go.Figure:
    rows = res["delay"]
    labels = [f"K = {r['k']}, g² = {r['g2']:g}".replace(".", ",") for r in rows]
    fig = go.Figure()
    for meth in D.METHODS:
        fig.add_trace(go.Bar(x=labels, y=[r[meth] for r in rows], name=D.METHOD_LABELS[meth], hovertemplate=D.METHOD_LABELS[meth] + ", %{x}: %{y:.4f}<extra></extra>"))
    fig.update_yaxes(title_text="RMSE der Verzögerung (Abtastwerte, log)", type="log")
    fig.update_layout(barmode="group")
    return _fixed(fig, height=340)


def build_transfer_exp(res: dict) -> go.Figure:
    rows = res["rows"]
    labels = [f"N = {r['out']:g}, r = {r['r']:g}".replace(".", ",") for r in rows]
    fig = go.Figure()
    for key, label in (("h1", "|H1|"), ("h2", "|H2|"), ("geo", "geometrisches Mittel")):
        fig.add_trace(go.Bar(x=labels, y=[100 * r[key] for r in rows], name=label, hovertemplate=label + ", %{x}: %{y:.0f} %<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.update_yaxes(title_text="Abweichung vom wahren Betrag (%)")
    fig.update_layout(barmode="group")
    return _fixed(fig, height=340)


def build_window_exp(res: dict) -> go.Figure:
    rows = res["rows"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["d"] for r in rows], y=[r["coh"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_EST), name="gemessen", hovertemplate="d = %{x}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[r["d"] for r in rows], y=[r["rho2"] for r in rows], mode="lines", line=dict(color=C.COLOR_TRUE, dash="dash"), name="ρ(d)²", hovertemplate="d = %{x}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[r["d"] for r in rows], y=[r["aligned"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_BLUE), name="nach dem Ausrichten", hovertemplate="d = %{x}: %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Verzögerung d (Abtastwerte)")
    fig.update_yaxes(title_text="mittlere Kohärenz", range=[0, 1.05])
    return _fixed(fig, height=320)


def build_wavelet_exp(res: dict) -> go.Figure:
    rows = res["null"]
    labels = [f"Zeit {r['tw']:g}, Skala {r['ns']}".replace(".", ",") for r in rows]
    fig = go.Figure(go.Bar(x=labels, y=[r["mean"] for r in rows], marker_color=C.COLOR_BLUE, hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Glättung (Gauß-Std der Zeit in Skalen, Kasten über Skala)")
    fig.update_yaxes(title_text="mittlere Kohärenz bei Kopplung null", range=[0, 1.05])
    return _fixed(fig, height=300)


def build_wavelet_coupled(res: dict) -> go.Figure:
    rows = res["coupled"]
    labels = [f"Zeit {r['tw']:g}, Skala {r['ns']}".replace(".", ",") for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["inside"] for r in rows], name="innen", marker_color=C.COLOR_EST, hovertemplate="innen, %{x}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(x=labels, y=[r["outside"] for r in rows], name="außen", marker_color=C.COLOR_NOISY, hovertemplate="außen, %{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=res["welch"]["coh"], line=dict(color="black", width=1, dash="dot"), annotation_text="Welch, ganze Aufzeichnung", annotation_position="top left")
    fig.add_hline(y=res["true_in"], line=dict(color=C.COLOR_TRUE, width=1.2, dash="dash"), annotation_text="wahr im Intervall", annotation_position="top right")
    fig.update_yaxes(title_text="Wavelet-Kohärenz bei f₀", range=[0, 1.05])
    fig.update_layout(barmode="group")
    return _fixed(fig, height=320)
