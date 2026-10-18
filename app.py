"""Kohärenz: Was zwei Signale verbindet - bei kurzen Daten ist fast alles kohärent, und Eingangsrauschen kippt die Übertragungsfunktion - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 15 der Reihe „Spektral- und Wavelet-Analyse“. Kreuzspektrum und Kohärenz zweier Reihen nach Welch (Bias, Schwelle, Phase, Verzögerung, Übertragungsfunktion H1 und H2) und die Wavelet-Kohärenz für eine
zeitweilige Kopplung; fünf Experimente gegen bekannte wahre Werte. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import koh_constants as C
import koh_delay as D
import koh_evaluation as ev
from koh_evaluation import Settings, analyse, de
from koh_pdf_export import build_pdf
from koh_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from koh_visualization import (
    build_bias,
    build_coherence,
    build_delay_exp,
    build_phase,
    build_phase_exp,
    build_signals,
    build_transfer,
    build_transfer_exp,
    build_wavelet,
    build_wavelet_coupled,
    build_wavelet_exp,
    build_wavelet_row,
    build_window_exp,
)

st.set_page_config(page_title="Kohärenz – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Rechne die Experimente …")
def _experiment(name):
    return getattr(ev, "sweep_" + name)()


st.title("🔗 Kohärenz: Was zwei Signale verbindet")
st.markdown(
    """
Das **Kreuzspektrum** $S_{xy} = E[\\overline{X}\\,Y]$ zweier Reihen sagt, bei welchen Frequenzen sie zusammenhängen: sein Betrag (genauer die **Kohärenz** $\\gamma^2 = |S_{xy}|^2/(S_{xx}S_{yy})$, zwischen 0 und 1) die Stärke,
seine **Phase** die Verzögerung, der Quotient $S_{xy}/S_{xx}$ die Übertragungsfunktion. Aber die Kohärenz aus wenigen Segmenten ist **nach oben verzerrt**: Bei einem Segment ist sie immer 1. Eingangsrauschen kippt die
Übertragungsfunktion, eine Verzögerung kostet Kohärenz durch das Fenster, und die **Wavelet-Kohärenz** findet eine zeitweilige Kopplung, die eine Kohärenz über die ganze Aufzeichnung verdünnt.

Dies ist die fünfzehnte Demo der Reihe „Spektral- und Wavelet-Analyse“, auf das **Welch-Verfahren** und das **Skalogramm** aufbauend. Alles ist **simuliert**: Kopplung, Verzögerung und Übertragungsfunktion sind bekannt,
die Schätzungen werden daran gemessen. Weiter unten: **🔬 Experimente**, **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So entsteht die Kohärenz", expanded=True):
    st.markdown(
        """
1. **Segmente:** Beide Reihen werden in $K$ Segmente (Hann-Fenster) geteilt, jedes Segment fouriertransformiert zu $X_j, Y_j$.
2. **Kreuzspektrum:** $S_{xx} = \\overline{|X_j|^2}$, $S_{yy} = \\overline{|Y_j|^2}$, $S_{xy} = \\overline{\\overline{X_j}\\,Y_j}$ (Mittel über die Segmente). Ohne Mitteln ($K = 1$) ist $\\gamma^2 = 1$ bei jeder Frequenz.
3. **Phase und Verzögerung:** $\\arg S_{xy} = -2\\pi f d$ für $y[t] = x[t - d]$; die Steigung der Phase gibt $d$.
4. **Übertragungsfunktion:** $H_1 = S_{xy}/S_{xx}$ (Rauschen am Ausgang), $H_2 = S_{yy}/S_{yx}$ (Rauschen am Eingang).
5. **Wavelet-Kohärenz:** dasselbe mit Morlet-Transformierten, geglättet in Zeit und Skala; ohne Glättung ebenfalls 1.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()
ss = st.session_state

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Szenario**")
    scenario = st.selectbox("Szenario", list(C.SCENARIOS), key="scenario_select",
                            help="Verzögerung: y ist x um d Abtastwerte verzögert plus Rauschen (wahre Kohärenz einstellbar). FIR-System: y ist x durch ein Filter plus Ausgangsrauschen, x mit Eingangsrauschen beobachtet. "
                                 "Zeitweilige Kopplung: zwei Bandrauschen bei 0,05, nur im Intervall gekoppelt (Wavelet-Kohärenz).")
    welch = scenario in C.WELCH_SCENARIOS
    if welch:
        seg = st.select_slider("Segmentlänge", options=list(C.SEG_CHOICES), value=ss["seg_kept"], help="Länge der Hann-gefensterten Segmente.")
        k = st.slider("Zahl der Segmente K", C.K_MIN, C.K_MAX, int(ss["k_kept"]), help="Mit K = 1 ist die Kohärenz immer 1; die Länge der Aufzeichnung folgt aus K, Segmentlänge und Überlappung.")
        overlap = st.radio("Überlappung", list(C.OVERLAPS), index=list(C.OVERLAPS).index(ss["overlap_kept"]), horizontal=True, help="Überlappende Segmente sind nicht unabhängig: Die äquivalente Segmentzahl ist kleiner als K.")
        ss["seg_kept"], ss["k_kept"], ss["overlap_kept"] = int(seg), int(k), overlap
    else:
        seg, k, overlap = int(ss["seg_kept"]), int(ss["k_kept"]), ss["overlap_kept"]
    if scenario == "Verzögerung":
        g2 = st.slider("Wahre Kohärenz", C.G2_MIN, C.G2_MAX, float(ss["g2_kept"]), step=C.G2_STEP, help="1 / (1 + Rauschleistung) bei Eingangsleistung 1; 0 heißt: y ist unabhängig von x.")
        delay = st.slider("Verzögerung d (Abtastwerte)", C.D_MIN, C.D_MAX, float(ss["delay_kept"]), step=C.D_STEP, help="y[t] = x[t − d], auch gebrochen (zirkulär per FFT).")
        ss["g2_kept"], ss["delay_kept"] = float(g2), float(delay)
    else:
        g2, delay = float(ss["g2_kept"]), float(ss["delay_kept"])
    if scenario == "FIR-System":
        out = st.slider("Ausgangsrauschen", C.OUT_MIN, C.OUT_MAX, float(ss["out_kept"]), step=C.OUT_STEP, help="Rauschleistung am Ausgang gegen die Eingangsleistung 1.")
        inn = st.slider("Eingangsrauschen r", C.IN_MIN, C.IN_MAX, float(ss["inn_kept"]), step=C.IN_STEP, help="Rauschleistung, mit der x beobachtet wird, gegen die Eingangsleistung 1.")
        ss["out_kept"], ss["inn_kept"] = float(out), float(inn)
    else:
        out, inn = float(ss["out_kept"]), float(ss["inn_kept"])
    if scenario == "Zeitweilige Kopplung":
        start = st.slider("Beginn der Kopplung", C.INTERVAL_MIN, C.INTERVAL_MAX, int(ss["start_kept"]), help="Abtastwert, ab dem y das gemeinsame Band enthält.")
        end = st.slider("Ende der Kopplung", C.INTERVAL_MIN, C.INTERVAL_MAX, int(ss["end_kept"]), help="Abtastwert, bis zu dem y das gemeinsame Band enthält.")
        amp = st.slider("Unabhängiges Rauschen", C.AMP_MIN, C.AMP_MAX, float(ss["amp_kept"]), step=C.AMP_STEP, help="Amplitude des unabhängigen Bandrauschens in x und y gegen das gemeinsame Band; die wahre Kohärenz im Intervall ist 1/(1 + Amplitude²)².")
        tw = st.slider("Glättung in der Zeit", C.TW_MIN, C.TW_MAX, float(ss["tw_kept"]), step=C.TW_STEP, help="Gauß-Standardabweichung in Skalen des Wavelets; 0 heißt keine Glättung (Kohärenz dann überall 1).")
        ns = st.select_slider("Glättung in der Skala", options=list(C.NS_CHOICES), value=ss["ns_kept"], help="Zahl benachbarter Frequenzen im Kasten (12 je Oktave).")
        ss["start_kept"], ss["end_kept"], ss["amp_kept"], ss["tw_kept"], ss["ns_kept"] = int(start), int(end), float(amp), float(tw), int(ns)
    else:
        start, end, amp, tw, ns = int(ss["start_kept"]), int(ss["end_kept"]), float(ss["amp_kept"]), float(ss["tw_kept"]), int(ss["ns_kept"])
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"scenario_select": scenario, "seg_kept": int(seg), "k_kept": int(k), "overlap_kept": overlap, "g2_kept": float(g2), "delay_kept": float(delay), "out_kept": float(out), "inn_kept": float(inn),
                   "start_kept": int(start), "end_kept": int(end), "amp_kept": float(amp), "tw_kept": float(tw), "ns_kept": int(ns), "seed_input": int(seed)})

if end <= start + 40:
    st.warning("Das Kopplungsintervall ist zu kurz (mindestens 40 Abtastwerte): Ende wird auf Beginn + 40 gesetzt.")
    end = min(C.INTERVAL_MAX + 40, start + 40)
settings = Settings(scenario, int(seg), int(k), overlap, float(g2), float(delay), float(out), float(inn), int(start), int(end), float(amp), float(tw), int(ns), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
if welch:
    metrics = [("Mittlere Kohärenz", de(a.mean_coh, 2), f"wahr {de(a.mean_true, 2)}, erwartet {de(a.mean_expected, 2)}",
                "Mittel über die Bins 3 bis seg/2 − 3; „erwartet“ ist der Mittelwert der Schätzung bei dieser Zahl unabhängiger Segmente (exakt, mit Bias)."),
               ("Schwelle 95 %", de(a.thr, 2), f"{de(100 * a.share_above, 0)} % der Bins darüber", "Kohärenz, die bei wahrer Kohärenz 0 nur in 5 % der Fälle überschritten wird: 1 − 0,05^(1/(K − 1))."),
               ("Segmente K", f"{a.k_segments}", f"äquivalent {de(a.k_eff, 1)}", "Zahl der Segmente und die äquivalente Zahl unabhängiger Segmente bei Überlappung.")]
    metrics.append(("Phasenfehler", de(a.phase_rmse, 3) if a.phase_rmse is not None else "–", "rad, über der Schwelle", "Quadratischer Mittelwert der Phasenabweichung von der wahren Phase an den Bins über der Schwelle."))
    if a.delays is not None:
        metrics.append(("Verzögerung geschätzt", de(a.delays["ML"], 3), f"gesetzt {de(settings.delay, 2)}", "Kohärenzgewichteter Schätzer aus dem Kreuzspektrum (Gitter 0,005); die anderen Schätzer stehen im Ergebnistext."))
    if a.rel_h1 is not None:
        metrics.append(("|H1| gegen wahr", f"{de(100 * a.rel_h1, 0)} %", f"|H2| {de(100 * a.rel_h2, 0)} %", "Mittlere relative Abweichung des Betrags der Schätzung vom wahren Betrag (Bins 3 bis seg/2 − 3)."))
else:
    metrics = [("Kohärenz innen", de(a.w_in, 2), f"wahr {de(a.w_true, 2)}", "Wavelet-Kohärenz bei f = 0,05, gemittelt im Inneren des Kopplungsintervalls (mit Rand)."),
               ("Kohärenz außen", de(a.w_out, 2), "wahr 0", "Dasselbe außerhalb des Intervalls (mit Rand)."),
               ("Welch, ganze Aufzeichnung", de(a.welch_coh, 2), f"Phase {de(a.welch_phase, 2)} rad", "Kohärenz bei der Frequenz nächst 0,05 über die ganze Aufzeichnung (Segmentlänge 128, 50 % Überlappung)."),
               ("Phase innen", de(a.w_phase_in, 2), f"gesetzt {de(C.PHASE0, 2)} rad", "Mittlere Phase der Wavelet-Kohärenz im Inneren des Intervalls.")]
for col, (label, value, delta, help_text) in zip(st.columns(len(metrics)), metrics):
    col.metric(label, value, delta=delta, delta_color="off", help=help_text)
st.plotly_chart(build_signals(a), width="stretch", key="signals")
if welch:
    st.plotly_chart(build_coherence(a), width="stretch", key="coherence")
    st.caption("Geschätzte Kohärenz (grün) gegen die wahre (rot gestrichelt) und die erwartete Schätzung mit Bias (orange gepunktet); waagerecht die 95-%-Schwelle: Darunter ist Kopplung nicht von null zu unterscheiden.")
    st.plotly_chart(build_phase(a), width="stretch", key="phase")
    st.caption("Phase von S_xy gegen die wahre; unterhalb der Schwelle ist sie Zufall.")
    fig_h = build_transfer(a)
    if fig_h is not None:
        st.plotly_chart(fig_h, width="stretch", key="transfer")
        st.caption("Betrag der Übertragungsfunktion: wahr (rot gestrichelt), H1 (blau) und H2 (orange). Mit Eingangsrauschen liegt H1 darunter und H2 darüber.")
else:
    st.plotly_chart(build_wavelet(a), width="stretch", key="wavelet")
    st.caption("Wavelet-Kohärenz R² (Morlet ω₀ = 6) über Zeit und Frequenz; orange gestrichelt das wahre Kopplungsintervall, weiß gepunktet f = 0,05. Der Rand der Aufzeichnung ist durch den Einflusskegel verfälscht, und außerhalb des Bandes 0,04 bis 0,06 stammen die Werte nur aus den Ausläufern des Wavelets.")
    st.plotly_chart(build_wavelet_row(a), width="stretch", key="wavelet_row")
    st.caption("Kohärenz und Phase bei f = 0,05 über der Zeit, dazu die Kohärenz von Welch über die ganze Aufzeichnung (grau gestrichelt) und die wahre Kohärenz im Intervall (rot gepunktet).")

if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = build_pdf(a)
    st.session_state["pdf_for"] = settings
if st.session_state.get("pdf_for") == settings and "pdf_bytes" in st.session_state:
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="kohaerenz.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie weit kann man der geschätzten Kohärenz trauen?")
kind_v, text = ev.verdict_main(a)
getattr(st, kind_v)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experimente")
st.caption("Feste Zufallsströme, unabhängig von den Reglern. Weißes Gauß-Rauschen als Eingang (Einheitsleistung), Hann-Fenster; „wahr“ ist der Wert, mit dem die Reihen gebaut sind.")

st.markdown("#### Kohärenzbias und Schwelle")
bias = _experiment("bias")
st.caption(f"Segmentlänge {bias['seg']}, {bias['runs']} Läufe, Bins 2 bis {bias['seg'] // 2 - 2}, Segmente ohne Überlappung (unabhängig). Mittel der geschätzten Kohärenz bei wahrer Kohärenz 0 bis 0,9: Simulation, erste Ordnung "
           "g² + (1 − g²)²/K und exakte Erwartung aus der Dichte.")
st.dataframe(pd.DataFrame([{"K": k, **{f"wahr {g:g}".replace(".", ","): f"{de(r['sim'], 3)} / {de(r['first'], 3)} / {de(r['exact'], 3)}" for g in sorted({x['g2'] for x in bias['rows']})
                                       for r in [next(x for x in bias["rows"] if x["k"] == k and x["g2"] == g)]}} for k in sorted({x["k"] for x in bias["rows"]})]), hide_index=True, width="stretch")
st.plotly_chart(build_bias(bias), width="stretch", key="exp_bias")
st.caption(f"95-%-Schwelle bei Kopplung null ({bias['thr_runs']} Läufe): 1 − 0,05^(1/(K − 1)) gegen den Anteil der Bins darüber.")
st.dataframe(pd.DataFrame([{"K": r["k"], "Schwelle": de(r["thr"], 3), "Anteil darüber": f"{de(100 * r['share'], 1)} %", "Mittel": de(r["mean"], 4), "1/K": de(r["inv_k"], 4)} for r in bias["thr"]]), hide_index=True, width="stretch")
st.caption(f"Überlappung (Hann, Segmentlänge {bias['seg']}, n = 4096, Kopplung null): Aus dem Mittel 1/K_eff geschätzte äquivalente Segmentzahl gegen die Formel aus der Segmentkorrelation.")
st.dataframe(pd.DataFrame([{"Überlappung": f"{de(100 * r['overlap'], 0)} %", "Segmente K": r["k"], "Mittel": de(r["mean"], 4), "K_eff gemessen": de(r["k_eff_measured"], 1), "K_eff Formel": de(r["k_eff_formula"], 1)} for r in bias["overlap"]]),
             hide_index=True, width="stretch")
kind_v, text = ev.verdict_bias(bias)
getattr(st, kind_v)(text)

st.markdown("#### Phase und Verzögerung")
ph = _experiment("phase")
st.caption(f"Streuung der Phase von S_xy (Segmentlänge {ph['seg']}, y um 1 Abtastwert verzögert, {ph['runs']} Läufe) gegen sqrt((1 − g²)/(2 K g²)).")
st.dataframe(pd.DataFrame([{"K": r["k"], "wahre Kohärenz": de(r["g2"], 2), "gemessen": de(r["std"], 3), "Formel": de(r["formula"], 3), "mittlere Phasenabweichung": de(r["mean"], 4)} for r in ph["phase"]]), hide_index=True, width="stretch")
st.plotly_chart(build_phase_exp(ph), width="stretch", key="exp_phase")
st.caption(f"Verzögerung d = {de(ph['d'], 1)} Abtastwerte (zirkulär, gebrochen), Segmentlänge {ph['delay_seg']}, 50 % Überlappung, {ph['delay_runs']} Läufe: RMSE der drei Schätzer (Abtastwerte); das Gitter ist {de(C.DELAY_STEP, 3)} fein.")
st.dataframe(pd.DataFrame([{"n": r["n"], "K": r["k"], "wahre Kohärenz": de(r["g2"], 2), **{D.METHOD_LABELS[m]: de(r[m], 4) for m in D.METHODS}} for r in ph["delay"]]), hide_index=True, width="stretch")
st.plotly_chart(build_delay_exp(ph), width="stretch", key="exp_delay_est")
kind_v, text = ev.verdict_phase(ph)
getattr(st, kind_v)(text)

st.markdown("#### Übertragungsfunktion bei Eingangsrauschen")
tr = _experiment("transfer")
st.caption(f"System h = (0; 0,6; 0,3; 0,1), Segmentlänge {tr['seg']}, K = {tr['k']}, {tr['runs']} Läufe; Betrag der wahren Übertragung bei den Bins 3, 10, 20, 30: {', '.join(de(v, 2) for v in tr['h_abs'])}. "
           "Mittlere relative Abweichung der geschätzten Beträge (Bins 3 bis 30) für Ausgangsrauschen N und Eingangsrauschen r gegen die Näherungen 1/(1 + r) − 1 für H1 und 1 + N/|H|² − 1 für H2.")
st.dataframe(pd.DataFrame([{"Ausgangsrauschen N": de(r["out"], 1), "Eingangsrauschen r": de(r["r"], 2), "H1": f"{de(100 * r['h1'], 1)} %", "Näherung H1": f"{de(100 * r['h1_pred'], 1)} %", "H2": f"{de(100 * r['h2'], 1)} %",
                            "Näherung H2": f"{de(100 * r['h2_pred'], 1)} %", "geometrisches Mittel": f"{de(100 * r['geo'], 1)} %", "Phase H1 (rad)": de(r["phase1"], 3)} for r in tr["rows"]]), hide_index=True, width="stretch")
st.plotly_chart(build_transfer_exp(tr), width="stretch", key="exp_transfer")
kind_v, text = ev.verdict_transfer(tr)
getattr(st, kind_v)(text)

st.markdown("#### Verzögerung gegen Segmentlänge")
dl = _experiment("delay")
st.caption(f"Reine Verzögerung d (fast ohne Rauschen), Segmentlänge {dl['seg']}, K = {dl['k']}, {dl['runs']} Läufe: mittlere Kohärenz (Bins 4 bis {dl['seg'] // 2 - 4}) gegen ρ(d)² mit ρ(d) = Σ w_m w_(m+d) / Σ w² "
           "und nach dem Zurückschieben um d.")
st.dataframe(pd.DataFrame([{"d": r["d"], "d / Segmentlänge": de(r["ratio"], 3), "gemessen": de(r["coh"], 4), "ρ(d)²": de(r["rho2"], 4), "nach dem Ausrichten": de(r["aligned"], 4)} for r in dl["rows"]]), hide_index=True, width="stretch")
st.plotly_chart(build_window_exp(dl), width="stretch", key="exp_window")
kind_v, text = ev.verdict_delay(dl)
getattr(st, kind_v)(text)

st.markdown("#### Wavelet-Kohärenz")
wv = _experiment("wavelet")
st.caption(f"Morlet ω₀ = 6, n = {wv['n']}, f₀ = {de(wv['f0'], 3)}, {wv['runs']} Läufe. Bei Kopplung null: Mittel der Wavelet-Kohärenz (Zeiten 200 bis 799) je Glättung (Gauß-Std der Zeit in Skalen, Kasten über Frequenzen), "
           "äquivalente Segmentzahl 1/Mittel, 95-%-Quantil gegen die Beta-Formel mit dieser Zahl.")
st.dataframe(pd.DataFrame([{"Glättung Zeit": de(r["tw"], 1), "Glättung Skala": r["ns"], "Mittel": de(r["mean"], 3), "K_eff": de(r["k_eff"], 2), "95-%-Quantil": de(r["q95"], 3), "Formel mit K_eff": de(r["formula"], 3)} for r in wv["null"]]),
             hide_index=True, width="stretch")
st.plotly_chart(build_wavelet_exp(wv), width="stretch", key="exp_wavelet_null")
st.caption(f"Zeitweilige Kopplung im Intervall [{wv['start']}, {wv['end']}) mit wahrer Kohärenz {de(wv['true_in'], 2)}: Wavelet-Kohärenz innen und außen und die Phase innen (gesetzt {de(C.PHASE0, 2)}); "
           f"Welch (Segmentlänge {wv['welch']['seg']}, 50 % Überlappung, K = {wv['welch']['k']}) über die ganze Aufzeichnung bei f = {de(wv['welch']['f'], 3)}: Kohärenz {de(wv['welch']['coh'], 2)}, Phase {de(wv['welch']['phase'], 2)} rad.")
st.dataframe(pd.DataFrame([{"Glättung Zeit": de(r["tw"], 1), "Glättung Skala": r["ns"], "innen": de(r["inside"], 3), "außen": de(r["outside"], 3), "Phase innen (rad)": de(r["phase"], 3)} for r in wv["coupled"]]),
             hide_index=True, width="stretch")
st.plotly_chart(build_wavelet_coupled(wv), width="stretch", key="exp_wavelet_coupled")
st.caption("Bei Kopplung null und weißem Rauschen hängt der Mittelwert der geglätteten Kohärenz nicht von der Frequenz ab (Glättung Zeit 1, Skala 7):")
st.dataframe(pd.DataFrame([{"f": de(r["f"], 4), "Mittel": de(r["mean"], 3)} for r in wv["freq"]]), hide_index=True, width="stretch")
kind_v, text = ev.verdict_wavelet(wv)
getattr(st, kind_v)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Genug unabhängige Segmente** | Mit wenigen Segmenten ist die Kohärenz nach oben verzerrt, bei einem Segment immer 1; die Schwelle 1 − 0,05^(1/(K − 1)) gilt nur für unabhängige Segmente. | mehr Segmente, Bias-Korrektur, simulierte Schwelle |
| **Das Rauschen steckt am Ausgang** | Mit Eingangsrauschen ist H1 zu klein, H2 zu groß; die Phase bleibt richtig. | Schätzung mit Fehlern in den Variablen, Messung des Rauschens |
| **Keine Verzögerung innerhalb des Fensters** | Eine Verzögerung kostet Kohärenz durch das Fenster (ρ(d)²), nicht durch das Signal. | Ausrichten vor dem Schätzen, längere Segmente |
| **Stationäre Kopplung** | Kopplung, die nur zeitweise besteht, wird über die ganze Aufzeichnung verdünnt. | Wavelet-Kohärenz, Kurzzeit-Kohärenz |
| **Wavelet-Kohärenz ist ein Maß ohne weiteres** | Ohne Glättung ist sie immer 1; mit Glättung liegt sie bei Kopplung null deutlich über 0, und die Beta-Schwelle passt nicht. | Schwelle per Simulation |
| **Kohärenz heißt Wirkung** | Ein gemeinsamer Treiber gibt dieselbe Kohärenz; Richtung und Partialkohärenz bleiben offen. | spektrale Kausalität (nicht gebaut) |
""",
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Kreuzspektrum.** Mit den Segment-Spektren $X_j, Y_j$ ($j = 1, \dots, K$, Hann-Fenster $w$, Skalierung $1/\sum w^2$) ist $S_{xy} = \tfrac1K \sum_j \overline{X_j} Y_j$, $S_{xx} = \tfrac1K \sum_j |X_j|^2$ und die Kohärenz
$\hat\gamma^2 = |S_{xy}|^2 / (S_{xx} S_{yy})$. Für $y[t] = x[t - d]$ ist $\arg S_{xy} = -2\pi f d$.

**Verteilung.** Bei $K$ unabhängigen komplexen Gauß-Segmenten und wahrer Kohärenz $\gamma^2$ hat $c = \hat\gamma^2$ die Dichte $p(c) = (K - 1)(1 - \gamma^2)^K (1 - c)^{K-2}\,{}_2F_1(K, K; 1; \gamma^2 c)$
(Carter, Knapp und Nuttall 1973); mit Eulers Umformung ist ${}_2F_1(K, K; 1; x) = (1 - x)^{1-2K} \sum_{m=0}^{K-1}\binom{K-1}{m}^2 x^m$. Für $\gamma^2 = 0$ ist $c \sim \mathrm{Beta}(1, K-1)$: $E[c] = 1/K$ und
$P(c > s) = (1 - s)^{K-1}$, also die 95-%-Schwelle $s = 1 - 0{,}05^{1/(K-1)}$. In erster Ordnung gilt $E[c] \approx \gamma^2 + (1 - \gamma^2)^2/K$. Für $K = 1$ ist $c \equiv 1$.

**Phase.** Für große $K$ ist die Streuung von $\arg \hat S_{xy}$ gleich $\sqrt{(1 - \gamma^2)/(2 K \gamma^2)}$.

**Übertragungsfunktion.** Bei Rauschen am Ausgang (Leistung $N$) ist $H_1 = S_{xy}/S_{xx} \to H$; bei Eingangsrauschen der Leistung $r S_{xx}$ ist $E|H_1| \approx |H|/(1 + r)$. $H_2 = S_{yy}/S_{yx}$ hat $|H_2| \approx |H|\,(1 + N/(|H|^2 S_{xx}))$.

**Verzögerung und Fenster.** Für eine reine Verzögerung $d$ ist die Kohärenz der Segmente $\rho(d)^2$ mit $\rho(d) = \sum_m w_m w_{m+d} / \sum_m w_m^2$ (Korrelation des Fensters mit seiner Verschiebung); nach dem Zurückschieben um $d$ ist sie 1.
**Verzögerungsschätzer:** $\hat d = \arg\max_d \operatorname{Re}\sum_k s_k e^{i2\pi f_k d}$ mit $s_k = S_{xy}$ (Kreuzkorrelation), $S_{xy}/|S_{xy}|$ (PHAT) oder $S_{xy}/|S_{xy}| \cdot \hat\gamma^2/(1 - \hat\gamma^2)$ (kohärenzgewichtet).

**Überlappung.** Der Streufaktor des Segmentmittels ist $1 + 2\sum_{j=1}^{K-1}(1 - j/K)\,\rho(j h)^2$ ($h$ Schrittweite), die äquivalente Zahl unabhängiger Segmente $K_{\mathrm{eff}} = K / (\text{Streufaktor})$.

**Wavelet-Kohärenz.** Mit der Morlet-Transformation $W(s, n) = \sum_m x[m]\,\overline{\psi((m - n)/s)}/\sqrt s$ und der Glättung $\mathcal S$ in Zeit (Gauß, Std $= t_w s$) und Skala (Kasten) ist
$R^2 = |\mathcal S(\overline{W_x} W_y)|^2 / (\mathcal S(|W_x|^2)\,\mathcal S(|W_y|^2))$; ohne $\mathcal S$ ist $R^2 = 1$.

**Quellen.** Lehrbuchwissen und Aufsätze (Carter, Knapp und Nuttall 1973; Knapp und Carter 1976; Bendat und Piersol; Torrence und Webster 1999; Percival und Walden 1993). Formeln hier per Handrechnung, Simulation und gegen die exakte
Verteilung geprüft; die Quellen selbst wurden nicht gelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
