"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import koh_delay as D
from koh_evaluation import Analysis, de

_REPLACE = {"–": "-", "—": "-", "−": "-", "²": "2", "€": "EUR", "→": "->", "σ": "sigma", "Σ": "Summe", "±": "+-", "√": "Wurzel aus ", "×": "x", "é": "e", "„": '"', "“": '"', "π": "pi", "≥": ">=",
            "λ": "lambda", "ν": "nu", "γ": "gamma", "ρ": "rho", "₀": "0", "₁": "1", "₂": "2"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Kohärenz: Einstellung und Kennzahlen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    lines = []
    if s.welch:
        head = f"{s.scenario}, Segmentlänge {s.seg}, {a.k_segments} Segmente (Überlappung {s.overlap}, äquivalent {de(a.k_eff, 1)}), n = {s.n}, Hann, Seed {s.seed}."
        if s.scenario == "Verzögerung":
            head += f" Wahre Kohärenz {de(s.g2, 2)}, Verzögerung {de(s.delay, 2)} Abtastwerte."
        else:
            head += f" Ausgangsrauschen {de(s.out, 2)}, Eingangsrauschen {de(s.inn, 2)}."
        lines = [f"Mittlere Kohärenz im Band: {de(a.mean_coh, 3)} (wahr {de(a.mean_true, 3)}, erwartet {de(a.mean_expected, 3)}).",
                 f"95-%-Schwelle bei Kopplung null: {de(a.thr, 3)}; Anteil der Bins darüber: {de(100 * a.share_above, 1)} %."]
        if a.phase_rmse is not None:
            lines.append(f"Phasenfehler über der Schwelle (RMSE): {de(a.phase_rmse, 3)} rad.")
        if a.delays is not None:
            lines.append("Verzögerung: " + ", ".join(f"{D.METHOD_LABELS[m]} {de(v, 3)}" for m, v in a.delays.items()) + ".")
        if a.rel_h1 is not None:
            lines.append(f"Betrag von H1 {de(100 * a.rel_h1, 1)} %, von H2 {de(100 * a.rel_h2, 1)} % gegen den wahren Betrag.")
    else:
        head = (f"Zeitweilige Kopplung, n = {s.n}, Intervall {s.start} bis {s.end}, Rauschen {de(s.amp, 2)}, Glättung Zeit {de(s.tw, 2)} Skalen und Skala {s.ns}, Seed {s.seed}.")
        lines = [f"Wavelet-Kohärenz bei f = 0,05: innen {de(a.w_in, 3)} (wahr {de(a.w_true, 3)}), außen {de(a.w_out, 3)}.",
                 f"Phase innen {de(a.w_phase_in, 3)} rad; Welch über die ganze Aufzeichnung: Kohärenz {de(a.welch_coh, 3)}."]
    pdf.multi_cell(0, 5, _s(head + " Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: kohaerenz-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
