# Kohärenz: Was zwei Signale verbindet – bei wenigen Segmenten ist die Kohärenz stark nach oben verzerrt, und Eingangsrauschen kippt die Übertragungsfunktion – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-kohaerenz-demo.streamlit.app/)**

Fünfzehnte Demo der Reihe **Spektral- und Wavelet-Analyse** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, aufbauend auf den Demos zum **Welch-Verfahren** und zum
**Skalogramm**. Aus zwei Reihen werden Kreuzspektrum, **Kohärenz**, Phase, **Verzögerung** und die Übertragungsfunktionen **H1** und **H2** geschätzt; die **Wavelet-Kohärenz** findet eine zeitweilige Kopplung. Gemessen
wird alles gegen die bekannten wahren Werte (Kopplung, Verzögerung, Übertragungsfunktion).

## Warum dieses Problem

Die Kohärenz ist das erste Maß, zu dem man bei zwei Messreihen greift: eine Zahl zwischen 0 und 1 je Frequenz. Aber sie wird aus einer endlichen Zahl von Segmenten geschätzt und ist dann nach oben verzerrt; mit einem
Segment ist sie immer 1. Dazu kommen die Fallen der Zwei-Kanal-Analyse: Rauschen am Eingang verfälscht die Übertragungsfunktion, eine Verzögerung kostet Kohärenz durch das Fenster, und eine nur zeitweilige Kopplung wird
über die ganze Aufzeichnung verdünnt. Die Demo misst jede dieser Fallen.

## Modell

Eingang x: weißes Gauß-Rauschen mit Einheitsleistung. **Verzögerung:** y[t] = x[t − d] (zirkulär, auch gebrochen, per FFT) plus Rauschen, dessen Leistung die wahre Kohärenz γ² = 1/(1 + Rauschleistung) einstellt.
**FIR-System:** y = h * x + e mit h = (0; 0,6; 0,3; 0,1), Ausgangsrauschen e der Leistung N, x mit Eingangsrauschen der Leistung r beobachtet; wahre Kohärenz |H|²/((1 + r)(|H|² + N)). **Zeitweilige Kopplung:** x = c + a b₁,
y = g(t) c(φ) + (1 − g(t)) b₃ + a b₂ mit unabhängigem Bandrauschen c, b₁, b₂, b₃ bei 0,05 ± 0,01 (Leistung 1), Kopplungsmaske g (tanh-Ränder), Phase φ = 1,0 rad und wahrer Kohärenz im Intervall 1/(1 + a²)².
Kreuzspektrum nach Welch (Hann-Fenster, Überlappung 0 oder 50 %): S_xy = Mittel von conj(X_j) Y_j; Phase von S_xy ist die von H; Verzögerungsschätzer auf einem Gitter der Schrittweite 0,005. Wavelet-Kohärenz mit dem analytischen
Morlet-Wavelet (ω₀ = 6), geglättet in der Zeit (Gauß, Standardabweichung gleich einem Vielfachen der Skala) und in der Skala (Kasten über benachbarte Frequenzen, 12 je Oktave). Alles in numpy; scipy dient nur als Gegenprobe
in den Tests.

## Methodik

Experimente über feste Zufallsströme (Seed 2100 und folgende): 200 Läufe beim Bias (Segmentlänge 64, Bins 2 bis 30), 400 Läufe bei der Schwelle, 300 Läufe bei der Phase, 150 Läufe bei der Verzögerung (Segmentlänge 256,
d = 3,4), 100 Läufe bei der Übertragungsfunktion (K = 64), bei der Verzögerung gegen die Segmentlänge (128, K = 64) und bei der Wavelet-Kohärenz (n = 1024, f₀ = 0,05). „Wahr“ ist der Wert, mit dem die Reihen gebaut sind;
„erwartet“ ist der Mittelwert der Schätzung bei dieser Zahl unabhängiger Segmente (numerische Integration der Dichte, ohne Bibliothek; bei Überlappung mit gerundeter K_eff nur eine Näherung).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` und `tests/test_experiments_golden.py`.

| Frage | Befund |
|---|---|
| Wie verzerrt ist die Kohärenz (Experiment *Kohärenzbias und Schwelle*)? | Bei einem Segment ist sie bei jeder Frequenz 1, auch bei unabhängigen Reihen. Bei K unabhängigen Segmenten und wahrer Kohärenz 0 ist der Mittelwert der Schätzung genau 1/K (K = 8: 0,125 gemessen 0,125; K = 32: 0,031 gemessen 0,032); bei wahrer Kohärenz 0,5 liegt er bei K = 2 bei 0,69 (exakt ln 2 = 0,693), bei K = 8 bei 0,54 und bei K = 32 bei 0,51. Die erste Ordnung g² + (1 − g²)²/K trifft ab K = 32 (g² = 0,25: 0,268 gemessen, 0,268). Die 95-%-Schwelle 1 − 0,05^(1/(K − 1)) (K = 2: 0,95; K = 8: 0,35; K = 64: 0,046) wird bei Kopplung null von 4,6 % bis 5,2 % der Bins überschritten. Bei 50 % Überlappung entsprechen 127 Segmente 120 unabhängigen (gemessen 122), bei 75 % entsprechen 253 Segmente 132 (gemessen 130). |
| Wie gut sind Phase und Verzögerung (*Phase und Verzögerung*)? | Die Phase von S_xy ist unverzerrt (Mittel im Bereich ± 0,005 rad), ihre Streuung folgt sqrt((1 − γ²)/(2 K γ²)) ab K = 32 (γ² = 0,5: 0,128 gegen 0,125; K = 128: 0,063 gegen 0,062); bei K = 8 und γ² = 0,25 liegt sie darüber (0,56 gegen 0,43). Die Verzögerung d = 3,4 wird bei γ² = 0,9 und K = 31 mit einem RMSE von 0,0044 (Kreuzkorrelation), 0,0082 (nur Phase, PHAT) und 0,0032 (kohärenzgewichtet) Abtastwerten geschätzt; bei γ² = 0,05 sind alle ähnlich schlecht (0,043, 0,045 und 0,047). Das Gitter ist 0,005 fein. |
| Was richtet Eingangsrauschen an (*Übertragungsfunktion bei Eingangsrauschen*)? | \|H1\| = \|S_xy/S_xx\| liegt bei Eingangsrauschen r um den Faktor 1/(1 + r) zu tief (r = 0,25: −20 %; r = 1: −50 %), \|H2\| = S_yy/S_yx bei Ausgangsrauschen 0,1 um 36 % und bei 0,5 um 182 % zu hoch. Das geometrische Mittel gleicht das nicht aus (−18 % bis 64 %). Die Phase ist in beiden unverzerrt (unter 0,01 rad). |
| Was kostet eine Verzögerung (*Verzögerung gegen Segmentlänge*)? | Bei reiner Verzögerung und Segmentlänge 128 fällt die Kohärenz auf etwa ρ(d)² (gemessen d = 16: 0,815; d = 32: 0,438; d = 64: 0,043; ρ(d)² selbst 0,814 / 0,434 / 0,028, bei d = 64 liegt die Messung wegen des Bias bei K = 64 darüber); Zurückschieben um d stellt 1,000 her. Das Fenster, nicht das Signal, verliert die Kohärenz. |
| Was leistet die Wavelet-Kohärenz (*Wavelet-Kohärenz*)? | Ohne Glättung ist sie 1,000 bei Kopplung null; mit Glättung (Zeit 1 Skala) 0,53 (K_eff 1,9), bei Zeit 2 und Skala 7 noch 0,29; das 95-%-Quantil (0,92) liegt unter der Beta-Formel mit K_eff (0,97). Bei zeitweiliger Kopplung (wahre Kohärenz 0,64) liegt sie innen bei 0,78 und außen bei 0,53, die Phase innen bei 0,99 rad (gesetzt 1,00); Welch über die ganze Aufzeichnung sieht 0,17. Bei weißem Rauschen und Kopplung null hängt sie nicht von der Frequenz ab (0,33 bis 0,35); bei Bandrauschen bei f₀ liegt sie mit derselben Glättung (Zeit 1, Skala 7) bei 0,45. |

## Ehrliche Grenzen

- **Heimvorteil:** weißes Gauß-Rauschen als Eingang, lineares zeitinvariantes System, additives weißes Rauschen: genau die Annahmen der Formeln. Farbige Eingänge, Nichtlinearität und gemeinsame Störungen (Scheinkohärenz durch
  einen dritten Einfluss) sind nicht gemessen.
- **Kohärenz ist kein Kausalitätsbeleg:** Ein gemeinsamer Treiber gibt dieselbe Kohärenz; Richtung und Partialkohärenz bleiben offen (spektrale Kausalität nicht gebaut).
- **Die Wavelet-Kohärenz hat je nach Glättung nur 1,4 bis 3,5 äquivalente Segmente:** Das Maß ist stark verzerrt (Bias 0,29 bis 0,73 bei Kopplung null), die Beta-Schwelle passt nicht, eine Schwelle müsste simuliert werden.
- **Überlappung:** Die äquivalente Segmentzahl K_eff ist eine Näherung aus der Segmentkorrelation (120,4 gegen gemessen 121,6 bei 50 %); die Beta-Schwelle mit K_eff ist für überlappende Segmente nur ein Anhalt.
- **Verzögerungsschätzer auf einem Gitter von 0,005:** Fehler unter 0,01 liegen nahe am Gitter. Bei gebrochener Verzögerung geht der Nyquist-Anteil des Rauschens teilweise verloren (die Verzögerung ist per FFT gebildet).
- **Wahre Kohärenz bei Verzögerung:** Der Vorgabewert gilt für das ausgerichtete Paar; die gemessene Kohärenz ist um ρ(d)² geringer (siehe Experiment).
- **Quellen** (Carter, Knapp und Nuttall 1973; Knapp und Carter 1976; Bendat und Piersol; Torrence und Webster 1999; Percival und Walden 1993) wurden nicht gelesen; die Formeln sind per Handrechnung, Simulation und gegen die exakte
  Verteilung geprüft.

## Verifikation

- **Kern:** Kreuzspektrum gegen die direkte Doppelsumme und gegen `scipy.signal.csd` und `scipy.signal.coherence`; Mini-Instanz von Hand; Kohärenz 1 bei einem Segment; Schwelle gegen die Beta-Verteilung; exakter Erwartungswert gegen die Dichte mit ₂F₁ (scipy), gegen die Handformel für K = 2 (bei γ² = 0,5 genau ln 2) und gegen Simulation; Phasenstreuung; Fensterkorrelation; Verzögerungsschätzer (Vorzeichen, Gitter); Morlet-Transformation gegen die direkte Summe, Glättung gegen direkte Faltung, Wavelet-Kohärenz ohne Glättung gleich 1.
- **Experimente:** feste Werte (`tests/test_experiments_golden.py`), jede Zahl dieser README und der Texte in der App an die Auswertung gebunden (`tests/test_claims.py`), Verdikte an handgesetzten Zeilen.
- **Mutationstest:** `tools/mutate_demo.py` über alle Kernmodule (Ergebnis in `tests/mutation_state.json`).
- **Presets, Oberfläche:** Vollständigkeit und Grenzen, AppTest-Rauchtest (jeder Button, jeder Regler an den Grenzen), Permalink, bedingte Regler, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `koh_signals.py` | Rauschen, Verzögerung, FIR-System, Bandrauschen, zeitweilig gekoppeltes Paar |
| `koh_spectra.py` | Welch-Kreuzspektrum, Kohärenz, Phase, H1 und H2 |
| `koh_theory.py` | Schwelle, exakter Erwartungswert, Phasenstreuung, Fensterkorrelation, äquivalente Segmentzahl |
| `koh_delay.py` | Verzögerungsschätzer (Kreuzkorrelation, PHAT, kohärenzgewichtet) |
| `koh_wavelet.py` | Morlet-Transformation, Glättung, Wavelet-Kohärenz |
| `koh_constants.py` | Regler-Grenzen, feste Annahmen, Experimente, Farben, Presets |
| `koh_evaluation.py` | Analyse, die fünf Experimente, Verdikte |
| `koh_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `koh_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `koh_pdf_export.py` | PDF-Export der Einstellung |
| `signal_generator.py` | gemeinsamer Generator (nur SplitMix64) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Orakel (Spektren, Theorie, Verzögerung, Wavelet, Signale, Auswertung), Experimente, Claims, Verdikte, Presets, Randfälle, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

Multitaper-Kohärenz; partielle und multiple Kohärenz; spektrale Kausalität (partielle gerichtete Kohärenz); Schätzung mit Fehlern in den Variablen (totale kleinste Quadrate); farbiger Eingang; Wavelet-Kohärenz mit anderen
Wavelets als Morlet ω₀ = 6 und Signifikanzlinien aus rotem Rauschen.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Spektren und Wavelets: von Fourier zum Vergleich](https://sebastianhanisch.net/konzepte-spektral-wavelet-analyse.html).
