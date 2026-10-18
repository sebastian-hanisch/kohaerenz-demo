"""Konstanten: Regler-Grenzen, Szenarien, feste Annahmen, Experimente, Farben und Presets (Kern: koh_signals.py, koh_spectra.py, koh_theory.py, koh_delay.py, koh_wavelet.py)."""

SCENARIOS = ("Verzögerung", "FIR-System", "Zeitweilige Kopplung")
DEFAULT_SCENARIO = "Verzögerung"
WELCH_SCENARIOS = ("Verzögerung", "FIR-System")
SEG_CHOICES = (32, 64, 128, 256)
DEFAULT_SEG = 64
K_MIN, K_MAX, DEFAULT_K = 1, 128, 32                                           # Zahl der Segmente
OVERLAPS = ("0 %", "50 %")
DEFAULT_OVERLAP = "0 %"
G2_MIN, G2_MAX, G2_STEP, DEFAULT_G2 = 0.0, 0.99, 0.01, 0.5                     # wahre Kohärenz bei der Verzögerung
D_MIN, D_MAX, D_STEP, DEFAULT_D = 0.0, 40.0, 0.1, 0.0                          # Verzögerung (Abtastwerte)
OUT_MIN, OUT_MAX, OUT_STEP, DEFAULT_OUT = 0.01, 2.0, 0.01, 0.1                 # Ausgangsrauschen (Leistung gegen die Eingangsleistung 1)
IN_MIN, IN_MAX, IN_STEP, DEFAULT_IN = 0.0, 2.0, 0.05, 0.0                      # Eingangsrauschen r (Leistung gegen die Eingangsleistung 1)
FIR = (0.0, 0.6, 0.3, 0.1)
N_WAVELET = 1024
INTERVAL_MIN, INTERVAL_MAX = 100, 900
DEFAULT_START, DEFAULT_END = 300, 700                                          # Intervall der Kopplung (Abtastwerte)
AMP_MIN, AMP_MAX, AMP_STEP, DEFAULT_AMP = 0.1, 2.0, 0.1, 0.5                   # Amplitude des unabhängigen Rauschens gegen das gemeinsame Band
TW_MIN, TW_MAX, TW_STEP, DEFAULT_TW = 0.0, 2.0, 0.25, 1.0                      # Gauß-Std der Zeitglättung in Skalen
NS_CHOICES = (1, 3, 5, 7, 9)
DEFAULT_NS = 7
SEED_MAX, DEFAULT_SEED = 999999, 43

F0, F_HALF = 0.05, 0.01                                                        # Band der gemeinsamen Schwingung
PHASE0 = 1.0                                                                   # Phasenlage der Kopplung (rad)
W0 = 6.0                                                                       # Morlet
FMIN, FMAX, VOICES = 0.02, 0.2, 12
WELCH_SEG_WAVELET = 128
EDGE = 3                                                                       # Bins am Rand, die nicht in die Bandmittel eingehen
ALPHA = 0.05
DELAY_STEP = 0.005                                                             # Gitter der Verzögerungsschätzer
DELAY_RANGE = 8.0

# Experimente (feste Ströme)
EXP_SEED = 2100
EXP_BIAS_SEG, EXP_BIAS_RUNS = 64, 200
EXP_BIAS_KS = (2, 4, 8, 16, 32, 64, 128)
EXP_BIAS_G2S = (0.0, 0.25, 0.5, 0.75, 0.9)
EXP_THR_KS, EXP_THR_RUNS = (2, 4, 8, 16, 32, 64), 400
EXP_PHASE_SEG, EXP_PHASE_RUNS = 64, 300
EXP_PHASE_KS = (8, 32, 128)
EXP_PHASE_G2S = (0.25, 0.5, 0.9)
EXP_DELAY_SEG, EXP_DELAY_D, EXP_DELAY_RUNS = 256, 3.4, 150
EXP_DELAY_CASES = ((1024, 0.9), (1024, 0.5), (1024, 0.2), (1024, 0.05), (4096, 0.9), (4096, 0.5), (4096, 0.2), (4096, 0.05))
EXP_TRANSFER_SEG, EXP_TRANSFER_K, EXP_TRANSFER_RUNS = 64, 64, 100
EXP_TRANSFER_OUT, EXP_TRANSFER_IN = (0.1, 0.5), (0.0, 0.25, 1.0)
EXP_WINDOW_SEG, EXP_WINDOW_K, EXP_WINDOW_RUNS = 128, 64, 100
EXP_WINDOW_DS = (0, 1, 2, 4, 8, 16, 32, 64)
EXP_WAVELET_RUNS = 100
EXP_WAVELET_SMOOTH = ((0.0, 1), (0.5, 1), (1.0, 1), (2.0, 1), (0.5, 7), (1.0, 7), (2.0, 7))
EXP_WAVELET_COUPLED = ((0.5, 1), (1.0, 1), (2.0, 1), (1.0, 7))
EXP_WAVELET_FREQS = (0.03, 0.05, 0.08, 0.12)

COLOR_TRUE = "#B03A2E"
COLOR_EST = "#4C9F70"
COLOR_NOISY = "rgba(120,120,120,0.7)"
COLOR_AUX = "#E8833A"
COLOR_BLUE = "rgba(76,120,168,0.95)"

_BASE = {"scenario": "Verzögerung", "seg": 64, "k": 32, "overlap": "0 %", "g2": 0.5, "delay": 0.0, "out": 0.1, "inn": 0.0, "start": 300, "end": 700, "amp": 0.5, "tw": 1.0, "ns": 7, "seed": DEFAULT_SEED}
PRESETS = {
    "Ein Segment: Kohärenz ist immer 1": {**_BASE, "k": 1, "g2": 0.0},
    "Kurze Daten: wahre Kohärenz 0,5, gemessen fast 0,7": {**_BASE, "k": 2, "g2": 0.5, "seed": 32},
    "Verzögerung: Phase und Kohärenzverlust durch das Fenster": {**_BASE, "seg": 128, "k": 64, "g2": 0.95, "delay": 16.0},
    "Eingangsrauschen: H1 zu klein, H2 zu groß": {**_BASE, "scenario": "FIR-System", "k": 64, "out": 0.1, "inn": 1.0},
    "Wavelet-Kohärenz: eine zeitweilige Kopplung finden": {**_BASE, "scenario": "Zeitweilige Kopplung", "seed": 80},
}
PRESET_HELP = {
    "Ein Segment: Kohärenz ist immer 1": "Zwei unabhängige Reihen (wahre Kohärenz 0), aber nur ein Segment: Die geschätzte Kohärenz ist bei jeder Frequenz genau 1. Erst mit mehreren Segmenten wird sie ein Maß.",
    "Kurze Daten: wahre Kohärenz 0,5, gemessen fast 0,7": "Zwei Segmente, wahre Kohärenz 0,5: Die Schätzung liegt bei 0,69 (im Mittel über viele Läufe; hier 0,69 im gezeigten Lauf), weil die Kohärenz aus wenigen Segmenten nach oben verzerrt ist.",
    "Verzögerung: Phase und Kohärenzverlust durch das Fenster": "Verzögerung 16 Abtastwerte bei Segmentlänge 128 und wahrer Kohärenz 0,95: Die Phase fällt linear mit der Frequenz (daraus die Verzögerung), die gemessene Kohärenz liegt um etwa 20 % tiefer, weil das Fenster gegen die verzögerte Reihe verschoben ist.",
    "Eingangsrauschen: H1 zu klein, H2 zu groß": "Übertragungsfunktion mit Eingangsrauschen von der Größe des Eingangs: H1 = S_xy/S_xx liegt bei der Hälfte des wahren Betrags, H2 = S_yy/S_yx darüber; die Phase stimmt in beiden.",
    "Wavelet-Kohärenz: eine zeitweilige Kopplung finden": "Zwei Bandrauschen bei 0,05, gekoppelt nur zwischen 300 und 700: Die Wavelet-Kohärenz zeigt die Kopplung als Intervall (innen 0,75, außen 0,49, jeweils mit dem Bias der Glättung), die Kohärenz über die ganze Aufzeichnung (Welch) sieht nur 0,25.",
}
