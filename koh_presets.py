"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Bedingt gezeigte Regler haben keinen eigenen Widget-Schlüssel: Ihr Wert steht unter `*_kept` und überlebt so die Läufe, in denen sie nicht erscheinen (Streamlit räumt den Zustand nicht gezeigter Widgets ab).
"""

import random
from dataclasses import dataclass

import streamlit as st

import koh_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "scenario_select": SettingSpec("sc", str, C.DEFAULT_SCENARIO, choices=C.SCENARIOS),
    "seg_kept": SettingSpec("seg", int, C.DEFAULT_SEG, choices=C.SEG_CHOICES),
    "k_kept": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "overlap_kept": SettingSpec("ov", str, C.DEFAULT_OVERLAP, choices=C.OVERLAPS),
    "g2_kept": SettingSpec("g2", float, C.DEFAULT_G2, C.G2_MIN, C.G2_MAX),
    "delay_kept": SettingSpec("d", float, C.DEFAULT_D, C.D_MIN, C.D_MAX),
    "out_kept": SettingSpec("out", float, C.DEFAULT_OUT, C.OUT_MIN, C.OUT_MAX),
    "inn_kept": SettingSpec("r", float, C.DEFAULT_IN, C.IN_MIN, C.IN_MAX),
    "start_kept": SettingSpec("t0", int, C.DEFAULT_START, C.INTERVAL_MIN, C.INTERVAL_MAX),
    "end_kept": SettingSpec("t1", int, C.DEFAULT_END, C.INTERVAL_MIN, C.INTERVAL_MAX),
    "amp_kept": SettingSpec("amp", float, C.DEFAULT_AMP, C.AMP_MIN, C.AMP_MAX),
    "tw_kept": SettingSpec("tw", float, C.DEFAULT_TW, C.TW_MIN, C.TW_MAX),
    "ns_kept": SettingSpec("ns", int, C.DEFAULT_NS, choices=C.NS_CHOICES),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"scenario": "scenario_select", "seg": "seg_kept", "k": "k_kept", "overlap": "overlap_kept", "g2": "g2_kept", "delay": "delay_kept", "out": "out_kept", "inn": "inn_kept", "start": "start_kept",
               "end": "end_kept", "amp": "amp_kept", "tw": "tw_kept", "ns": "ns_kept", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    preset = C.PRESETS[name]
    for key, state_key in PRESET_KEYS.items():
        if key in preset:
            st.session_state[state_key] = preset[key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
