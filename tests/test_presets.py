"""Presets: Vollständigkeit, gültige Werte, Standardwerte, Parameterschlüssel."""

import koh_constants as C
import koh_evaluation as ev
import koh_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_choices_and_bounds():
    for name, p in C.PRESETS.items():
        assert p["scenario"] in C.SCENARIOS and p["seg"] in C.SEG_CHOICES and C.K_MIN <= p["k"] <= C.K_MAX and p["overlap"] in C.OVERLAPS and p["ns"] in C.NS_CHOICES, name
        assert C.G2_MIN <= p["g2"] <= C.G2_MAX and C.D_MIN <= p["delay"] <= C.D_MAX and C.OUT_MIN <= p["out"] <= C.OUT_MAX and C.IN_MIN <= p["inn"] <= C.IN_MAX, name
        assert C.INTERVAL_MIN <= p["start"] < p["end"] <= C.INTERVAL_MAX and C.AMP_MIN <= p["amp"] <= C.AMP_MAX and C.TW_MIN <= p["tw"] <= C.TW_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert all(isinstance(p[k], float) for k in ("g2", "delay", "out", "inn", "amp", "tw")) and all(isinstance(p[k], int) for k in ("seg", "k", "start", "end", "ns", "seed")), name


def test_default_settings_are_valid_choices():
    s = ev.Settings()
    assert s.scenario in C.SCENARIOS and s.seg in C.SEG_CHOICES and C.K_MIN <= s.k <= C.K_MAX and s.overlap in C.OVERLAPS and s.ns in C.NS_CHOICES and set(C.WELCH_SCENARIOS) < set(C.SCENARIOS)
    assert C.G2_MIN <= s.g2 <= C.G2_MAX and C.D_MIN <= s.delay <= C.D_MAX and C.OUT_MIN <= s.out <= C.OUT_MAX and C.IN_MIN <= s.inn <= C.IN_MAX and C.AMP_MIN <= s.amp <= C.AMP_MAX and C.TW_MIN <= s.tw <= C.TW_MAX


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    s = ev.Settings()
    defaults = {"scenario": s.scenario, "seg": s.seg, "k": s.k, "overlap": s.overlap, "g2": s.g2, "delay": s.delay, "out": s.out, "inn": s.inn, "start": s.start, "end": s.end, "amp": s.amp, "tw": s.tw,
                "ns": s.ns, "seed": s.seed}
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == defaults[key], key


def test_url_params_are_unique_and_not_password_like():
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params)) and max(len(n) for n in C.PRESETS) <= 70
    assert not {"pw", "pass", "password", "pwd", "token", "key"} & set(params)


def test_bounds_and_the_constant_relations():
    assert P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("k_kept") == (C.K_MIN, C.K_MAX) and P.bounds("g2_kept") == (C.G2_MIN, C.G2_MAX) and P.bounds("scenario_select") == (None, None)
    assert C.DEFAULT_SCENARIO in C.SCENARIOS and C.DEFAULT_SEG in C.SEG_CHOICES and C.DEFAULT_OVERLAP in C.OVERLAPS and C.DEFAULT_NS in C.NS_CHOICES and all(n & (n - 1) == 0 for n in C.SEG_CHOICES)
    assert C.INTERVAL_MAX + 40 < C.N_WAVELET and C.D_MAX < C.SEG_CHOICES[-1] / 4 and C.FMIN < C.F0 - C.F_HALF and C.F0 + C.F_HALF < C.FMAX and C.WELCH_SEG_WAVELET == 128
