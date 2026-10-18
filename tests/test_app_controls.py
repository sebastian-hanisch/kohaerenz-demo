"""AppTest: Szenarien, Permalink, bedingte Regler, PDF (die Rauchtests in test_app.py decken nur Knöpfe und Slider ab)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import koh_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
G2 = "Wahre Kohärenz"
DELAY = "Verzögerung d (Abtastwerte)"
OUT = "Ausgangsrauschen"
INN = "Eingangsrauschen r"
START = "Beginn der Kopplung"
TW = "Glättung in der Zeit"
SEG = "Segmentlänge"
K = "Zahl der Segmente K"


def _run(params=None):
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    if params:
        at.query_params.update(params)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _labels(at):
    return [x.label for x in at.slider] + [x.label for x in at.select_slider]


def _widget(at, label):
    return {x.label: x for x in list(at.slider) + list(at.select_slider)}[label]


@pytest.mark.parametrize("scenario", list(C.SCENARIOS))
def test_every_scenario_runs(scenario):
    at = _run()
    at.selectbox(key="scenario_select").set_value(scenario).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_the_controls_follow_the_scenario():
    at = _run()
    assert {G2, DELAY, SEG, K} <= set(_labels(at)) and not {OUT, INN, START, TW} & set(_labels(at))
    at.selectbox(key="scenario_select").set_value("FIR-System").run(timeout=TIMEOUT)
    assert {OUT, INN, SEG, K} <= set(_labels(at)) and not {G2, DELAY, START, TW} & set(_labels(at))
    at.selectbox(key="scenario_select").set_value("Zeitweilige Kopplung").run(timeout=TIMEOUT)
    assert {START, TW} <= set(_labels(at)) and not {G2, DELAY, OUT, INN, SEG, K} & set(_labels(at))


def test_conditional_values_survive_the_runs_without_their_widget():
    at = _run()
    _widget(at, G2).set_value(0.8).run(timeout=TIMEOUT)
    _widget(at, DELAY).set_value(7.5).run(timeout=TIMEOUT)
    _widget(at, K).set_value(9).run(timeout=TIMEOUT)
    at.selectbox(key="scenario_select").set_value("FIR-System").run(timeout=TIMEOUT)
    _widget(at, INN).set_value(0.5).run(timeout=TIMEOUT)
    at.selectbox(key="scenario_select").set_value("Zeitweilige Kopplung").run(timeout=TIMEOUT)
    _widget(at, TW).set_value(1.5).run(timeout=TIMEOUT)
    at.selectbox(key="scenario_select").set_value("Verzögerung").run(timeout=TIMEOUT)
    assert not at.exception and _widget(at, G2).value == 0.8 and _widget(at, DELAY).value == 7.5 and _widget(at, K).value == 9
    at.selectbox(key="scenario_select").set_value("FIR-System").run(timeout=TIMEOUT)
    assert _widget(at, INN).value == 0.5 and _widget(at, K).value == 9
    at.selectbox(key="scenario_select").set_value("Zeitweilige Kopplung").run(timeout=TIMEOUT)
    assert _widget(at, TW).value == 1.5


def test_the_metric_columns_follow_the_scenario():
    at = _run()
    assert [m.label for m in at.metric] == ["Mittlere Kohärenz", "Schwelle 95 %", "Segmente K", "Phasenfehler", "Verzögerung geschätzt"]
    at.selectbox(key="scenario_select").set_value("FIR-System").run(timeout=TIMEOUT)
    assert [m.label for m in at.metric] == ["Mittlere Kohärenz", "Schwelle 95 %", "Segmente K", "Phasenfehler", "|H1| gegen wahr"]
    at.selectbox(key="scenario_select").set_value("Zeitweilige Kopplung").run(timeout=TIMEOUT)
    assert [m.label for m in at.metric] == ["Kohärenz innen", "Kohärenz außen", "Welch, ganze Aufzeichnung", "Phase innen"]


def test_permalink_is_applied_and_invalid_values_are_ignored():
    at = _run({"sc": "FIR-System", "seg": "128", "k": "12", "ov": "50 %", "out": "0.5", "r": "0.25", "seed": "42"})
    assert at.selectbox(key="scenario_select").value == "FIR-System" and _widget(at, SEG).value == 128 and _widget(at, K).value == 12 and _widget(at, OUT).value == 0.5 and _widget(at, INN).value == 0.25
    assert at.radio[0].value == "50 %" and at.number_input(key="seed_input").value == 42
    at2 = _run({"sc": "Unsinn", "seg": "99", "k": "999", "ov": "30 %", "g2": "5", "d": "-4"})
    assert at2.selectbox(key="scenario_select").value == C.DEFAULT_SCENARIO and _widget(at2, SEG).value == C.DEFAULT_SEG and _widget(at2, K).value == C.K_MAX and _widget(at2, G2).value == C.G2_MAX
    assert _widget(at2, DELAY).value == C.D_MIN and at2.radio[0].value == C.DEFAULT_OVERLAP
    at3 = _run({"sc": "Zeitweilige Kopplung", "t0": "850", "t1": "860"})
    assert not at3.exception and len(at3.warning) >= 1


def test_preset_button_sets_the_widgets():
    at = _run()
    names = list(C.PRESETS)
    at.button(key=f"preset_{names[3]}").click().run(timeout=TIMEOUT)
    assert not at.exception and at.selectbox(key="scenario_select").value == "FIR-System" and _widget(at, INN).value == 1.0 and _widget(at, K).value == 64
    at.button(key=f"preset_{names[2]}").click().run(timeout=TIMEOUT)
    assert at.selectbox(key="scenario_select").value == "Verzögerung" and _widget(at, DELAY).value == 16.0 and _widget(at, SEG).value == 128
    at.button(key=f"preset_{names[4]}").click().run(timeout=TIMEOUT)
    assert at.selectbox(key="scenario_select").value == "Zeitweilige Kopplung" and at.number_input(key="seed_input").value == 80


def test_pdf_button_builds_a_download():
    at = _run()
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any(d.proto.label.startswith("⬇️") for d in at.get("download_button"))


def test_the_dice_button_changes_the_seed():
    at = _run()
    before = at.number_input(key="seed_input").value
    [b for b in at.button if "würfeln" in b.label][0].click().run(timeout=TIMEOUT)
    assert not at.exception and at.number_input(key="seed_input").value != before
