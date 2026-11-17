import json
from dataclasses import replace
from datetime import timedelta

import pandas as pd
import pytest

from football_ml.config import SHADOW_PREDICTOR_NAME
from football_ml.facts import shadow
from football_ml.facts.extractor import NewsFactExtractor
from football_ml.llm.base import LLMConfigError
from football_ml.pipeline import gameweek_context, run_gameweek
from football_ml.predict.scorers import Player
from football_ml.predictors.dixon_coles import DixonColesPredictor
from football_ml.sources.espn_schedule import ScheduledMatch
from football_ml.snapshots import snapshot_path
from test_backtest import BOOTSTRAP, START, _inputs

SAKA_OUT = Player(1, "Saka", "Arsenal", 3.0, 3, 500, False, availability=0.0)
INPUTS = replace(_inputs(), players=(SAKA_OUT,))
KICKOFF = START + timedelta(weeks=1, days=1)  # GW2: Arsenal v Hull (20), Leeds v Tottenham (21)
SNAPSHOT = {
    "gameweek": 2, "taken_at": (KICKOFF - timedelta(days=2)).isoformat(), "teams": BOOTSTRAP["teams"],
    "elements": [{"id": 1, "web_name": "Saka", "team": 1, "element_type": 3, "status": "i",
                  "chance_of_playing_next_round": 0, "news": "Hamstring injury - Unknown return date",
                  "news_added": (KICKOFF - timedelta(days=3)).isoformat()}],
}
FACTS = {"players": [{"player_id": 1, "reason": "injury", "expected_return": None}]}
# Hull played a European tie two days before.
SCHEDULE = (ScheduledMatch("Hull", KICKOFF - timedelta(days=2), "uefa.conference"),)


class ScriptedProvider:
    model = "fake-model"

    def __init__(self, *responses):
        self.responses = list(responses)

    def complete_json(self, system, user, schema):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _run(provider, schedule=SCHEDULE):
    base = gameweek_context(2, INPUTS)
    predictor = DixonColesPredictor(name=SHADOW_PREDICTOR_NAME)
    return shadow.run_shadow(2, INPUTS, base, SNAPSHOT, predictor,
                             extractor=NewsFactExtractor(provider), fetch_schedules=lambda teams: schedule)


def test_shadow_report_adjusts_rates_and_records_why():
    report = _run(ScriptedProvider(FACTS))
    base = gameweek_context(2, INPUTS)
    arsenal = next(m for m in report["matches"] if m["home"] == "Arsenal")
    base_rates = next(c.rates for c in base.fixtures if c.fixture.home == "Arsenal")
    assert report["predictor"] == SHADOW_PREDICTOR_NAME
    assert arsenal["rates"]["home"] < base_rates[0]  # Saka out
    assert {a["kind"] for a in arsenal["adjustments"]["items"]} == {"key_attacker_out", "short_rest"}
    assert arsenal["adjustments"]["home_multiplier"] < 1
    assert report["model"]["shadow"] == {
        "news_model": "fake-model", "news_fallback": False, "news_items": 1, "news_facts": 1,
        "press_teams": [], "press_facts": 0, "press_fallbacks": 0, "schedule_matches": 1, "as_of": (KICKOFF - timedelta(days=2)).date().isoformat(),
    }


def test_shadow_without_usable_facts_matches_the_baseline_rates():
    report = _run(ScriptedProvider({"players": []}, {"players": []}), schedule=())
    base = gameweek_context(2, INPUTS)
    assert report["model"]["shadow"]["news_fallback"] is True
    for match, ctx in zip(report["matches"], base.fixtures):
        assert match["rates"]["home"] == pytest.approx(ctx.rates[0], abs=1e-4)
        assert match["adjustments"]["items"] == []


def _patch_inputs(monkeypatch):
    from football_ml import pipeline
    monkeypatch.setattr(pipeline, "load_season_inputs", lambda **_: INPUTS)


def _write_snapshot(snapshots_dir):
    path = snapshot_path(2, snapshots_dir)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(SNAPSHOT))


def test_run_gameweek_writes_the_shadow_next_to_the_main_prediction(monkeypatch, tmp_path):
    _patch_inputs(monkeypatch)
    _write_snapshot(tmp_path / "s")
    calls = []

    def fake_shadow(gameweek, inputs, base, snapshot, predictor):
        calls.append((gameweek, snapshot["gameweek"], predictor.name, predictor.pick_strategy))
        return {"gameweek": gameweek, "predictor": predictor.name, "matches": []}

    reports = run_gameweek(2, (DixonColesPredictor("modal"),), predictions_dir=tmp_path, snapshots_dir=tmp_path / "s",
                           now=START + timedelta(weeks=3), variants={}, shadow=fake_shadow)
    assert calls == [(2, 2, SHADOW_PREDICTOR_NAME, "modal")]
    assert [r["predictor"] for r in reports] == ["dixon-coles", SHADOW_PREDICTOR_NAME]
    assert reports[1]["variant_of"] == "dixon-coles"
    assert (tmp_path / "gw02" / f"{SHADOW_PREDICTOR_NAME}.json").exists()


def test_a_failing_shadow_never_blocks_the_main_prediction(monkeypatch, tmp_path, caplog):
    _patch_inputs(monkeypatch)
    _write_snapshot(tmp_path / "s")

    def broken(*_):
        raise LLMConfigError("bad key")

    reports = run_gameweek(2, (DixonColesPredictor(),), predictions_dir=tmp_path, snapshots_dir=tmp_path / "s",
                           now=START + timedelta(weeks=3), variants={}, shadow=broken)
    assert [r["predictor"] for r in reports] == ["dixon-coles"]
    assert "Skipping the shadow prediction" in caplog.text and "bad key" in caplog.text


def test_shadow_is_skipped_without_a_snapshot(monkeypatch, tmp_path, caplog):
    _patch_inputs(monkeypatch)
    reports = run_gameweek(2, (DixonColesPredictor(),), predictions_dir=tmp_path, snapshots_dir=tmp_path / "s",
                           now=START + timedelta(weeks=3), variants={}, shadow=lambda *_: pytest.fail("ran"))
    assert [r["predictor"] for r in reports] == ["dixon-coles"]
    assert "no GW2 snapshot" in caplog.text


def test_score_marks_the_shadow_like_any_prediction():
    from football_ml.evaluate.season import score_gameweek
    from football_ml.sources.results import ActualMatch
    report = _run(ScriptedProvider(FACTS))
    actual = tuple(ActualMatch(m["fixture_id"], 2, m["home"], m["away"], 1, 0, ()) for m in report["matches"])
    result = score_gameweek(report, actual)
    assert result["predictor"] == SHADOW_PREDICTOR_NAME and result["complete"]
    assert result["totals"]["log_loss_matches"] == 2


def test_shadow_uses_pasted_press_text_over_fpl_news(tmp_path):
    from football_ml.facts.press import PressFactExtractor
    (tmp_path / "gw02").mkdir()
    (tmp_path / "gw02" / "arsenal.md").write_text("Saka has recovered and is fit to start.", encoding="utf-8")
    press = {"players": [{"player_id": 1, "status": "fit", "reason": "returned", "expected_return": None}]}
    base = gameweek_context(2, INPUTS)
    report = shadow.run_shadow(2, INPUTS, base, SNAPSHOT, DixonColesPredictor(name=SHADOW_PREDICTOR_NAME),
                               extractor=NewsFactExtractor(ScriptedProvider(FACTS)), fetch_schedules=lambda t: (),
                               press_extractor=PressFactExtractor(ScriptedProvider(press)), previews_dir=tmp_path)
    arsenal = next(m for m in report["matches"] if m["home"] == "Arsenal")
    # FPL had him out; the press conference says he's back, so no absence, a return instead
    assert [i["kind"] for i in arsenal["adjustments"]["items"]] == ["key_attacker_back"]
    assert report["model"]["shadow"]["press_teams"] == ["Arsenal"]
    assert report["model"]["shadow"]["press_facts"] == 1 and report["model"]["shadow"]["press_fallbacks"] == 0


def test_shadow_without_a_press_extractor_ignores_previews(tmp_path):
    (tmp_path / "gw02").mkdir()
    (tmp_path / "gw02" / "arsenal.md").write_text("anything", encoding="utf-8")
    base = gameweek_context(2, INPUTS)
    report = shadow.run_shadow(2, INPUTS, base, SNAPSHOT, DixonColesPredictor(name=SHADOW_PREDICTOR_NAME),
                               extractor=NewsFactExtractor(ScriptedProvider(FACTS)), fetch_schedules=lambda t: (),
                               previews_dir=tmp_path)
    assert report["model"]["shadow"]["press_teams"] == []
