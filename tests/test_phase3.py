"""Phase 3 tests: spec discipline and output integrity."""

from pathlib import Path

import pandas as pd
import pytest

from src.phase3_hypotheses import ml_suite

OUT = Path(__file__).resolve().parents[1] / "outputs" / "phase3"
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

needs_panel = pytest.mark.skipif(
    not (PROCESSED / "panel.csv").exists(), reason="panel not built")
needs_results = pytest.mark.skipif(
    not (OUT / "econometrics_results.csv").exists(), reason="phase3 not run")


def test_ml_features_are_leakage_free():
    # every feature must be a lag, a pre-season flag, or macro known ex ante
    allowed_unlaged = {"newly_promoted", "mco_flag", "covid_season",
                       "is_premier_league", "uk_gilt_10y_season",
                       "eur_gbp_season"}
    for f in ml_suite.FEATURES:
        assert f.endswith("_lag1") or f in allowed_unlaged, f


def test_temporal_split_boundary():
    assert ml_suite.TRAIN_MAX_YEAR == 2020  # train <=2020-21, test 2021-22+


@needs_panel
def test_h1_outcome_survivorship_documented():
    # forward value growth must be NaN for clubs absent at t+1 (never filled)
    from src.phase3_hypotheses.econometrics import load_panel
    panel = load_panel().reset_index()
    last = panel[panel.year == panel.year.max()]
    assert last.fwd_value_growth.isna().all()


@needs_results
def test_econometrics_outputs_complete():
    res = pd.read_csv(OUT / "econometrics_results.csv")
    assert set(res.hypothesis) == {"H1", "H2", "H3", "H4"}
    assert (res.n > 100).all()
    assert res.p.between(0, 1).all()
    # every hypothesis has an ex-COI robustness row except H3 (no COI club
    # in the identifying variation... keep the explicit check honest):
    for h in ("H1", "H2", "H4"):
        assert (res[res.hypothesis == h].spec.str.contains("ex-COI")).any()


@needs_results
def test_ml_scores_have_baseline_and_ensemble():
    scores = pd.read_csv(OUT / "ml_scores.csv")
    for target in scores.target.unique():
        models = set(scores[scores.target == target].model)
        assert {"naive_train_mean", "lasso", "random_forest", "gbm",
                "ensemble_mean"} <= models
    # revenue growth must beat naive OOS (headline ML result)
    rev = scores[(scores.target == "revenue_growth_yoy")]
    best = rev[rev.model != "naive_train_mean"].oos_r2.max()
    naive = rev[rev.model == "naive_train_mean"].oos_r2.iloc[0]
    assert best > naive + 0.2
