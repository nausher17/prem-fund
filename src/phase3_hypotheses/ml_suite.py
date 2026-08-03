"""ML suite: LASSO, Random Forest, Gradient Boosting + mean ensemble.

Targets (one model set per target):
- revenue_growth_yoy      (PL clubs only — H4's outcome)
- squad_value_growth_yoy  (both leagues)
- wage_to_revenue         (PL clubs only)

Leakage discipline:
- features are ONLY season-t-1 lags, time-invariant flags, and macro series
  known before season t;
- temporal split: train seasons 2015-16..2020-21, test 2021-22..2023-24 —
  never random splits on panel data;
- imputation (median) and scaling are fit on TRAIN only inside a Pipeline;
- all randomness seeded (RANDOM_STATE = 42).

Reported: out-of-sample R2 and RMSE per model + naive baseline (train-mean
prediction), and impurity feature importances / LASSO coefficients. With
~250 training rows, hyperparameters are kept conservative and fixed —
honest small-data practice, not a tuning exercise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT = PROJECT_ROOT / "outputs" / "phase3"

RANDOM_STATE = 42
TRAIN_MAX_YEAR = 2020  # season 2020-21 is the last training season

FEATURES = [
    "ppg_lag1", "position_lag1", "points_lag1", "gd_pg_lag1",
    "total_market_value_eur_m_lag1", "net_spend_eur_m_lag1",
    "minutes_weighted_age_lag1", "squad_mean_age_lag1",
    "newly_promoted", "ucl_spot_lag1", "mco_flag", "covid_season",
    "is_premier_league", "uk_gilt_10y_season", "eur_gbp_season",
]

TARGETS = {
    "revenue_growth_yoy": "premier-league",
    "squad_value_growth_yoy": None,          # both leagues
    "wage_to_revenue": "premier-league",
}


def models() -> dict[str, Pipeline]:
    prep = lambda: [("impute", SimpleImputer(strategy="median")),  # noqa: E731
                    ("scale", StandardScaler())]
    return {
        "lasso": Pipeline(prep() + [("m", LassoCV(cv=5, random_state=RANDOM_STATE,
                                                  max_iter=50_000))]),
        "random_forest": Pipeline(prep() + [("m", RandomForestRegressor(
            n_estimators=500, max_depth=4, min_samples_leaf=5,
            random_state=RANDOM_STATE))]),
        "gbm": Pipeline(prep() + [("m", GradientBoostingRegressor(
            n_estimators=300, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE))]),
    }


def run_target(panel: pd.DataFrame, target: str, league: str | None):
    d = panel if league is None else panel[panel.league == league]
    d = d.dropna(subset=[target])
    train = d[d.year <= TRAIN_MAX_YEAR]
    test = d[d.year > TRAIN_MAX_YEAR]
    X_tr, y_tr = train[FEATURES], train[target]
    X_te, y_te = test[FEATURES], test[target]
    # growth targets are heavy-tailed (promotion seasons produce multi-x
    # jumps that let linear models extrapolate wildly): winsorise at the
    # TRAIN 1st/99th percentiles, applied to both splits — no leakage, and
    # documented in the limitations section
    lo, hi = y_tr.quantile(0.01), y_tr.quantile(0.99)
    y_tr, y_te = y_tr.clip(lo, hi), y_te.clip(lo, hi)

    rows, importances, preds = [], {}, {}
    baseline = np.full(len(y_te), y_tr.mean())
    rows.append({"target": target, "model": "naive_train_mean",
                 "oos_r2": r2_score(y_te, baseline),
                 "oos_rmse": float(np.sqrt(mean_squared_error(y_te, baseline))),
                 "n_train": len(y_tr), "n_test": len(y_te)})
    for name, pipe in models().items():
        pipe.fit(X_tr, y_tr)
        p = pipe.predict(X_te)
        preds[name] = p
        rows.append({"target": target, "model": name,
                     "oos_r2": r2_score(y_te, p),
                     "oos_rmse": float(np.sqrt(mean_squared_error(y_te, p))),
                     "n_train": len(y_tr), "n_test": len(y_te)})
        m = pipe.named_steps["m"]
        if hasattr(m, "feature_importances_"):
            importances[name] = dict(zip(FEATURES, m.feature_importances_))
        elif hasattr(m, "coef_"):
            importances[name] = dict(zip(FEATURES, np.abs(m.coef_)))
    ens = np.mean(list(preds.values()), axis=0)
    rows.append({"target": target, "model": "ensemble_mean",
                 "oos_r2": r2_score(y_te, ens),
                 "oos_rmse": float(np.sqrt(mean_squared_error(y_te, ens))),
                 "n_train": len(y_tr), "n_test": len(y_te)})
    return rows, importances


def main() -> int:
    from .econometrics import load_panel
    panel = load_panel().reset_index()

    all_rows, all_imp = [], []
    for target, league in TARGETS.items():
        rows, imp = run_target(panel, target, league)
        all_rows += rows
        for model, d in imp.items():
            for feat, val in d.items():
                all_imp.append({"target": target, "model": model,
                                "feature": feat, "importance": float(val)})

    OUT.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(all_rows)
    scores[["oos_r2", "oos_rmse"]] = scores[["oos_r2", "oos_rmse"]].round(4)
    scores.to_csv(OUT / "ml_scores.csv", index=False)
    pd.DataFrame(all_imp).to_csv(OUT / "ml_feature_importances.csv", index=False)
    print(scores.to_string(index=False))
    print("\nTop-5 GBM features per target:")
    imp_df = pd.DataFrame(all_imp)
    for target in TARGETS:
        top = (imp_df[(imp_df.target == target) & (imp_df.model == "gbm")]
               .nlargest(5, "importance"))
        print(f"  {target}: " + ", ".join(
            f"{r.feature} ({r.importance:.2f})" for r in top.itertuples()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
