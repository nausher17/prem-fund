"""Streamlit dashboard — presentation layer for the whole project.

    .venv/bin/streamlit run src/phase6_dashboard/app.py

Reads only committed artifacts from outputs/ and data/processed/ — the app
contains no analysis logic of its own, so every number shown is reproducible
from the phase entry points.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"

st.set_page_config(page_title="Multi-Club Investment Fund", layout="wide")


@st.cache_data
def load(path, **kw):
    return pd.read_csv(path, **kw)


page = st.sidebar.radio("Pages", [
    "Overview & hypotheses", "Club explorer", "Hypothesis results",
    "Portfolio lab", "BlueCo case study", "Methodology & limitations"])

if page == "Overview & hypotheses":
    st.title("Multi-Club Investment Fund")
    st.markdown("""
**Premier League and Ligue 1 clubs as alternative assets** — fundamental
valuation (DCF over an empirical league-position Markov chain, comparable
transactions, UCL real options), panel-econometric tests of four
market-inefficiency hypotheses, and a Markowitz portfolio targeting a
15% IRR.

| Hypothesis | Verdict |
|---|---|
| H1 Promotion overvaluation | Null (survivorship caveat documented) |
| H2 MCO premium | Null — but see the BlueCo case study |
| H3 UCL optionality mispricing | Linear effect priced (+14.9%, p=0.0015); convexity zero |
| H4 Performance → revenue growth | **Supported** (β=0.204, p=0.048; OOS R²=0.54) |

**Headline finding — the trophy-asset premium:** Forbes values the top PL
clubs at a median **2.98×** our cash-flow DCF. The gap is the price of
scarcity and prestige, quantified.
""")
    st.info("Conflict of interest: the author is a Chelsea supporter. "
            "Chelsea and Strasbourg carry coi_flag=1 — flagged, never excluded.")

elif page == "Club explorer":
    st.title("Club explorer — FY2024 valuations")
    v = load(OUT / "phase2/valuations.csv")
    club = st.selectbox("Club", v.club)
    r = v[v.club == club].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DCF (£m)", f"{r.dcf_value_gbp_m:,.0f}")
    c2.metric("Comps (£m)", f"{r.comps_value_gbp_m:,.0f}")
    c3.metric("Blend (£m)", f"{r.blend_value_gbp_m:,.0f}")
    c4.metric("UCL option (£m)", f"{r.ucl_option_gbp_m:,.1f}")
    st.caption(f"State {r.state} · revenue £{r.revenue_fy24_gbp_m:.0f}m · "
               f"wage ratio {r.wage_ratio:.0%} · WACC {r.wacc:.1%} · "
               f"P(UCL in 5y) {r.p_ucl_5y:.0%}")
    fig = px.bar(v.sort_values("blend_value_gbp_m"), x="blend_value_gbp_m",
                 y="club", orientation="h",
                 labels={"blend_value_gbp_m": "Blend value (£m)", "club": ""})
    st.plotly_chart(fig, use_container_width=True)
    tp = load(OUT / "phase2/trophy_premium.csv")
    st.subheader("Trophy-asset premium (Forbes May-2024 vs DCF)")
    st.dataframe(tp[["club", "forbes_gbp_m", "dcf_value_gbp_m",
                     "trophy_premium_gbp_m", "premium_x_dcf"]],
                 hide_index=True)

elif page == "Hypothesis results":
    st.title("Hypothesis tests")
    st.dataframe(load(OUT / "phase3/econometrics_results.csv"), hide_index=True)
    st.subheader("ML suite (temporal splits)")
    st.dataframe(load(OUT / "phase3/ml_scores.csv"), hide_index=True)
    st.markdown((OUT / "phase3/findings.md").read_text())

elif page == "Portfolio lab":
    st.title("Portfolio lab")
    er = load(OUT / "phase4/expected_returns.csv", index_col=0)
    w = load(OUT / "phase4/weights.csv", index_col=0)
    stats = load(OUT / "phase4/portfolio_stats.csv", index_col=0)
    fr = load(OUT / "phase4/frontier.csv")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Efficient frontier (long-only, 20% cap)")
        fig = go.Figure()
        fig.add_scatter(x=fr.vol, y=fr.target_return, mode="lines",
                        name="frontier")
        for name in ("max_sharpe", "min_var"):
            fig.add_scatter(x=[stats.loc[name, "vol"]],
                            y=[stats.loc[name, "exp_return"]],
                            mode="markers+text", text=[name],
                            textposition="top center", name=name)
        fig.update_layout(xaxis_title="Volatility", yaxis_title="E[return]")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Max-Sharpe weights")
        st.plotly_chart(px.bar(w.max_sharpe[w.max_sharpe > 0.001]),
                        use_container_width=True)
    st.subheader("Risk & simulation")
    st.dataframe(stats)
    st.dataframe(load(OUT / "phase4/backtest_summary.csv"), hide_index=True)
    st.markdown((OUT / "phase4/benchmark.md").read_text())
    st.caption("Mark-to-model returns (squad-value proxy) — clubs do not "
               "trade annually; see limitations.")

elif page == "BlueCo case study":
    st.title("BlueCo: Chelsea & Strasbourg")
    st.warning("COI: the author is a Chelsea supporter (flagged, never excluded).")
    st.image(str(OUT / "phase5/trajectory_fig.png"))
    st.image(str(OUT / "phase5/network_fig.png"))
    st.markdown((OUT / "phase5/findings.md").read_text())

else:
    st.title("Methodology & limitations")
    st.markdown((ROOT / "README.md").read_text())
