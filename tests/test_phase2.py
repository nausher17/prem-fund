"""Phase 2 unit + integration tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.phase2_valuation import dcf
from src.phase2_valuation.comps import load_multiples
from src.phase2_valuation.transitions import FULL_STATES, position_state
from src.phase2_valuation.ucl_option import lattice_value

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
OUT = Path(__file__).resolve().parents[1] / "outputs" / "phase2"

needs_panel = pytest.mark.skipif(
    not (PROCESSED / "panel.csv").exists(), reason="panel not built")


def test_position_states():
    assert position_state(1) == "UCL" and position_state(4) == "UCL"
    assert position_state(5) == "EUR" and position_state(7) == "EUR"
    assert position_state(8) == "MID" and position_state(14) == "MID"
    assert position_state(15) == "LOW" and position_state(17) == "LOW"
    assert position_state(18) == "REL" and position_state(20) == "REL"


@needs_panel
def test_transition_matrix_is_stochastic():
    from src.phase2_valuation.transitions import estimate
    P, counts = estimate()
    assert list(P.index) == FULL_STATES
    assert np.allclose(P.sum(axis=1), 1.0)
    assert (P.values >= 0).all()
    # empirically required features of top-flight football:
    assert P.loc["UCL", "UCL"] > 0.4        # top-4 persistence
    assert P.loc["CHAMP", "CHAMP"] > 0.8    # championship absorption
    assert P.loc["LOW", "CHAMP"] > P.loc["UCL", "CHAMP"]  # relegation gradient


def test_comps_multiples_ordered():
    m = load_multiples()
    assert m["p25"] < m["median_core"] < m["p75"]
    assert 0.5 < m["p25"] and m["p75"] < 6.0  # sector-plausible band


def test_lattice_option_properties():
    # deeper out-of-the-money -> cheaper option
    near = lattice_value(x0=6, sigma=3, annual_payoff=1e6, wacc=0.12)
    far = lattice_value(x0=15, sigma=3, annual_payoff=1e6, wacc=0.12)
    assert near > far >= 0
    # more volatility raises an out-of-the-money option's value
    lo_vol = lattice_value(x0=12, sigma=1, annual_payoff=1e6, wacc=0.12)
    hi_vol = lattice_value(x0=12, sigma=4, annual_payoff=1e6, wacc=0.12)
    assert hi_vol > lo_vol
    # payoff bound: cannot exceed undiscounted sum of payoffs
    v = lattice_value(x0=1, sigma=1, annual_payoff=1e6, wacc=0.12)
    assert v < 10 * 1e6


@needs_panel
def test_dcf_end_to_end_sane():
    res = dcf.run()
    assert len(res) == 20  # every 2023-24 PL club has revenue
    assert res.dcf_value_gbp_m.max() < 5000   # no club DCFs above GBP 5bn
    assert res.dcf_value_gbp_m.min() > -500   # losses bounded
    assert res.wacc.between(0.10, 0.16).all()
    # big six discount vs promoted premium
    spurs = res[res.club == "Tottenham Hotspur"].iloc[0]
    luton = res[res.club == "Luton Town"].iloc[0]
    assert spurs.wacc < luton.wacc
    # loss-makers must have zeroed terminal value
    lossmakers = res[res.ebitda_margin <= 0]
    assert lossmakers.terminal_zeroed_lossmaker.all()


@needs_panel
def test_valuations_output_consistency():
    path = OUT / "valuations.csv"
    if not path.exists():
        pytest.skip("run_phase2 not executed yet")
    v = pd.read_csv(path)
    assert np.allclose(v.blend_value_gbp_m,
                       (v.dcf_value_gbp_m + v.comps_value_gbp_m) / 2, atol=0.06)
    assert (v.ucl_option_gbp_m >= 0).all()
    assert (v.comps_low_gbp_m <= v.comps_high_gbp_m).all()
