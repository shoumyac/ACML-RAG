"""Integration test: the shipped data reproduce the paper's headline numbers."""
import json
import pathlib

import pytest

from ncsrag import ALL_FEATURES, answered_rows, hallucination_labels, out_of_fold_risk, run_noise_shift

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "controlled" / "results_flan_t5_base.json"


@pytest.mark.skipif(not DATA.exists(), reason="shipped data not present")
@pytest.mark.slow
def test_headline_table():
    rows = answered_rows(json.load(open(DATA)))
    y = hallucination_labels(rows)
    p = out_of_fold_risk(rows, ALL_FEATURES)
    res = run_noise_shift(p, y, [r["qid"] for r in rows], [r["cond"] for r in rows], n_resplits=200)
    # The paper reports marginal 0.215 / 100% / 70.5% and risk-binned 0.063 / 0% / 29.2%.
    # Gradient-boosting output drifts slightly across scikit-learn versions, so we assert the
    # qualitative result and a tolerance band rather than exact digits.
    assert res["marginal"]["violation"] == pytest.approx(1.0)
    assert res["marginal"]["risk"] > 0.15
    assert res["riskbin"]["violation"] <= 0.05
    assert res["riskbin"]["risk"] < 0.10
    assert 0.25 <= res["riskbin"]["answer_rate"] <= 0.35
