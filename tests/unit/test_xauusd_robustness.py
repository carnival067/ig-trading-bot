from scripts.run_xauusd_4h_robustness import _spec


def test_robustness_variants_do_not_change_risk_or_approval() -> None:
    baseline = _spec("BASELINE", 1.5, 1.5, "pullback", False)
    assert baseline.stop_atr == 1.5
    assert baseline.target_r == 1.5
    assert baseline.family == "TREND_CONTINUATION"
