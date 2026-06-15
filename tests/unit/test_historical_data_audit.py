from pathlib import Path

from scripts.audit_historical_data import infer_identity


def test_infer_identity() -> None:
    assert infer_identity(Path("/tmp/XAU:USD/M1/DAT_ASCII_XAUUSD_M1_2025.csv")) == ("XAUUSD", "M1")
    assert infer_identity(Path("/tmp/BTC:USDT/M1/BTCUSDT-1m-2025-01.zip")) == ("BTCUSDT", "M1")
    assert infer_identity(Path("/tmp/EUR:USD/TICK/DAT_ASCII_EURUSD_T_202501.csv")) == ("EURUSD", "TICK")
