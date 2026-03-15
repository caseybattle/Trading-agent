"""Session state for the trading bot CLI REPL."""
import json
import os
from typing import Optional


class Session:
    """Manages state between CLI commands in REPL mode."""

    def __init__(self):
        self._json_mode: bool = False
        self._last_btc_price: Optional[float] = None
        self._last_signals: list = []
        self._last_bankroll: Optional[float] = None

    # ── JSON output toggle ────────────────────────────────────────────────────

    @property
    def json_mode(self) -> bool:
        return self._json_mode

    @json_mode.setter
    def json_mode(self, value: bool):
        self._json_mode = value

    # ── Cached state ──────────────────────────────────────────────────────────

    @property
    def btc_price(self) -> Optional[float]:
        return self._last_btc_price

    @btc_price.setter
    def btc_price(self, value: float):
        self._last_btc_price = value

    @property
    def signals(self) -> list:
        return self._last_signals

    @signals.setter
    def signals(self, value: list):
        self._last_signals = value

    @property
    def bankroll(self) -> Optional[float]:
        return self._last_bankroll

    @bankroll.setter
    def bankroll(self, value: float):
        self._last_bankroll = value

    def to_dict(self) -> dict:
        return {
            "btc_price": self._last_btc_price,
            "signal_count": len(self._last_signals),
            "bankroll": self._last_bankroll,
        }
