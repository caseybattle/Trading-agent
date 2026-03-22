"""
watchdog.py
Daily health check: verifies bankroll file exists, last trade is recent, and
key env vars are set. Prints a status report. Exits non-zero on critical failures
so Railway logs the alert.
"""

import os
import sys
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

try:
    from storage_backend import get_storage
    _storage = get_storage()
except ImportError:
    _storage = None


def check_env() -> list[str]:
    missing = []
    for var in ("KALSHI_API_KEY_ID",):
        if not os.getenv(var):
            missing.append(var)
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
    key_contents = os.getenv("KALSHI_PRIVATE_KEY_CONTENTS", "")
    if not key_path and not key_contents:
        missing.append("KALSHI_PRIVATE_KEY_PATH or KALSHI_PRIVATE_KEY_CONTENTS")
    return missing


def check_bankroll() -> dict:
    try:
        if _storage:
            data = _storage.read_json("trades/bankroll.json")
        else:
            p = _REPO_ROOT / "trades" / "bankroll.json"
            with open(p) as f:
                data = json.load(f)
        return {"ok": True, "balance": data.get("balance", "?"), "data": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def check_last_trade() -> dict:
    try:
        if _storage:
            import pandas as pd
            import io
            raw = _storage.read_bytes("trades/live_trades.parquet")
            df = pd.read_parquet(io.BytesIO(raw))
        else:
            import pandas as pd
            p = _REPO_ROOT / "trades" / "live_trades.parquet"
            df = pd.read_parquet(p)
        if df.empty:
            return {"ok": True, "note": "no trades yet"}
        last_ts = df["timestamp"].max() if "timestamp" in df.columns else None
        if last_ts is None:
            return {"ok": True, "note": "timestamp column missing"}
        age = datetime.now(timezone.utc) - last_ts.to_pydatetime().replace(tzinfo=timezone.utc)
        stale = age > timedelta(hours=2)
        return {"ok": not stale, "last_trade_age_hours": round(age.total_seconds() / 3600, 1)}
    except FileNotFoundError:
        return {"ok": True, "note": "no trade ledger yet"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def main():
    now = datetime.now(timezone.utc).isoformat()
    print(f"[WATCHDOG] {now}")

    failures = 0

    missing_env = check_env()
    if missing_env:
        print(f"  [ENV] MISSING: {missing_env}")
        failures += 1
    else:
        print("  [ENV] OK")

    br = check_bankroll()
    if br["ok"]:
        print(f"  [BANKROLL] OK — balance: {br.get('balance', '?')}")
    else:
        print(f"  [BANKROLL] FAIL — {br.get('error')}")
        failures += 1

    lt = check_last_trade()
    if lt["ok"]:
        print(f"  [TRADES] OK — {lt.get('note', '')} {lt.get('last_trade_age_hours', '')}")
    else:
        print(f"  [TRADES] STALE — last trade {lt.get('last_trade_age_hours')}h ago")
        failures += 1

    if failures:
        print(f"[WATCHDOG] {failures} issue(s) detected")
        sys.exit(1)
    print("[WATCHDOG] All checks passed")


if __name__ == "__main__":
    main()
