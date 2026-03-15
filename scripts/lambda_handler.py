"""
lambda_handler.py — Unified AWS Lambda entry point for the prediction market bot.

Dispatches to the appropriate script based on the 'action' field in the event:

  {"action": "morning_scan"}      -> kalshi_btc_trader one-shot scan
  {"action": "active_monitor"}    -> kalshi_btc_trader scan + auto-trade
  {"action": "expiry_intensive"}  -> kalshi_btc_trader scan + auto-trade (same as active_monitor)
  {"action": "auto_resolve"}      -> auto_resolver single pass
  {"action": "optimize"}          -> strategy_optimizer run

EventBridge rules invoke this handler with the appropriate action.
Time-window filtering for active_monitor / expiry_intensive is handled here
so the EventBridge rules can use simple rate() expressions.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone, timedelta

# Ensure scripts/ is on sys.path for sibling imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


# ---------------------------------------------------------------------------
# Time-window helpers (all times in US Eastern)
# ---------------------------------------------------------------------------

def _eastern_hour() -> int:
    """Return the current hour in US Eastern (UTC-5, no DST adjustment)."""
    # Lambda runs in UTC; EST = UTC-5, EDT = UTC-4.
    # For simplicity we use UTC-5 (EST). The 1-hour drift during EDT is
    # acceptable — the bot just starts/stops an hour early in summer.
    eastern = datetime.now(timezone(timedelta(hours=-5)))
    return eastern.hour


def _in_active_window() -> bool:
    """Active monitor window: 11 AM – 9 PM EST."""
    h = _eastern_hour()
    return 11 <= h < 21


def _in_expiry_window() -> bool:
    """Expiry intensive window: 1 PM – 6 PM EST."""
    h = _eastern_hour()
    return 13 <= h < 18


# ---------------------------------------------------------------------------
# Secrets Manager helper
# ---------------------------------------------------------------------------

def _load_secrets() -> None:
    """
    Pull Kalshi credentials from Secrets Manager into environment variables
    so existing scripts can read them from os.getenv() as usual.

    Expected secret JSON keys:
      KALSHI_API_KEY_ID   — API key ID
      KALSHI_PRIVATE_KEY  — RSA private key PEM (full text)
    """
    secret_arn = os.getenv("KALSHI_SECRET_ARN")
    if not secret_arn:
        return  # Running locally; .env already loaded

    # Only import boto3 when actually in Lambda
    import boto3

    client = boto3.client("secretsmanager")
    resp = client.get_secret_value(SecretId=secret_arn)
    secret = json.loads(resp["SecretString"])

    # Inject into env so existing scripts pick them up
    os.environ.setdefault("KALSHI_API_KEY_ID", secret.get("KALSHI_API_KEY_ID", ""))

    # For the private key: scripts expect a file path, so write it to /tmp
    pem_text = secret.get("KALSHI_PRIVATE_KEY", "")
    if pem_text:
        pem_path = "/tmp/kalshi_private_key.pem"
        with open(pem_path, "w") as fh:
            fh.write(pem_text)
        os.environ.setdefault("KALSHI_PRIVATE_KEY_PATH", pem_path)


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: dict, context) -> dict:
    """
    AWS Lambda entry point. Dispatches based on event["action"].

    Parameters
    ----------
    event : dict
        Must contain "action" key. Optional "bankroll" (default "10").
    context : LambdaContext
        AWS Lambda context object (unused).

    Returns
    -------
    dict
        JSON-serializable result with "status" and action-specific data.
    """
    action = event.get("action", "morning_scan")
    bankroll = str(event.get("bankroll", "10"))

    try:
        # Load secrets on every cold/warm start
        _load_secrets()

        if action == "morning_scan":
            from kalshi_btc_trader import main as trader_main
            trader_main(["--once", "--bankroll", bankroll])
            return {"status": "ok", "action": action}

        elif action == "active_monitor":
            if not _in_active_window():
                return {"status": "skipped", "action": action, "reason": "outside 11AM-9PM EST window"}
            from kalshi_btc_trader import main as trader_main
            trader_main(["--once", "--auto-trade", "--bankroll", bankroll])
            return {"status": "ok", "action": action}

        elif action == "expiry_intensive":
            if not _in_expiry_window():
                return {"status": "skipped", "action": action, "reason": "outside 1PM-6PM EST window"}
            from kalshi_btc_trader import main as trader_main
            trader_main(["--once", "--auto-trade", "--bankroll", bankroll])
            return {"status": "ok", "action": action}

        elif action == "auto_resolve":
            from auto_resolver import main as resolver_main
            resolver_main(["--once"])
            return {"status": "ok", "action": action}

        elif action == "optimize":
            from strategy_optimizer import run_optimization
            result = run_optimization(dry_run=False)
            # Ensure the result is JSON-serializable
            return {"status": "ok", "action": action, "config": _sanitize(result)}

        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[lambda_handler] action={action} FAILED:\n{tb}")
        return {
            "status": "error",
            "action": action,
            "error": str(exc),
            "traceback": tb,
        }


def _sanitize(obj) -> dict | list | str | int | float | bool | None:
    """Coerce an object to JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)
