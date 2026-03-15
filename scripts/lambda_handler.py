"""Unified AWS Lambda entry point for all scheduled trading bot tasks."""

import os
import sys
import json
import logging
from pathlib import Path

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add project root to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def handle_morning_scan(bankroll: float) -> dict:
    """Morning 9 AM EST single-cycle scan."""
    try:
        from scripts.kalshi_btc_trader import main as trader_main
        
        logger.info(f"Morning scan started with bankroll=${bankroll}")
        result = trader_main(
            bankroll=bankroll,
            auto_trade=False,
            once=True  # Single cycle
        )
        return {"status": "success", "message": f"Morning scan completed", "result": result}
    except Exception as e:
        logger.error(f"Morning scan failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}


def handle_active_monitor(bankroll: float) -> dict:
    """Active monitoring every 30 min, 11 AM - 9 PM EST."""
    try:
        from scripts.kalshi_btc_trader import main as trader_main
        
        logger.info(f"Active monitor started with bankroll=${bankroll}")
        result = trader_main(
            bankroll=bankroll,
            auto_trade=False,
            once=True  # Single cycle
        )
        return {"status": "success", "message": "Active monitor cycle completed", "result": result}
    except Exception as e:
        logger.error(f"Active monitor failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}


def handle_expiry_intensive(bankroll: float) -> dict:
    """Expiry-intensive monitoring every 15 min, 1 - 6 PM EST."""
    try:
        from scripts.kalshi_btc_trader import main as trader_main
        
        logger.info(f"Expiry intensive started with bankroll=${bankroll}")
        result = trader_main(
            bankroll=bankroll,
            auto_trade=False,
            once=True  # Single cycle
        )
        return {"status": "success", "message": "Expiry intensive cycle completed", "result": result}
    except Exception as e:
        logger.error(f"Expiry intensive failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}


def handle_auto_resolve() -> dict:
    """Auto-resolve settled trades every 30 min, 9 AM - 10 PM."""
    try:
        from scripts.auto_resolver import main as resolver_main
        
        logger.info("Auto-resolve started")
        result = resolver_main()
        return {"status": "success", "message": "Auto-resolve completed", "result": result}
    except Exception as e:
        logger.error(f"Auto-resolve failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}


def handle_optimizer() -> dict:
    """Standalone strategy optimizer daily at 10 PM EST."""
    try:
        from scripts.strategy_optimizer import main as optimizer_main
        
        logger.info("Strategy optimizer started")
        result = optimizer_main()
        return {"status": "success", "message": "Strategy optimizer completed", "result": result}
    except Exception as e:
        logger.error(f"Strategy optimizer failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}


def lambda_handler(event, context):
    """Main Lambda handler dispatching to task-specific handlers."""
    task_name = os.getenv("TASK_NAME", "").lower()
    bankroll = float(os.getenv("BANKROLL", "10"))
    
    logger.info(f"Lambda invoked: TASK_NAME={task_name}, BANKROLL=${bankroll}")
    
    try:
        if task_name == "morning-scan":
            result = handle_morning_scan(bankroll)
        elif task_name == "active-monitor":
            result = handle_active_monitor(bankroll)
        elif task_name == "expiry-intensive":
            result = handle_expiry_intensive(bankroll)
        elif task_name == "auto-resolve":
            result = handle_auto_resolve()
        elif task_name == "optimizer":
            result = handle_optimizer()
        else:
            result = {
                "status": "error",
                "message": f"Unknown TASK_NAME: {task_name}. Valid options: morning-scan, active-monitor, expiry-intensive, auto-resolve, optimizer"
            }
        
        return {
            "statusCode": 200 if result["status"] == "success" else 500,
            "body": json.dumps(result)
        }
    
    except Exception as e:
        logger.error(f"Lambda handler failed: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({
                "status": "error",
                "message": f"Lambda handler exception: {str(e)}"
            })
        }


if __name__ == "__main__":
    # For local testing
    os.environ.setdefault("TASK_NAME", "morning-scan")
    os.environ.setdefault("BANKROLL", "10")
    result = lambda_handler({}, None)
    print(json.dumps(result, indent=2))
