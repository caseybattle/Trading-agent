"""
live_optimizer.py
Thin wrapper that runs the strategy optimizer every 6 hours in Railway.
Delegates entirely to strategy_optimizer.run_optimization().
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from strategy_optimizer import run_optimization

if __name__ == "__main__":
    result = run_optimization(dry_run=False)
    print(f"[OPTIMIZER] Done — {result}")
