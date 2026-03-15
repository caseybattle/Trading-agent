# Trading Bot CLI — Agent Harness

CLI-Anything harness for the Kalshi BTC Range trading bot. Makes all bot operations
agent-native via a Click CLI with JSON output and interactive REPL.

## Installation

```bash
cd prediction-market-bot/agent-harness
pip install -e .
```

Requires the bot's scripts directory on the path (handled automatically via setup.py).
Also requires the root `requirements.txt` dependencies to be installed.

## Commands

```
trading-bot [--json] <group> <command> [options]
trading-bot          # REPL mode
```

### market — Scan Kalshi BTC range markets
| Command | Description |
|---------|-------------|
| `market price` | Current BTC spot price |
| `market scan [--bankroll N] [--min-edge N] [--vol N]` | Scan all KXBTC markets, return signals |
| `market markets` | List all available KXBTC markets |

### trade — Place orders and view positions
| Command | Description |
|---------|-------------|
| `trade positions` | Show open Kalshi positions |
| `trade execute --ticker T --side YES/NO --contracts N --price P [--dry-run]` | Place limit order |

### backtest — Historical backtests
| Command | Description |
|---------|-------------|
| `backtest run [--bankroll N] [--min-edge N] [--vol N] [--kelly-fraction N]` | Walk-forward backtest |
| `backtest sweep [--bankroll N]` | Parameter sweep (edge × vol grid) |

### bankroll — Ledger management
| Command | Description |
|---------|-------------|
| `bankroll status` | Current balance and daily P&L |
| `bankroll init --amount N` | Initialize with starting balance |
| `bankroll add-trade --ticker T --direction YES/NO --contracts N --entry-price P --fair-value F --edge E --btc-price B` | Record trade |
| `bankroll resolve-trade --trade-id ID --outcome WIN/LOSS` | Resolve a trade |
| `bankroll trades` | List all ledger trades |

### optimize — Strategy optimizer
| Command | Description |
|---------|-------------|
| `optimize run [--dry-run]` | Run optimizer (writes strategy_config.json) |
| `optimize config` | Show current strategy_config.json |
| `optimize bounds` | Show parameter search bounds |

### resolve — Auto-resolve settled trades
| Command | Description |
|---------|-------------|
| `resolve run [--dry-run]` | Fetch settled markets, resolve trades, run postmortem |

### status — Quick status
```bash
trading-bot status      # BTC price + bankroll + open positions
trading-bot --json status   # Machine-readable JSON
```

## JSON Output (Agent Mode)

All commands support `--json` for structured output:

```bash
trading-bot --json market scan --bankroll 10
# {"btc_price_usd": 84500.0, "markets_found": 12, "signals": [...]}

trading-bot --json status
# {"btc_price_usd": 84500.0, "bankroll_usd": 9.85, "open_positions": [...]}
```

## REPL Mode

```bash
trading-bot          # starts REPL
trading-bot> market scan
trading-bot> trade positions
trading-bot> status
trading-bot> exit
```

## Architecture

```
agent-harness/
  setup.py                              # pip-installable package
  HARNESS.md                            # this file
  cli_anything/trading_bot/
    __init__.py
    __main__.py                         # python -m cli_anything.trading_bot
    trading_bot_cli.py                  # Click CLI — all commands
    core/
      session.py                        # REPL session state
    utils/
      __init__.py
```

The harness imports directly from `../scripts/` — no code duplication.
All existing risk safeguards (daily loss stop, file lock, bankroll floor) remain active.

## Design Decisions

- **Stateful REPL + subcommand CLI**: follows CLI-Anything methodology
- **`--json` global flag**: one flag enables JSON mode for all subcommands
- **`--dry-run` on destructive commands**: `trade execute` and `resolve run` both support it
- **No code duplication**: harness wraps existing scripts, not rewrite them
- **Session caching**: REPL caches BTC price and signals between commands for speed
