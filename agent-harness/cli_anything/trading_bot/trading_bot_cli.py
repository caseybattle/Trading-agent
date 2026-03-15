"""
Trading Bot CLI — CLI-Anything harness for the Kalshi BTC Range trading bot.

Exposes all major bot operations as agent-usable CLI subcommands with JSON output.

Usage:
    trading-bot [--json] <command> [args...]
    trading-bot                    # enters interactive REPL
"""

import sys
import os
import json
import traceback
from typing import Optional

import click

# ── Path setup — allow running from any directory ────────────────────────────
_HARNESS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS_DIR = os.path.join(os.path.dirname(_HARNESS_DIR), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from cli_anything.trading_bot.core.session import Session

# ── Global session (REPL state) ───────────────────────────────────────────────
_session = Session()
_repl_mode = False


# ── Output helpers ────────────────────────────────────────────────────────────

def output(data: dict | list, message: str = ""):
    """Print data as JSON (agent mode) or human-readable table."""
    if _session.json_mode:
        click.echo(json.dumps(data, indent=2, default=str))
    else:
        if message:
            click.echo(message)
        if isinstance(data, list):
            _print_list(data)
        elif isinstance(data, dict):
            _print_dict(data)


def _print_dict(d: dict, indent: int = 0):
    pad = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            click.echo(f"{pad}{k}:")
            _print_dict(v, indent + 1)
        elif isinstance(v, list):
            click.echo(f"{pad}{k}: [{len(v)} items]")
        else:
            click.echo(f"{pad}{k}: {v}")


def _print_list(items: list, indent: int = 0):
    for i, item in enumerate(items):
        click.echo(f"  [{i}]")
        if isinstance(item, dict):
            _print_dict(item, indent + 1)
        else:
            click.echo(f"  {item}")


def handle_error(func):
    """Decorator: catch errors, print as JSON or human-readable."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SystemExit:
            raise
        except Exception as exc:
            err = {"error": str(exc), "type": type(exc).__name__}
            if _session.json_mode:
                click.echo(json.dumps(err))
            else:
                click.echo(f"Error: {exc}", err=True)
                if os.getenv("DEBUG"):
                    traceback.print_exc()
            if not _repl_mode:
                sys.exit(1)

    return wrapper


# ── Root CLI group ─────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.option("--json", "use_json", is_flag=True, help="Output as JSON (agent mode)")
@click.pass_context
def cli(ctx, use_json):
    """
    Kalshi BTC Range trading bot — agent-native CLI.

    Run without a subcommand to enter interactive REPL mode.
    Use --json globally for machine-readable output.

    \b
    Command groups:
      market    Scan markets and generate trading signals
      trade     Place orders and view positions
      backtest  Run historical backtests and parameter sweeps
      bankroll  Manage and inspect the bankroll ledger
      optimize  Run strategy optimizer
      resolve   Auto-resolve settled trades and trigger postmortem
    """
    _session.json_mode = use_json
    ctx.ensure_object(dict)
    ctx.obj["session"] = _session

    if ctx.invoked_subcommand is None:
        ctx.invoke(repl)


# ── market group ──────────────────────────────────────────────────────────────

@cli.group()
def market():
    """Scan Kalshi BTC range markets and generate trading signals."""


@market.command("price")
@handle_error
def market_price():
    """Fetch current BTC spot price."""
    from kalshi_btc_trader import get_btc_price
    price = get_btc_price()
    _session.btc_price = price
    output({"btc_price_usd": price}, f"BTC: ${price:,.2f}")


@market.command("scan")
@click.option("--bankroll", "-b", type=float, default=10.0, help="Bankroll in USD (default: 10)")
@click.option("--min-edge", "-e", type=float, default=0.08, help="Minimum edge threshold (default: 0.08)")
@click.option("--vol", "-v", type=float, default=0.01, help="BTC hourly vol fraction (default: 0.01)")
@handle_error
def market_scan(bankroll, min_edge, vol):
    """Scan all KXBTC markets and show active trading signals."""
    from kalshi_btc_trader import (
        get_btc_price, get_btc_range_markets, generate_signals
    )
    import kalshi_btc_trader as trader

    trader.MIN_EDGE_PCT = min_edge
    trader.BTC_HOURLY_VOL_PCT = vol

    btc_price = get_btc_price()
    markets = get_btc_range_markets()
    signals = generate_signals(markets, btc_price, bankroll)

    _session.btc_price = btc_price
    _session.signals = signals

    result = {
        "btc_price_usd": btc_price,
        "markets_found": len(markets),
        "signals": signals,
    }
    output(result, f"BTC: ${btc_price:,.2f} | {len(markets)} markets | {len(signals)} signal(s)")


@market.command("markets")
@handle_error
def market_markets():
    """List all available KXBTC range markets."""
    from kalshi_btc_trader import get_btc_price, get_btc_range_markets
    price = get_btc_price()
    markets = get_btc_range_markets()
    result = {"btc_price_usd": price, "markets": markets}
    output(result, f"BTC: ${price:,.2f} | {len(markets)} markets")


# ── trade group ───────────────────────────────────────────────────────────────

@cli.group()
def trade():
    """Place orders and inspect open positions."""


@trade.command("positions")
@handle_error
def trade_positions():
    """Show all currently held Kalshi positions."""
    from kalshi_btc_trader import get_open_position_tickers
    tickers = get_open_position_tickers()
    output({"open_positions": sorted(tickers)}, f"Open positions ({len(tickers)}): {', '.join(sorted(tickers)) or 'none'}")


@trade.command("execute")
@click.option("--ticker", "-t", required=True, help="Market ticker (e.g. KXBTC-25MAR1415-B95250)")
@click.option("--side", "-s", required=True, type=click.Choice(["YES", "NO"]), help="Trade side")
@click.option("--contracts", "-c", required=True, type=int, help="Number of contracts")
@click.option("--price", "-p", required=True, type=float, help="Limit price in dollars (e.g. 0.35)")
@click.option("--dry-run", is_flag=True, default=False, help="Preview order without placing it")
@handle_error
def trade_execute(ticker, side, contracts, price, dry_run):
    """Place a limit order on Kalshi."""
    if dry_run:
        result = {
            "dry_run": True,
            "ticker": ticker,
            "side": side,
            "contracts": contracts,
            "limit_price": price,
            "estimated_cost_usd": round(contracts * price, 4),
        }
        output(result, f"DRY RUN: {side} {contracts}x {ticker} @ ${price:.2f}")
        return

    from kalshi_btc_trader import place_order
    result = place_order(ticker, side, contracts, price)
    if result:
        output(result, f"Order placed: {side} {contracts}x {ticker} @ ${price:.2f}")
    else:
        output({"error": "Order failed — check logs"}, "Order failed")


# ── backtest group ────────────────────────────────────────────────────────────

@cli.group()
def backtest():
    """Run walk-forward backtests and parameter sweeps."""


@backtest.command("run")
@click.option("--bankroll", "-b", type=float, default=10.0, help="Starting bankroll (default: 10)")
@click.option("--min-edge", "-e", type=float, default=0.08, help="Min edge threshold")
@click.option("--vol", "-v", type=float, default=0.01, help="BTC hourly vol fraction")
@click.option("--kelly-fraction", "-k", type=float, default=0.25, help="Kelly fraction (default: 0.25)")
@handle_error
def backtest_run(bankroll, min_edge, vol, kelly_fraction):
    """Run a single walk-forward backtest with given parameters."""
    from kxbtc_backtest import run_backtest
    result = run_backtest(
        bankroll=bankroll,
        min_edge_pct=min_edge,
        btc_hourly_vol_pct=vol,
        kelly_frac=kelly_fraction,
    )
    output(result._asdict() if hasattr(result, "_asdict") else vars(result) if hasattr(result, "__dict__") else {"result": str(result)},
           "Backtest complete")


@backtest.command("sweep")
@click.option("--bankroll", "-b", type=float, default=10.0, help="Starting bankroll (default: 10)")
@handle_error
def backtest_sweep(bankroll):
    """Run parameter sweep across edge/vol combinations."""
    from kxbtc_backtest import run_sweep
    results = run_sweep(bankroll=bankroll)
    output({"sweep_results": results, "count": len(results)},
           f"Sweep complete — {len(results)} combinations")


# ── bankroll group ────────────────────────────────────────────────────────────

@cli.group()
def bankroll():
    """Manage and inspect the bankroll and trade ledger."""


@bankroll.command("status")
@handle_error
def bankroll_status():
    """Show current bankroll balance and daily P&L."""
    from bankroll_tracker import cmd_status
    import argparse
    args = argparse.Namespace()
    if _session.json_mode:
        # Capture stdout for JSON wrapping
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            cmd_status(args)
        output({"raw_output": buf.getvalue()})
    else:
        cmd_status(args)


@bankroll.command("init")
@click.option("--amount", "-a", required=True, type=float, help="Initial bankroll in USD")
@handle_error
def bankroll_init(amount):
    """Initialize bankroll with starting balance."""
    from bankroll_tracker import cmd_init
    import argparse
    args = argparse.Namespace(bankroll=amount)
    cmd_init(args)
    output({"initialized": True, "bankroll_usd": amount},
           f"Bankroll initialized: ${amount:.2f}")


@bankroll.command("add-trade")
@click.option("--ticker", required=True, help="Market ticker")
@click.option("--direction", required=True, type=click.Choice(["YES", "NO"]))
@click.option("--contracts", required=True, type=int)
@click.option("--entry-price", required=True, type=float, help="Entry price in dollars")
@click.option("--fair-value", required=True, type=float, help="Model fair value 0-1")
@click.option("--edge", required=True, type=float, help="Edge fraction (e.g. 0.12)")
@click.option("--btc-price", required=True, type=float, help="BTC price at entry")
@handle_error
def bankroll_add_trade(ticker, direction, contracts, entry_price, fair_value, edge, btc_price):
    """Record a new trade in the ledger."""
    from bankroll_tracker import cmd_add_trade
    import argparse
    args = argparse.Namespace(
        ticker=ticker,
        direction=direction,
        contracts=contracts,
        entry_price=entry_price,
        fair_value=fair_value,
        edge=edge,
        btc_price=btc_price,
        range_label=None,
    )
    cmd_add_trade(args)
    output({"recorded": True, "ticker": ticker, "direction": direction},
           f"Trade recorded: {direction} {contracts}x {ticker}")


@bankroll.command("resolve-trade")
@click.option("--trade-id", required=True, help="Trade ID to resolve")
@click.option("--outcome", required=True, type=click.Choice(["WIN", "LOSS"]))
@handle_error
def bankroll_resolve_trade(trade_id, outcome):
    """Mark a trade as WIN or LOSS in the ledger."""
    from bankroll_tracker import cmd_resolve
    import argparse
    args = argparse.Namespace(trade_id=trade_id, outcome=outcome)
    cmd_resolve(args)
    output({"resolved": True, "trade_id": trade_id, "outcome": outcome},
           f"Trade {trade_id}: {outcome}")


@bankroll.command("trades")
@handle_error
def bankroll_trades():
    """List all trades in the ledger."""
    from bankroll_tracker import cmd_trades
    import argparse
    if _session.json_mode:
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            cmd_trades(argparse.Namespace())
        output({"raw_output": buf.getvalue()})
    else:
        cmd_trades(argparse.Namespace())


# ── optimize group ────────────────────────────────────────────────────────────

@cli.group()
def optimize():
    """Run strategy optimizer and inspect config."""


@optimize.command("run")
@click.option("--dry-run", is_flag=True, default=False, help="Compute but don't write config")
@handle_error
def optimize_run(dry_run):
    """Run strategy optimizer (updates strategy_config.json)."""
    from strategy_optimizer import run_optimization
    result = run_optimization(dry_run=dry_run)
    output(result, f"Optimization complete (dry_run={dry_run})")


@optimize.command("config")
@handle_error
def optimize_config():
    """Show current strategy_config.json."""
    from kalshi_btc_trader import load_strategy_config
    cfg = load_strategy_config()
    output(cfg, "Current strategy config:")


@optimize.command("bounds")
@handle_error
def optimize_bounds():
    """Show parameter bounds for the optimizer."""
    from strategy_optimizer import get_config_bounds
    bounds = get_config_bounds()
    output(bounds, "Config bounds:")


# ── resolve group ─────────────────────────────────────────────────────────────

@cli.group()
def resolve():
    """Auto-resolve settled Kalshi trades and run postmortem."""


@resolve.command("run")
@click.option("--dry-run", is_flag=True, default=False, help="Check without writing")
@handle_error
def resolve_run(dry_run):
    """Fetch settled markets, resolve trades, trigger postmortem."""
    from auto_resolver import main as resolver_main
    argv = ["--dry-run"] if dry_run else []
    resolver_main(argv)
    output({"ran": True, "dry_run": dry_run}, "Auto-resolve complete")


# ── status command (top-level) ────────────────────────────────────────────────

@cli.command("status")
@handle_error
def status():
    """Quick status: BTC price + bankroll + open positions."""
    from kalshi_btc_trader import get_btc_price, get_open_position_tickers

    btc_price = get_btc_price()
    positions = get_open_position_tickers()
    _session.btc_price = btc_price

    # Try to read bankroll state
    try:
        from bankroll_tracker import _load_bankroll
        bk = _load_bankroll()
        bankroll_usd = bk.get("balance_usd")
        daily_pnl = bk.get("daily_pnl_usd")
    except Exception:
        bankroll_usd = None
        daily_pnl = None

    result = {
        "btc_price_usd": btc_price,
        "bankroll_usd": bankroll_usd,
        "daily_pnl_usd": daily_pnl,
        "open_positions": sorted(positions),
        "open_position_count": len(positions),
    }
    msg = (
        f"BTC: ${btc_price:,.2f} | "
        f"Bankroll: {'$'+f'{bankroll_usd:.2f}' if bankroll_usd else 'n/a'} | "
        f"Positions: {len(positions)}"
    )
    output(result, msg)


# ── REPL ──────────────────────────────────────────────────────────────────────

@cli.command("repl")
@handle_error
def repl():
    """Start interactive REPL (Read-Eval-Print Loop)."""
    global _repl_mode
    _repl_mode = True

    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        _has_prompt_toolkit = True
    except ImportError:
        _has_prompt_toolkit = False

    click.echo("Trading Bot CLI — REPL mode  (type 'help' or 'exit')")
    click.echo("Commands: market scan/price/markets | trade positions/execute | backtest run/sweep")
    click.echo("          bankroll status/init/add-trade | optimize run/config | resolve run | status")
    click.echo()

    if _has_prompt_toolkit:
        session = PromptSession(history=InMemoryHistory(), auto_suggest=AutoSuggestFromHistory())

    while True:
        try:
            if _has_prompt_toolkit:
                line = session.prompt("trading-bot> ").strip()
            else:
                line = input("trading-bot> ").strip()
        except (EOFError, KeyboardInterrupt):
            click.echo("\nGoodbye.")
            break

        if not line:
            continue
        if line in ("exit", "quit", "q"):
            click.echo("Goodbye.")
            break
        if line in ("help", "?"):
            click.echo(cli.get_help(click.Context(cli)))
            continue

        # Inject --json if session is in JSON mode
        parts = line.split()
        if _session.json_mode and "--json" not in parts:
            parts = ["--json"] + parts

        try:
            cli.main(parts, standalone_mode=False)
        except SystemExit:
            pass
        except Exception as exc:
            click.echo(f"Error: {exc}", err=True)


# ── __main__ ──────────────────────────────────────────────────────────────────

def main():
    cli()


if __name__ == "__main__":
    main()
