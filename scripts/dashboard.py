"""
dashboard.py -- Kalshi BTC Live Trading Dashboard

Tabs:
  1. Live Markets     -- All open KXBTC markets with fair value + edge columns
  2. Active Signals   -- Markets with edge >= 8pp; near-misses if none
  3. Signal History   -- signals_log.csv, edge distribution, win rate, counts
  4. Portfolio        -- Bankroll, P&L, trade history, daily P&L chart
  5. Strategy & Backtest -- Config, backtest results, optimization history

Run: streamlit run scripts/dashboard.py
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from storage_backend import get_storage

# ---------------------------------------------------------------------------
# Optional Plotly (degrade gracefully)
# ---------------------------------------------------------------------------
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KALSHI_BASE    = "https://api.elections.kalshi.com/trade-api/v2"

# Resolve all file paths relative to project root, not CWD
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_storage = get_storage()


def _load_config_params():
    """Load model params from strategy_config.json if available."""
    try:
        cfg = _storage.read_json("backtest/strategy_config.json")
        return (
            cfg.get("btc_hourly_vol", 0.01),
            cfg.get("min_edge_pp", 8.0) / 100.0,
        )
    except Exception:
        pass
    return (0.01, 0.08)


BTC_HOURLY_VOL, MIN_EDGE = _load_config_params()

SIGNAL_LOG    = _PROJECT_ROOT / "trades" / "signals_log.csv"
BANKROLL_FILE = _PROJECT_ROOT / "trades" / "bankroll.json"
TRADES_FILE   = _PROJECT_ROOT / "trades" / "live_trades.parquet"

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Kalshi BTC Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Fair value model (log-normal, inline)
# ---------------------------------------------------------------------------

def compute_fair(btc: float, low: float, high: float, minutes_left: float) -> float:
    if btc <= 0 or minutes_left <= 0:
        return 0.0
    hours_left = max(minutes_left / 60, 1 / 60)
    sigma = BTC_HOURLY_VOL * math.sqrt(hours_left)

    def ncdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    p_high = ncdf(math.log(high / btc) / sigma) if btc < high else 1.0
    p_low  = ncdf(math.log(low  / btc) / sigma) if btc > low  else 0.0
    return max(0.0, min(1.0, p_high - p_low))

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10)
def get_btc_price() -> float:
    try:
        r = requests.get(
            "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            timeout=5,
        )
        return float(r.json()["data"]["amount"])
    except Exception:
        return 0.0


@st.cache_data(ttl=15)
def get_btc_markets() -> list[dict]:
    try:
        r = requests.get(
            f"{KALSHI_BASE}/markets",
            params={"status": "open", "limit": 200, "series_ticker": "KXBTC"},
            headers={"Accept": "application/json"},
            timeout=10,
        )
        raw = r.json().get("markets", [])
    except Exception:
        return []

    out = []
    for m in raw:
        ticker = m.get("ticker", "")
        parts  = ticker.split("-B")
        if len(parts) < 2 or not parts[-1].isdigit():
            continue

        range_low  = int(parts[-1])
        step       = 250
        range_high = range_low + step

        close_str    = m.get("close_time") or m.get("expiration_time") or ""
        minutes_left = 9999.0
        if close_str:
            try:
                ct = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                minutes_left = (ct - datetime.now(timezone.utc)).total_seconds() / 60
            except Exception:
                pass

        yes_ask = float(m.get("yes_ask_dollars") or 0)
        yes_bid = float(m.get("yes_bid_dollars") or 0)
        vol24   = float(m.get("volume_24h_fp")   or 0)

        out.append({
            "ticker":      ticker,
            "range_low":   range_low,
            "range_high":  range_high,
            "range_label": f"${range_low:,}–${range_high:,}",
            "yes_ask":     yes_ask,
            "yes_bid":     yes_bid,
            "mid":         (yes_ask + yes_bid) / 2 if yes_ask and yes_bid else 0.0,
            "volume_24h":  vol24,
            "minutes_left": minutes_left,
            "close_str":   close_str,
        })

    out.sort(key=lambda x: x["range_low"])
    return out

# ---------------------------------------------------------------------------
# Data file loaders
# ---------------------------------------------------------------------------

def load_bankroll() -> tuple[float, float]:
    """Returns (current_bankroll, starting_bankroll)."""
    try:
        data = _storage.read_json("trades/bankroll.json")
        current  = float(data.get("current",  data.get("bankroll", 10.0)))
        starting = float(data.get("starting", data.get("start",    current)))
        return current, starting
    except Exception:
        pass
    return 10.0, 10.0


def load_signal_log() -> pd.DataFrame:
    try:
        rows = _storage.read_csv("trades/signals_log.csv")
        df = pd.DataFrame(rows)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp", ascending=False)
        return df
    except Exception:
        return pd.DataFrame()


def load_live_trades() -> pd.DataFrame:
    try:
        return _storage.read_parquet("trades/live_trades.parquet")
    except Exception:
        return pd.DataFrame()

# ---------------------------------------------------------------------------
# Signal computation (runs once per page load, reused across tabs)
# ---------------------------------------------------------------------------

def compute_signals(markets: list[dict], btc: float) -> list[dict]:
    signals = []
    for mkt in markets:
        fair = mkt.get("fair", 0.0)
        if fair <= 0:
            continue
        in_range = mkt["range_low"] <= btc <= mkt["range_high"]

        # YES underpriced
        if mkt["yes_ask"] > 0:
            edge_yes = fair - mkt["yes_ask"]
            if edge_yes >= MIN_EDGE:
                signals.append({
                    "direction": "YES",
                    "ticker":    mkt["ticker"],
                    "range":     mkt["range_label"],
                    "fair":      fair,
                    "ask":       mkt["yes_ask"],
                    "edge":      edge_yes,
                    "min_left":  mkt["minutes_left"],
                    "in_range":  in_range,
                })

        # NO underpriced (YES overpriced)
        if mkt["yes_bid"] > 0:
            edge_no = mkt["yes_bid"] - fair
            if edge_no >= MIN_EDGE:
                signals.append({
                    "direction": "NO",
                    "ticker":    mkt["ticker"],
                    "range":     mkt["range_label"],
                    "fair":      fair,
                    "ask":       1.0 - mkt["yes_bid"],
                    "edge":      edge_no,
                    "min_left":  mkt["minutes_left"],
                    "in_range":  in_range,
                })

    signals.sort(key=lambda s: s["edge"], reverse=True)
    return signals

# ---------------------------------------------------------------------------
# Header: title + auto-refresh control
# ---------------------------------------------------------------------------

st.title("Kalshi BTC Range Market — Live Dashboard")

hcol1, hcol2 = st.columns([1, 4])
with hcol1:
    auto = st.checkbox("Auto-refresh (30s)", value=True)
with hcol2:
    st.caption(f"Last loaded: {datetime.now().strftime('%H:%M:%S')}")

# ---------------------------------------------------------------------------
# Fetch live data (once per 10-15 s, cached)
# ---------------------------------------------------------------------------

btc     = get_btc_price()
markets = get_btc_markets()

# Annotate each market with fair value
for mkt in markets:
    mkt["fair"] = compute_fair(btc, mkt["range_low"], mkt["range_high"], mkt["minutes_left"])

signals = compute_signals(markets, btc)

current_bankroll, starting_bankroll = load_bankroll()

# ---------------------------------------------------------------------------
# Top metrics row
# ---------------------------------------------------------------------------

m1, m2, m3, m4 = st.columns(4)
m1.metric("BTC / USD",           f"${btc:,.2f}" if btc else "—")
m2.metric("Open KXBTC Markets",  len(markets))
m3.metric(
    "Live Signals",
    len(signals),
    delta="TRADE" if signals else None,
    delta_color="normal",
)
m4.metric("Bankroll", f"${current_bankroll:,.2f}")

st.divider()

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Live Markets",
    "Active Signals",
    "Signal History & Analytics",
    "Portfolio",
    "Strategy & Backtest",
    "Loss Postmortem",
])

# ===========================================================================
# Tab 1 — Live Markets
# ===========================================================================

with tab1:
    if not markets:
        st.warning(
            "No open KXBTC markets found. "
            "Markets may be between sessions — check back after 9 AM EST."
        )
    else:
        rows = []
        for mkt in markets:
            in_range = mkt["range_low"] <= btc <= mkt["range_high"]
            fair     = mkt.get("fair", 0.0)
            edge_yes = (fair - mkt["yes_ask"]) * 100 if mkt["yes_ask"] else None
            edge_no  = (mkt["yes_bid"] - fair) * 100  if mkt["yes_bid"] else None

            # Close time in EST (UTC-5, no DST handling for simplicity)
            close_est = "—"
            if mkt["close_str"]:
                try:
                    ct = datetime.fromisoformat(mkt["close_str"].replace("Z", "+00:00"))
                    ct_est = ct - timedelta(hours=5)
                    close_est = ct_est.strftime("%m/%d %I:%M %p EST")
                except Exception:
                    pass

            def fmt_edge(e: float | None) -> str:
                if e is None:
                    return "—"
                sign = "+" if e >= 0 else ""
                return f"{sign}{e:.1f}"

            rows.append({
                "Range":         ("-> " if in_range else "   ") + mkt["range_label"],
                "Close (EST)":   close_est,
                "Min Left":      f"{mkt['minutes_left']:.0f}m" if mkt["minutes_left"] < 9999 else "—",
                "Fair Value":    f"{fair:.3f}" if fair else "—",
                "YES Ask":       f"${mkt['yes_ask']:.3f}" if mkt["yes_ask"] else "—",
                "YES Bid":       f"${mkt['yes_bid']:.3f}" if mkt["yes_bid"] else "—",
                "Edge YES (pp)": fmt_edge(edge_yes),
                "Edge NO (pp)":  fmt_edge(edge_no),
                "Vol 24h":       f"${mkt['volume_24h']:,.0f}",
                "In Range":      "YES" if in_range else "",
                # Keep raw values for styling (will be dropped from display)
                "_in_range":     in_range,
                "_edge_yes":     edge_yes,
                "_edge_no":      edge_no,
            })

        df_markets = pd.DataFrame(rows)
        display_cols = [c for c in df_markets.columns if not c.startswith("_")]

        def highlight_edge(val: object) -> str:
            if isinstance(val, str) and val.startswith("+"):
                try:
                    if float(val) >= 8.0:
                        return "color: #00ff88; font-weight: bold"
                except ValueError:
                    pass
            return ""

        st.dataframe(
            df_markets[display_cols].style
            .apply(
                lambda row: (
                    ["background-color: #1a3a1a; color: #00ff88"] * len(row)
                    if df_markets.loc[row.name, "_in_range"]
                    else [""] * len(row)
                ),
                axis=1,
            )
            .map(highlight_edge, subset=["Edge YES (pp)", "Edge NO (pp)"]),
            width="stretch",
            height=min(700, 55 + len(rows) * 38),
        )

        st.caption(
            f"-> = BTC (${btc:,.2f}) is currently in this range bucket. "
            "Edges >= 8pp highlighted in green."
        )

# ===========================================================================
# Tab 2 — Active Signals
# ===========================================================================

with tab2:
    if not signals:
        st.info("No signals above the 8pp edge threshold right now.")

        # Near-misses
        if markets:
            candidates = []
            for mkt in markets:
                fair = mkt.get("fair", 0.0)
                if fair and mkt["yes_ask"]:
                    e = (fair - mkt["yes_ask"]) * 100
                    candidates.append({
                        "Edge (pp)": f"{e:.1f}",
                        "Direction": "YES",
                        "Ticker":    mkt["ticker"],
                        "Range":     mkt["range_label"],
                        "Fair":      f"{fair:.3f}",
                        "Ask":       f"${mkt['yes_ask']:.3f}",
                        "_edge_raw": e,
                    })
                if fair and mkt["yes_bid"]:
                    e = (mkt["yes_bid"] - fair) * 100
                    candidates.append({
                        "Edge (pp)": f"{e:.1f}",
                        "Direction": "NO",
                        "Ticker":    mkt["ticker"],
                        "Range":     mkt["range_label"],
                        "Fair":      f"{fair:.3f}",
                        "Ask":       f"${1 - mkt['yes_bid']:.3f}",
                        "_edge_raw": e,
                    })

            candidates.sort(key=lambda x: x["_edge_raw"], reverse=True)
            top5 = [
                {k: v for k, v in c.items() if not k.startswith("_")}
                for c in candidates[:5]
            ]

            if top5:
                st.subheader("Near-Misses (closest to 8pp threshold)")
                st.dataframe(pd.DataFrame(top5), width="stretch")
    else:
        st.subheader(f"{len(signals)} Active Signal(s)")
        for sig in signals:
            min_left = sig["min_left"]
            if min_left < 30:
                urgency_label = "URGENT (<30 min)"
                urgency_color = "#ff4444"
            elif min_left < 120:
                urgency_label = "ACTIVE (<2 hrs)"
                urgency_color = "#ffaa00"
            else:
                urgency_label = "OPEN"
                urgency_color = "#00aaff"

            st.markdown(
                f"<span style='color:{urgency_color}; font-weight:bold; font-size:1.1em'>"
                f"[{urgency_label}] {sig['direction']} — {sig['ticker']}</span>",
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Direction",    sig["direction"])
            c2.metric("Edge",         f"+{sig['edge'] * 100:.1f}pp")
            c3.metric("Fair Value",   f"{sig['fair']:.3f}")
            c4.metric("Market Ask",   f"${sig['ask']:.3f}")
            c5.metric("Min to Close", f"{min_left:.0f}m" if min_left < 9999 else "—")

            st.code(sig["ticker"], language=None)
            st.caption(
                f"Range: {sig['range']} — "
                f"Search this ticker on Kalshi and buy {sig['direction']}"
            )
            st.divider()

# ===========================================================================
# Tab 3 — Signal History & Analytics
# ===========================================================================

with tab3:
    df_log = load_signal_log()

    if df_log.empty:
        st.info(
            "No signals logged yet. "
            "Signal log will appear at: trades/signals_log.csv"
        )
    else:
        # ---- Summary counts ----
        now_utc = datetime.now(timezone.utc)
        today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start  = today_start - timedelta(days=today_start.weekday())

        total_signals = len(df_log)
        today_signals = 0
        week_signals  = 0

        if "timestamp" in df_log.columns:
            ts = pd.to_datetime(df_log["timestamp"], utc=True, errors="coerce")
            today_signals = int((ts >= today_start).sum())
            week_signals  = int((ts >= week_start).sum())

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Signals Today",     today_signals)
        sc2.metric("Signals This Week", week_signals)
        sc3.metric("All-Time Signals",  total_signals)

        st.divider()

        # ---- Win rate (if outcome column present) ----
        outcome_col = next(
            (c for c in df_log.columns if "outcome" in c.lower() or "win" in c.lower()),
            None,
        )
        if outcome_col:
            win_rate = df_log[outcome_col].astype(float).mean()
            st.metric("Win Rate", f"{win_rate:.1%}", help=f"Column: {outcome_col}")
            st.divider()

        # ---- Full log table ----
        st.subheader("Full Signal Log")
        st.dataframe(df_log, width="stretch")

        # ---- Edge distribution chart ----
        edge_col = next(
            (c for c in df_log.columns if "edge" in c.lower()),
            None,
        )
        if edge_col:
            st.subheader("Edge Distribution")
            edge_vals = pd.to_numeric(df_log[edge_col], errors="coerce").dropna()
            if len(edge_vals) > 0:
                if HAS_PLOTLY:
                    fig = px.histogram(
                        x=edge_vals * 100 if edge_vals.max() <= 1.0 else edge_vals,
                        nbins=30,
                        title="Edge Distribution (pp)",
                        labels={"x": "Edge (pp)"},
                        color_discrete_sequence=["steelblue"],
                    )
                    fig.add_vline(
                        x=MIN_EDGE * 100,
                        line_dash="dash",
                        line_color="green",
                        annotation_text=f"Min edge ({MIN_EDGE * 100:.0f}pp)",
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    # Fallback: simple bar chart via Streamlit
                    edge_display = edge_vals * 100 if edge_vals.max() <= 1.0 else edge_vals
                    st.bar_chart(edge_display)

# ===========================================================================
# Tab 4 — Portfolio
# ===========================================================================

with tab4:
    # ---- Bankroll summary ----
    pnl        = current_bankroll - starting_bankroll
    pnl_pct    = pnl / starting_bankroll if starting_bankroll else 0.0

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Current Bankroll",  f"${current_bankroll:,.2f}")
    p2.metric("Starting Bankroll", f"${starting_bankroll:,.2f}")
    p3.metric(
        "P&L",
        f"${pnl:+,.2f}",
        delta_color="normal" if pnl >= 0 else "inverse",
    )
    p4.metric(
        "P&L %",
        f"{pnl_pct:+.2%}",
        delta_color="normal" if pnl_pct >= 0 else "inverse",
    )

    if not BANKROLL_FILE.exists():
        st.caption("Bankroll file not found — showing $10.00 default. Create trades/bankroll.json to persist balance.")

    st.divider()

    # ---- Trade history ----
    trades_df = load_live_trades()

    if trades_df.empty:
        st.info(
            "No trade history yet. "
            "Trades will appear here once stored in trades/live_trades.parquet"
        )
    else:
        st.subheader("Trade History")

        # Show relevant columns if they exist
        preferred_cols = [
            "ticker", "direction", "range", "entry_price", "exit_price",
            "stake", "pnl", "pnl_pct", "status", "entered_at", "closed_at",
        ]
        display_cols_t = [c for c in preferred_cols if c in trades_df.columns]
        if not display_cols_t:
            display_cols_t = list(trades_df.columns)

        st.dataframe(trades_df[display_cols_t], width="stretch")

        # ---- Daily P&L chart ----
        date_col = next(
            (c for c in trades_df.columns if "date" in c.lower() or "closed" in c.lower() or "entered" in c.lower()),
            None,
        )
        pnl_col  = next(
            (c for c in trades_df.columns if c.lower() in ("pnl", "pnl_pct", "profit")),
            None,
        )

        if date_col and pnl_col and HAS_PLOTLY:
            try:
                chart_df = trades_df[[date_col, pnl_col]].copy()
                chart_df[date_col] = pd.to_datetime(chart_df[date_col], errors="coerce")
                chart_df = chart_df.dropna()
                chart_df["date"] = chart_df[date_col].dt.date
                daily = chart_df.groupby("date")[pnl_col].sum().reset_index()
                daily.columns = ["date", "daily_pnl"]
                daily = daily.sort_values("date")
                daily["cumulative_pnl"] = daily["daily_pnl"].cumsum()

                st.subheader("Daily P&L")
                fig = go.Figure()
                fig.add_bar(
                    x=daily["date"],
                    y=daily["daily_pnl"],
                    name="Daily P&L",
                    marker_color=[
                        "#00cc66" if v >= 0 else "#ff4444"
                        for v in daily["daily_pnl"]
                    ],
                )
                fig.add_scatter(
                    x=daily["date"],
                    y=daily["cumulative_pnl"],
                    mode="lines+markers",
                    name="Cumulative P&L",
                    yaxis="y2",
                    line=dict(color="royalblue"),
                )
                fig.update_layout(
                    title="Daily & Cumulative P&L",
                    xaxis_title="Date",
                    yaxis_title="Daily P&L",
                    yaxis2=dict(title="Cumulative P&L", overlaying="y", side="right"),
                    legend=dict(x=0, y=1),
                )
                st.plotly_chart(fig, width="stretch")
            except Exception:
                pass

        # ---- Per-market win/loss breakdown ----
        market_col = next(
            (c for c in trades_df.columns if "ticker" in c.lower() or "market" in c.lower()),
            None,
        )
        if market_col and pnl_col and HAS_PLOTLY:
            try:
                mkt_df = trades_df[[market_col, pnl_col]].copy()
                mkt_df[pnl_col] = pd.to_numeric(mkt_df[pnl_col], errors="coerce")
                mkt_summary = (
                    mkt_df.groupby(market_col)[pnl_col]
                    .agg(total_pnl="sum", trade_count="count")
                    .reset_index()
                    .sort_values("total_pnl", ascending=False)
                )
                st.subheader("Per-Market P&L Breakdown")
                fig2 = px.bar(
                    mkt_summary,
                    x=market_col,
                    y="total_pnl",
                    color="total_pnl",
                    color_continuous_scale="RdYlGn",
                    title="Total P&L by Market",
                    labels={market_col: "Ticker", "total_pnl": "Total P&L"},
                )
                st.plotly_chart(fig2, width="stretch")
            except Exception:
                pass

# ===========================================================================
# Tab 5 — Strategy & Backtest
# ===========================================================================

with tab5:
    st.subheader("Current Strategy Configuration")
    try:
        cfg = _storage.read_json("backtest/strategy_config.json")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vol Estimate", f"{cfg.get('btc_hourly_vol', 0.01):.4f}")
        c2.metric("Min Edge", f"{cfg.get('min_edge_pp', 8.0):.1f}pp")
        c3.metric("Kelly Fraction", f"{cfg.get('fractional_kelly', 0.25):.2f}x")
        c4.metric("Iteration", cfg.get("iteration", 0))

        avoid = cfg.get("avoid_hours", [])
        if avoid:
            st.warning(f"Avoiding hours (UTC): {avoid}")

        notes = cfg.get("notes", "")
        if notes:
            st.caption(f"Last change: {notes}")
    except Exception:
        st.info("No strategy_config.json yet. Run: python scripts/strategy_optimizer.py")

    st.divider()
    st.subheader("Latest Backtest Results")
    try:
        bt_data = _storage.read_json("backtest/kxbtc_backtest_results.json")

        for label_key in [("in_sample", "In-Sample (70%)"), ("out_of_sample", "Out-of-Sample (30%)")]:
            key, label = label_key
            seg = bt_data.get(key, {})
            if seg:
                st.markdown(f"**{label}**")
                bc1, bc2, bc3, bc4, bc5 = st.columns(5)
                bc1.metric("Trades", seg.get("n_trades", 0))
                bc2.metric("Win Rate", f"{seg.get('win_rate', 0):.1f}%")
                bc3.metric("Return", f"{seg.get('return_pct', 0):+.1f}%")
                bc4.metric("Sharpe", f"{seg.get('sharpe', 0):.2f}")
                bc5.metric("Max DD", f"{seg.get('max_drawdown_pct', 0):.1f}%")

        # Calibration table if available
        for key in ["in_sample", "out_of_sample"]:
            seg = bt_data.get(key, {})
            cal = seg.get("calibration", {})
            if cal:
                st.markdown(f"**Calibration ({key.replace('_', ' ').title()})**")
                cal_rows = []
                for bucket, data in cal.items():
                    cal_rows.append({
                        "FV Bucket": bucket,
                        "N Trades": data.get("n", 0),
                        "Avg Fair Value": f"{data.get('avg_fair_value_pct', 0):.1f}%",
                        "Actual Win Rate": f"{data.get('actual_win_rate_pct', 0):.1f}%",
                    })
                if cal_rows:
                    st.dataframe(pd.DataFrame(cal_rows), width="stretch")
    except Exception:
        st.info("No backtest results yet. Run: python scripts/kxbtc_backtest.py --bankroll 10")

    st.divider()
    st.subheader("Optimization History")
    try:
        opt_df = _storage.read_csv("backtest/optimization_log.csv")
        st.dataframe(opt_df, width="stretch")
    except Exception:
        st.info("No optimization history yet.")

# ===========================================================================
# Tab 6 — Loss Postmortem
# ===========================================================================

with tab6:
    st.subheader("Loss Postmortem — Round Table Assessment")

    try:
        _pm = _storage.read_json("backtest/loss_postmortem.json")

        # Summary header
        _pm_ts    = _pm.get("timestamp", "unknown")
        _pm_n     = _pm.get("n_losses", 0)
        _pm_top   = _pm.get("top_finding", "N/A")
        _pm_action = _pm.get("action_taken", "N/A")

        col_a, col_b = st.columns(2)
        col_a.metric("Losses Analyzed", _pm_n)
        col_b.metric("Report Date", _pm_ts[:10] if len(_pm_ts) >= 10 else _pm_ts)

        st.markdown(f"**Top Finding:** {_pm_top}")
        st.markdown(f"**Action Taken:** {_pm_action}")

        st.divider()

        # 5 specialist expandable sections
        _SPECIALIST_LABELS = [
            ("vol_analyst",     "Vol Analyst",         "Volatility analysis of losing trades"),
            ("timing_analyst",  "Timing Analyst",      "Entry timing and time-bucket analysis"),
            ("market_intel",    "Market Intelligence", "Fair value vs. market pricing on losses"),
            ("pattern_matcher", "Pattern Matcher",     "Loss clusters by strategy and direction"),
            ("counterfactual",  "Counterfactual",      "Wrong direction and missed opportunity analysis"),
        ]

        for _key, _label, _desc in _SPECIALIST_LABELS:
            _spec = _pm.get(_key, {})
            if not _spec:
                continue
            with st.expander(f"{_label} — {_spec.get('summary', _desc)}", expanded=False):
                _display = {k: v for k, v in _spec.items() if k not in ("specialist", "summary")}
                for _k, _v in _display.items():
                    if isinstance(_v, list):
                        st.markdown(f"**{_k}:**")
                        if _v:
                            st.dataframe(pd.DataFrame(_v))
                        else:
                            st.caption("(empty)")
                    else:
                        st.markdown(f"**{_k}:** {_v}")

        st.divider()

    except FileNotFoundError:
        st.info("No postmortem data yet. Losses must be resolved first.")
    except Exception as _pm_err:
        st.error(f"Could not load postmortem report: {_pm_err}")

    # Postmortem history log
    st.subheader("Postmortem History")
    try:
        _pm_log = _storage.read_csv("backtest/postmortem_log.csv")
        st.dataframe(_pm_log, use_container_width=True)
    except Exception:
        st.caption("No postmortem history yet.")

# ---------------------------------------------------------------------------
# Auto-refresh (sleep then st.rerun — blocks the Python thread for 30s)
# ---------------------------------------------------------------------------
if auto:
    time.sleep(30)
    st.rerun()
