"""
dashboard.py -- Streamlit 5-tab trading dashboard

Tabs:
  1. Trade Lifecycle   -- Open positions, pending signals, trade history
  2. Strategy Perf     -- Per-strategy P&L, Thompson sampling weights, Kelly returns by fold
  3. Calibration       -- Brier score history, calibration curve, isotonic correction
  4. Correlation Map   -- Portfolio correlation heatmap + arbitrage signals
  5. Live Positions    -- Real-time exposure, daily P&L, risk limits gauge

Run: streamlit run scripts/dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---- Optional imports (degrade gracefully if data missing) ----
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

DATA_DIR = Path("data")
BACKTEST_DIR = Path("backtest")
TRADES_DIR = Path("trades")

RISK_LIMITS = {
    "per_market_pct": 0.05,
    "category_pct": 0.15,
    "total_exposure_pct": 0.40,
    "daily_loss_stop_pct": 0.03,
}

st.set_page_config(
    page_title="Prediction Market Bot",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loaders (cached for performance)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_live_trades() -> pd.DataFrame:
    path = TRADES_DIR / "live_trades.parquet"
    if not path.exists():
        return pd.DataFrame(columns=[
            "market_id", "platform", "category", "direction", "stake_pct",
            "entry_price", "current_price", "pnl_pct", "status",
            "entered_at", "signal_source",
        ])
    return pd.read_parquet(path)


@st.cache_data(ttl=300)
def load_backtest_summary() -> Optional[Dict]:
    path = BACKTEST_DIR / "backtest_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_fold_results() -> pd.DataFrame:
    path = BACKTEST_DIR / "fold_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_mc_returns() -> Optional[np.ndarray]:
    path = BACKTEST_DIR / "monte_carlo_returns.npy"
    if not path.exists():
        return None
    return np.load(path)


@st.cache_data(ttl=300)
def load_correlations() -> pd.DataFrame:
    path = DATA_DIR / "market_correlations.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=300)
def load_features() -> pd.DataFrame:
    path = DATA_DIR / "features" / "market_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    # Return only a sample for display performance
    return df.tail(500) if len(df) > 500 else df


def _placeholder_chart(message: str) -> None:
    st.info(message)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    st.sidebar.title("Prediction Market Bot")
    st.sidebar.markdown("---")

    summary = load_backtest_summary()
    if summary:
        st.sidebar.metric("Holdout AUC", f"{summary['holdout']['auc']:.4f}")
        st.sidebar.metric("Holdout Brier", f"{summary['holdout']['brier_score']:.4f}")
        holdout_kelly = summary['holdout']['kelly_return']
        st.sidebar.metric("Holdout Kelly Return", f"{holdout_kelly:+.1%}")
        mc = summary.get("monte_carlo", {})
        st.sidebar.metric("MC Median Return", f"{mc.get('median_return', 0):+.1%}")
        st.sidebar.metric("Sharpe (MC)", f"{mc.get('sharpe_median', 0):.2f}")
        prob_ruin = mc.get("prob_ruin", 0)
        st.sidebar.metric("Prob Ruin", f"{prob_ruin:.1%}",
                          delta_color="inverse" if prob_ruin > 0.05 else "normal")
    else:
        st.sidebar.warning("No backtest data. Run backtest_runner.py first.")

    st.sidebar.markdown("---")
    if st.sidebar.button("Refresh All Data"):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Tab 1: Trade Lifecycle
# ---------------------------------------------------------------------------

def tab_trade_lifecycle() -> None:
    st.header("Trade Lifecycle")

    trades = load_live_trades()

    if trades.empty:
        st.info("No trades found. Trades will appear here once live trading starts.")
        st.markdown(
            "**Data location**: `trades/live_trades.parquet`\n\n"
            "Expected columns: `market_id, platform, category, direction, "
            "stake_pct, entry_price, current_price, pnl_pct, status, "
            "entered_at, signal_source`"
        )
        return

    col1, col2, col3, col4 = st.columns(4)
    open_trades = trades[trades["status"] == "open"] if "status" in trades.columns else trades
    col1.metric("Open Positions", len(open_trades))
    if "pnl_pct" in trades.columns:
        total_pnl = trades[trades["status"] == "open"]["pnl_pct"].sum() if "status" in trades.columns else 0
        col2.metric("Unrealized PnL", f"{total_pnl:+.2%}")
    if "stake_pct" in trades.columns:
        total_exposure = open_trades["stake_pct"].sum() if not open_trades.empty else 0
        limit = RISK_LIMITS["total_exposure_pct"]
        col3.metric(
            "Total Exposure",
            f"{total_exposure:.1%}",
            delta=f"{total_exposure - limit:.1%} vs {limit:.0%} limit",
            delta_color="inverse" if total_exposure > limit else "normal",
        )
    col4.metric("Platforms", trades["platform"].nunique() if "platform" in trades.columns else 0)

    st.subheader("Open Positions")
    if not open_trades.empty:
        display_cols = [c for c in [
            "market_id", "platform", "category", "direction",
            "stake_pct", "entry_price", "current_price", "pnl_pct", "signal_source"
        ] if c in open_trades.columns]
        st.dataframe(open_trades[display_cols], use_container_width=True)
    else:
        st.info("No open positions.")

    st.subheader("Trade History")
    closed = trades[trades["status"] == "closed"] if "status" in trades.columns else pd.DataFrame()
    if not closed.empty:
        st.dataframe(closed.tail(50), use_container_width=True)

        if "pnl_pct" in closed.columns and "entered_at" in closed.columns and HAS_PLOTLY:
            closed_sorted = closed.sort_values("entered_at")
            closed_sorted["cumulative_pnl"] = closed_sorted["pnl_pct"].cumsum()
            fig = px.line(
                closed_sorted, x="entered_at", y="cumulative_pnl",
                title="Cumulative PnL (Closed Trades)",
                labels={"cumulative_pnl": "Cumulative PnL", "entered_at": "Date"},
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No closed trades yet.")


# ---------------------------------------------------------------------------
# Tab 2: Strategy Performance
# ---------------------------------------------------------------------------

def tab_strategy_performance() -> None:
    st.header("Strategy Performance")

    fold_df = load_fold_results()
    summary = load_backtest_summary()
    mc_returns = load_mc_returns()

    if fold_df.empty:
        st.info("No fold results. Run `python scripts/backtest_runner.py` first.")
        return

    # Fold metrics summary
    st.subheader("Walk-Forward CV Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean AUC", f"{fold_df['auc'].mean():.4f}", f"+/-{fold_df['auc'].std():.4f}")
    col2.metric("Mean Brier", f"{fold_df['brier_score'].mean():.4f}")
    col3.metric("Mean Kelly Return", f"{fold_df['kelly_return'].mean():+.2%}")

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=fold_df["fold"], y=fold_df["kelly_return"],
                             name="Kelly Return", marker_color="green"))
        fig.add_trace(go.Scatter(x=fold_df["fold"], y=fold_df["auc"],
                                 name="AUC", yaxis="y2", mode="lines+markers"))
        fig.update_layout(
            title="Fold Results: Kelly Return + AUC",
            xaxis_title="Fold",
            yaxis_title="Kelly Return",
            yaxis2=dict(title="AUC", overlaying="y", side="right"),
            legend=dict(x=0, y=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(fold_df, use_container_width=True)

    # Holdout results
    if summary:
        st.subheader("Holdout Evaluation (Sealed Test Set)")
        h = summary["holdout"]
        hcol1, hcol2, hcol3, hcol4 = st.columns(4)
        hcol1.metric("AUC", f"{h['auc']:.4f}")
        hcol2.metric("Brier Score", f"{h['brier_score']:.4f}")
        hcol3.metric("Kelly Return", f"{h['kelly_return']:+.2%}")
        hcol4.metric("Trades", h['n_trades'])

        mc = summary.get("monte_carlo", {})
        st.subheader("Monte Carlo Simulation")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Median Return", f"{mc.get('median_return', 0):+.2%}")
        mc2.metric("P5 / P95", f"{mc.get('p5_return', 0):+.2%} / {mc.get('p95_return', 0):+.2%}")
        mc3.metric("Sharpe", f"{mc.get('sharpe_median', 0):.2f}")
        mc4.metric("Prob Ruin", f"{mc.get('prob_ruin', 0):.1%}",
                   delta_color="inverse" if mc.get('prob_ruin', 0) > 0.05 else "normal")

    # MC return distribution
    if mc_returns is not None and len(mc_returns) > 0 and HAS_PLOTLY:
        fig = px.histogram(
            x=mc_returns, nbins=50,
            title=f"Monte Carlo Return Distribution ({len(mc_returns)} trials)",
            labels={"x": "Return"},
            color_discrete_sequence=["steelblue"],
        )
        fig.add_vline(x=np.median(mc_returns), line_dash="dash",
                      annotation_text=f"Median: {np.median(mc_returns):+.2%}")
        fig.add_vline(x=np.percentile(mc_returns, 5), line_color="red", line_dash="dot",
                      annotation_text=f"P5: {np.percentile(mc_returns, 5):+.2%}")
        st.plotly_chart(fig, use_container_width=True)

    # Thompson sampling weights (simulated if no live data)
    st.subheader("Strategy Weights (Thompson Sampling)")
    strategies = ["Sentiment", "Momentum", "ML (XGBoost)", "LLM", "Ensemble"]
    weights_path = Path("data/thompson_weights.json")
    if weights_path.exists():
        with open(weights_path) as f:
            weights_data = json.load(f)
        alphas = [weights_data.get(s, {}).get("alpha", 1) for s in strategies]
        betas = [weights_data.get(s, {}).get("beta", 1) for s in strategies]
        means = [a / (a + b) for a, b in zip(alphas, betas)]
    else:
        # Placeholder uniform weights
        means = [0.2] * 5

    if HAS_PLOTLY:
        fig = px.bar(
            x=strategies, y=means,
            title="Current Thompson Sampling Weights (Beta posterior means)",
            labels={"x": "Strategy", "y": "Estimated Win Rate"},
            color=means, color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Calibration
# ---------------------------------------------------------------------------

def tab_calibration() -> None:
    st.header("Calibration")
    st.markdown("Brier score tracking and probability calibration curves.")

    cal_path = Path("data/calibration_history.parquet")
    if cal_path.exists():
        cal_df = pd.read_parquet(cal_path)

        if "date" in cal_df.columns and "brier_score" in cal_df.columns and HAS_PLOTLY:
            fig = px.line(
                cal_df, x="date", y="brier_score",
                title="Brier Score Over Time (lower is better)",
                labels={"brier_score": "Brier Score"},
            )
            fig.add_hline(y=0.25, line_dash="dash", line_color="red",
                          annotation_text="Coin-flip baseline (0.25)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No calibration history yet. Data will appear after live trading starts.")

    # Calibration curve from features
    # Use price_at_T1d as predicted probability (written by collect_polymarket.py);
    # fall back to price_at_T7d if T1d is absent.
    features_df = load_features()
    prob_col: Optional[str] = None
    if not features_df.empty and "outcome_label" in features_df.columns:
        for candidate in ("price_at_T1d", "price_at_T7d"):
            if candidate in features_df.columns:
                col_valid = features_df[candidate].dropna()
                if len(col_valid) >= 10:
                    prob_col = candidate
                    break

    if not features_df.empty and "outcome_label" in features_df.columns and prob_col is not None:
        st.subheader(f"Calibration Curve: {prob_col} vs. Actual Outcome")

        plot_df = features_df[[prob_col, "outcome_label"]].dropna().copy()
        bins = np.linspace(0, 1, 11)
        plot_df["prob_bin"] = pd.cut(plot_df[prob_col], bins=bins, labels=False)
        cal_data = plot_df.groupby("prob_bin").agg(
            predicted_mean=(prob_col, "mean"),
            actual_rate=("outcome_label", "mean"),
            count=(prob_col, "count"),
        ).reset_index()

        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cal_data["predicted_mean"], y=cal_data["actual_rate"],
                mode="markers+lines", name="Calibration",
                marker=dict(size=cal_data["count"].clip(upper=200) / 10 + 5),
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Perfect calibration", line=dict(dash="dash", color="gray"),
            ))
            fig.update_layout(
                title="Calibration Curve",
                xaxis_title="Predicted Probability",
                yaxis_title="Actual Frequency",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(cal_data, use_container_width=True)
    else:
        _placeholder_chart(
            "Calibration curve requires features data. Run collect_polymarket.py then build_base_rates.py first."
        )

    # Isotonic correction info
    iso_path = Path("data/isotonic_correction.json")
    if iso_path.exists():
        with open(iso_path) as f:
            iso_data = json.load(f)
        st.subheader("Isotonic Regression Correction")
        iso_df = pd.DataFrame({
            "raw_prob": iso_data.get("iso_x", []),
            "corrected_prob": iso_data.get("iso_y", []),
        })
        if not iso_df.empty and HAS_PLOTLY:
            fig = px.line(
                iso_df, x="raw_prob", y="corrected_prob",
                title="Isotonic Calibration Map (raw -> corrected)",
            )
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     name="Identity", line=dict(dash="dash")))
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: Correlation Map
# ---------------------------------------------------------------------------

def tab_correlation_map() -> None:
    st.header("Portfolio Correlation Map")

    corr_df = load_correlations()
    trades = load_live_trades()

    if corr_df.empty:
        st.info(
            "No correlation data. Run:\n"
            "```\npython scripts/correlation_engine.py\n```"
        )
        return

    st.metric("Total Correlation Edges", len(corr_df))
    st.metric("Unique Markets in Graph", corr_df["market_a"].nunique() + corr_df["market_b"].nunique())

    # Correlation distribution
    if HAS_PLOTLY:
        fig = px.histogram(
            corr_df, x="correlation", nbins=30,
            title="Distribution of Market Correlations",
            labels={"correlation": "Correlation Coefficient"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Top correlated pairs
    st.subheader("Top 20 Most Correlated Pairs")
    top_pairs = corr_df.nlargest(20, "correlation")
    st.dataframe(top_pairs, use_container_width=True)

    # Portfolio correlation heatmap (open positions only)
    if not trades.empty and "market_id" in trades.columns:
        open_ids = trades[trades["status"] == "open"]["market_id"].tolist() if "status" in trades.columns else trades["market_id"].tolist()
        if len(open_ids) >= 2:
            st.subheader("Open Portfolio Correlation Heatmap")
            # Build correlation matrix from edges
            n = len(open_ids)
            matrix = np.eye(n)
            id_to_idx = {mid: i for i, mid in enumerate(open_ids)}

            for _, row in corr_df.iterrows():
                a, b, c = row["market_a"], row["market_b"], row["correlation"]
                if a in id_to_idx and b in id_to_idx:
                    i, j = id_to_idx[a], id_to_idx[b]
                    matrix[i, j] = c
                    matrix[j, i] = c

            if HAS_PLOTLY:
                labels = [mid[:20] for mid in open_ids]  # Truncate for display
                fig = px.imshow(
                    matrix, x=labels, y=labels,
                    color_continuous_scale="RdBu_r",
                    title="Portfolio Correlation Matrix",
                    zmin=-1, zmax=1,
                )
                st.plotly_chart(fig, use_container_width=True)

    # Arbitrage signals section
    st.subheader("Arbitrage Signal Detection")
    st.markdown(
        "Markets with high correlation but inconsistent prices may present arbitrage opportunities. "
        "Run `correlation_engine.py` with live prices to surface real-time signals."
    )
    arb_path = Path("data/arbitrage_signals.json")
    if arb_path.exists():
        with open(arb_path) as f:
            signals = json.load(f)
        if signals:
            arb_df = pd.DataFrame(signals)
            st.dataframe(arb_df.nlargest(10, "signal_strength"), use_container_width=True)
        else:
            st.success("No significant arbitrage signals detected.")
    else:
        st.info("No arbitrage signal file found. Will appear when live prices are available.")


# ---------------------------------------------------------------------------
# Tab 5: Live Positions
# ---------------------------------------------------------------------------

def tab_live_positions() -> None:
    st.header("Live Positions & Risk Dashboard")

    trades = load_live_trades()

    # Risk gauges
    col1, col2, col3 = st.columns(3)
    open_trades = trades[trades["status"] == "open"] if ("status" in trades.columns and not trades.empty) else trades

    total_exposure = open_trades["stake_pct"].sum() if ("stake_pct" in open_trades.columns and not open_trades.empty) else 0.0
    daily_pnl = open_trades["pnl_pct"].sum() if ("pnl_pct" in open_trades.columns and not open_trades.empty) else 0.0

    # Total exposure gauge
    exposure_limit = RISK_LIMITS["total_exposure_pct"]
    col1.metric(
        "Total Exposure",
        f"{total_exposure:.1%}",
        f"Limit: {exposure_limit:.0%}",
        delta_color="inverse" if total_exposure > exposure_limit * 0.9 else "normal",
    )

    # Daily PnL
    daily_loss_limit = -RISK_LIMITS["daily_loss_stop_pct"]
    col2.metric(
        "Daily PnL",
        f"{daily_pnl:+.2%}",
        "STOP TRIGGERED" if daily_pnl < daily_loss_limit else "Within limits",
        delta_color="inverse" if daily_pnl < daily_loss_limit else "normal",
    )

    # Number of open positions
    col3.metric("Open Positions", len(open_trades))

    # Category exposure breakdown
    if not open_trades.empty and "category" in open_trades.columns and "stake_pct" in open_trades.columns:
        st.subheader("Category Exposure vs. Limits")
        cat_exposure = open_trades.groupby("category")["stake_pct"].sum().reset_index()
        cat_exposure.columns = ["category", "exposure"]
        cat_exposure["limit"] = RISK_LIMITS["category_pct"]
        cat_exposure["pct_of_limit"] = cat_exposure["exposure"] / cat_exposure["limit"]

        if HAS_PLOTLY:
            fig = px.bar(
                cat_exposure,
                x="category", y="exposure",
                title="Exposure by Category",
                color="pct_of_limit",
                color_continuous_scale="RdYlGn_r",
                labels={"exposure": "Exposure (fraction of bankroll)"},
            )
            fig.add_hline(
                y=RISK_LIMITS["category_pct"],
                line_dash="dash", line_color="red",
                annotation_text=f"Category limit ({RISK_LIMITS['category_pct']:.0%})",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(cat_exposure, use_container_width=True)

    # Platform breakdown
    if not open_trades.empty and "platform" in open_trades.columns:
        st.subheader("Exposure by Platform")
        platform_exp = open_trades.groupby("platform")["stake_pct"].sum() if "stake_pct" in open_trades.columns else open_trades.groupby("platform").size()
        st.bar_chart(platform_exp)

    # Full positions table
    st.subheader("All Open Positions")
    if not open_trades.empty:
        st.dataframe(open_trades, use_container_width=True)
    else:
        st.info("No open positions. System is in standby or waiting for signals above minimum edge.")

    # Risk rules reference
    with st.expander("Risk Rules Reference"):
        st.markdown(
            f"""
| Rule | Limit |
|------|-------|
| Per-market max | {RISK_LIMITS['per_market_pct']:.0%} of bankroll |
| Category max | {RISK_LIMITS['category_pct']:.0%} of bankroll |
| Total exposure | {RISK_LIMITS['total_exposure_pct']:.0%} max (correlation-adjusted) |
| Daily loss stop | {RISK_LIMITS['daily_loss_stop_pct']:.0%} hard stop |
| Min edge to trade | 5 percentage points |
| Fractional Kelly | 0.25x baseline, max 0.5x |
| Min liquidity | $10k Polymarket / $5k Kalshi |
            """
        )


# ---------------------------------------------------------------------------
# Main: render tabs
# ---------------------------------------------------------------------------

def main() -> None:
    render_sidebar()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Trade Lifecycle",
        "Strategy Performance",
        "Calibration",
        "Correlation Map",
        "Live Positions",
    ])

    with tab1:
        tab_trade_lifecycle()

    with tab2:
        tab_strategy_performance()

    with tab3:
        tab_calibration()

    with tab4:
        tab_correlation_map()

    with tab5:
        tab_live_positions()


if __name__ == "__main__":
    main()
