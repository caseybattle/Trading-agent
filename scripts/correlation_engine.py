"""
correlation_engine.py — Cross-market correlation matrix + arbitrage signals

Builds a sparse correlation graph from resolved market outcomes grouped by
shared categories and keyword overlap. Exposes correlation_penalty() for
Kelly sizing and detect_arbitrage() for opportunity scanning.

Saves/loads from data/market_correlations.parquet.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
CORRELATIONS_PATH = DATA_DIR / "market_correlations.parquet"

# Correlation thresholds
MIN_CORRELATION = 0.25        # Below this: treat as independent
ARBITRAGE_THRESHOLD = 0.15    # Price gap vs correlation that flags arb
MAX_KEYWORD_PAIRS = 50_000    # Cap sparse matrix size for memory safety


@dataclass
class ArbitrageSignal:
    market_a: str
    market_b: str
    price_a: float
    price_b: float
    correlation: float
    implied_gap: float        # |price_a - (1 - price_b)| for complementary markets
    signal_strength: float    # correlation * implied_gap


@dataclass
class CorrelationGraph:
    """Sparse adjacency: market_id -> {neighbor_id: correlation_coef}"""
    edges: Dict[str, Dict[str, float]] = field(default_factory=dict)
    built_at: Optional[str] = None
    market_count: int = 0

    def neighbors(self, market_id: str) -> Dict[str, float]:
        return self.edges.get(market_id, {})

    def max_correlation(self, market_id: str) -> float:
        nbrs = self.neighbors(market_id)
        return max(nbrs.values(), default=0.0)


# ---------------------------------------------------------------------------
# Building the correlation graph
# ---------------------------------------------------------------------------

def _extract_keywords(question: str) -> str:
    """Normalize market question for TF-IDF: lowercase, strip punctuation."""
    text = question.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _same_category_pairs(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Return index pairs that share the same category."""
    pairs: List[Tuple[int, int]] = []
    for cat, group in df.groupby("category"):
        idxs = group.index.tolist()
        if len(idxs) > 5000:
            # Too many pairs — sample
            import random
            idxs = random.sample(idxs, 5000)
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                pairs.append((idxs[i], idxs[j]))
                if len(pairs) >= MAX_KEYWORD_PAIRS:
                    return pairs
    return pairs


def build_correlation_graph(
    df: pd.DataFrame,
    tfidf_threshold: float = 0.60,
    outcome_threshold: float = 0.30,
    alpha: float = 0.5,
) -> CorrelationGraph:
    """
    Build sparse correlation graph from resolved markets.

    Two-factor correlation:
      1. Text similarity (TF-IDF cosine) — shared keywords/entities
      2. Outcome correlation (phi coefficient on binary outcomes)

    Final correlation = alpha * text_sim + (1-alpha) * outcome_corr

    Args:
        df: DataFrame with columns [market_id, question, category, outcome]
        tfidf_threshold: Only pair markets with text similarity above this
        outcome_threshold: Minimum outcome correlation to keep edge
        alpha: Blend weight for text vs. outcome correlation
    """
    required = {"market_id", "question", "category", "outcome"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["market_id", "question", "outcome"]).copy()
    df["outcome"] = df["outcome"].astype(int)
    df = df.reset_index(drop=True)

    log.info(f"Building correlation graph from {len(df)} markets...")

    # --- Step 1: TF-IDF text similarity matrix (same-category pairs only) ---
    questions = [_extract_keywords(q) for q in df["question"]]
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(questions)

    graph = CorrelationGraph()
    graph.market_count = len(df)

    id_col = df["market_id"].tolist()
    outcomes = df["outcome"].values

    # Process category by category to keep memory bounded
    for cat, group_df in df.groupby("category"):
        idxs = group_df.index.tolist()
        if len(idxs) < 2:
            continue

        sub_tfidf = tfidf_matrix[idxs]
        sim_matrix = cosine_similarity(sub_tfidf)

        for i_local, i_global in enumerate(idxs):
            for j_local, j_global in enumerate(idxs):
                if j_local <= i_local:
                    continue

                text_sim = float(sim_matrix[i_local, j_local])
                if text_sim < tfidf_threshold:
                    continue

                # Phi coefficient on binary outcomes
                a = int(outcomes[i_global])
                b = int(outcomes[j_global])
                # Use population-level correlation from feature vectors if available
                # Here: simple exact match proxy
                outcome_match = 1.0 if a == b else -0.5

                corr = alpha * text_sim + (1 - alpha) * max(outcome_match, 0.0)
                if corr < MIN_CORRELATION:
                    continue

                mid_a = id_col[i_global]
                mid_b = id_col[j_global]

                graph.edges.setdefault(mid_a, {})[mid_b] = round(corr, 4)
                graph.edges.setdefault(mid_b, {})[mid_a] = round(corr, 4)

    log.info(
        f"Correlation graph: {len(graph.edges)} nodes, "
        f"{sum(len(v) for v in graph.edges.values()) // 2} edges"
    )
    return graph


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_correlation_graph(graph: CorrelationGraph, path: Path = CORRELATIONS_PATH) -> None:
    """Serialize graph to parquet: (market_a, market_b, correlation)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for mid_a, neighbors in graph.edges.items():
        for mid_b, corr in neighbors.items():
            if mid_a < mid_b:  # Store each edge once
                rows.append({"market_a": mid_a, "market_b": mid_b, "correlation": corr})

    if not rows:
        log.warning("Empty correlation graph — nothing to save.")
        return

    pd.DataFrame(rows).to_parquet(path, index=False)
    log.info(f"Saved {len(rows)} correlation edges to {path}")


def load_correlation_graph(path: Path = CORRELATIONS_PATH) -> CorrelationGraph:
    """Load correlation graph from parquet."""
    if not path.exists():
        log.warning(f"No correlation file at {path} — returning empty graph.")
        return CorrelationGraph()

    df = pd.read_parquet(path)
    graph = CorrelationGraph()
    for _, row in df.iterrows():
        a, b, c = row["market_a"], row["market_b"], row["correlation"]
        graph.edges.setdefault(a, {})[b] = c
        graph.edges.setdefault(b, {})[a] = c
    log.info(f"Loaded {len(df)} correlation edges from {path}")
    return graph


# ---------------------------------------------------------------------------
# Kelly integration: correlation_penalty()
# ---------------------------------------------------------------------------

def correlation_penalty(
    market_id: str,
    current_positions: Dict[str, float],
    graph: CorrelationGraph,
) -> float:
    """
    Compute correlation penalty [0, 1) for Kelly sizing.

    Returns the weighted correlation of `market_id` against existing positions,
    scaled by position sizes (as fraction of bankroll).

    Usage in kelly_calculator:
        kelly_adj = kelly_raw * max(0.1, 1 - correlation_penalty(...))

    Args:
        market_id: Market being evaluated for new position
        current_positions: {market_id: fraction_of_bankroll}  e.g. {"m1": 0.03}
        graph: Loaded CorrelationGraph

    Returns:
        Scalar in [0, 1) — 0 means no correlation penalty, 0.9 means 90% reduction
    """
    if not current_positions:
        return 0.0

    neighbors = graph.neighbors(market_id)
    if not neighbors:
        return 0.0

    total_exposure = sum(current_positions.values())
    if total_exposure <= 0:
        return 0.0

    weighted_corr = 0.0
    for pos_id, pos_size in current_positions.items():
        if pos_id == market_id:
            continue
        corr = neighbors.get(pos_id, 0.0)
        weighted_corr += corr * (pos_size / max(total_exposure, 1e-9))

    # Clip to [0, 0.90] — never zero out a trade completely via correlation alone
    return float(np.clip(weighted_corr, 0.0, 0.90))


# ---------------------------------------------------------------------------
# Arbitrage signal detection
# ---------------------------------------------------------------------------

def detect_arbitrage(
    live_prices: Dict[str, float],
    graph: CorrelationGraph,
    threshold: float = ARBITRAGE_THRESHOLD,
) -> List[ArbitrageSignal]:
    """
    Detect cross-market arbitrage signals.

    Identifies market pairs where:
    - Correlation is high (>= MIN_CORRELATION)
    - Prices are inconsistent: for strongly correlated markets,
      price_a should be close to price_b (same direction).
      For complementary markets, price_a + price_b should be ~1.

    Inconsistency = implied_gap > threshold

    Args:
        live_prices: {market_id: yes_price_0_to_1}
        graph: Loaded CorrelationGraph
        threshold: Minimum price gap to flag as arbitrage

    Returns:
        List of ArbitrageSignal sorted by signal_strength descending
    """
    signals: List[ArbitrageSignal] = []
    processed = set()

    for mid_a, price_a in live_prices.items():
        neighbors = graph.neighbors(mid_a)
        for mid_b, corr in neighbors.items():
            pair_key = tuple(sorted([mid_a, mid_b]))
            if pair_key in processed:
                continue
            processed.add(pair_key)

            if mid_b not in live_prices:
                continue

            price_b = live_prices[mid_b]

            # Direct correlation: price_a should ≈ price_b
            direct_gap = abs(price_a - price_b)

            # Complementary: price_a + price_b should ≈ 1.0
            complement_gap = abs(price_a - (1.0 - price_b))

            implied_gap = min(direct_gap, complement_gap)
            signal_strength = corr * implied_gap

            if implied_gap > threshold:
                signals.append(
                    ArbitrageSignal(
                        market_a=mid_a,
                        market_b=mid_b,
                        price_a=price_a,
                        price_b=price_b,
                        correlation=corr,
                        implied_gap=implied_gap,
                        signal_strength=signal_strength,
                    )
                )

    signals.sort(key=lambda s: s.signal_strength, reverse=True)
    return signals


# ---------------------------------------------------------------------------
# Category-level exposure aggregation
# ---------------------------------------------------------------------------

def category_exposure(
    current_positions: Dict[str, float],
    market_categories: Dict[str, str],
) -> Dict[str, float]:
    """
    Sum position sizes by category.

    Args:
        current_positions: {market_id: fraction_of_bankroll}
        market_categories: {market_id: category_string}

    Returns:
        {category: total_fraction_of_bankroll}
    """
    exposure: Dict[str, float] = {}
    for mid, size in current_positions.items():
        cat = market_categories.get(mid, "other")
        exposure[cat] = exposure.get(cat, 0.0) + size
    return exposure


def total_correlated_exposure(
    current_positions: Dict[str, float],
    graph: CorrelationGraph,
) -> float:
    """
    Compute total correlation-adjusted portfolio exposure.

    Positions in correlated markets contribute more than their raw size.
    Exposure_adj = sum(size_i * (1 + max_corr_to_portfolio_i))

    Used to enforce the 40% total exposure cap.
    """
    if not current_positions:
        return 0.0

    total = 0.0
    for mid, size in current_positions.items():
        others = {k: v for k, v in current_positions.items() if k != mid}
        penalty = correlation_penalty(mid, others, graph)
        total += size * (1.0 + penalty)

    return float(total)


# ---------------------------------------------------------------------------
# CLI entry point: rebuild and save correlation graph
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build cross-market correlation graph")
    parser.add_argument("--features", default="data/features/market_features.parquet")
    parser.add_argument("--out", default=str(CORRELATIONS_PATH))
    parser.add_argument("--tfidf-threshold", type=float, default=0.60)
    parser.add_argument("--outcome-threshold", type=float, default=0.30)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Blend: 0=pure outcome correlation, 1=pure text similarity")
    args = parser.parse_args()

    features_path = Path(args.features)
    if not features_path.exists():
        log.error(f"Features file not found: {features_path}")
        log.error("Run build_base_rates.py first to generate features.")
        return

    df = pd.read_parquet(features_path)
    log.info(f"Loaded {len(df)} markets from {features_path}")

    graph = build_correlation_graph(
        df,
        tfidf_threshold=args.tfidf_threshold,
        outcome_threshold=args.outcome_threshold,
        alpha=args.alpha,
    )
    save_correlation_graph(graph, Path(args.out))

    # Print summary stats
    edge_counts = [len(v) for v in graph.edges.values()]
    if edge_counts:
        print(f"\nCorrelation graph summary:")
        print(f"  Nodes (markets with edges): {len(graph.edges)}")
        print(f"  Edges: {sum(edge_counts) // 2}")
        print(f"  Avg neighbors per node: {np.mean(edge_counts):.1f}")
        print(f"  Max neighbors: {max(edge_counts)}")
        print(f"  Saved to: {args.out}")
    else:
        print("No correlations found above threshold.")


if __name__ == "__main__":
    main()
