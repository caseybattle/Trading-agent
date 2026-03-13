# Data Collection Reference

## Platform APIs

### Polymarket (Gamma API — No Auth Required)
**Base URL**: `https://gamma-api.polymarket.com`

**Resolved markets endpoint**:
```
GET /markets?closed=true&limit=100&offset=0
```

**Key fields per market**:
| Field | Description |
|-------|-------------|
| `id` | Unique market ID |
| `question` | Market question text |
| `category` | politics / crypto / science / sports / other |
| `resolution` | YES / NO / INVALID |
| `volume` | Total trading volume (USD) |
| `liquidity` | Current liquidity (USD) |
| `startDate` | Market open timestamp |
| `endDate` | Market close/resolution timestamp |
| `tokens[0].price` | Current YES price (0–1) |
| `tokens[0].priceHistory` | Hourly price history array |

**Incremental caching**: Store cursor/offset in `data/.polymarket_cursor.json`. On each run, resume from last fetched page. Avoids re-fetching 10k+ markets.

**Target**: All closed markets from last 12 months (~5,000–15,000 markets)

---

### Kalshi (REST API — API Key Required)
**Base URL**: `https://trading.kalshi.com/v2`

**Auth header**: `Authorization: Token <KALSHI_API_KEY>`

**Settled markets endpoint**:
```
GET /markets?status=settled&limit=200&cursor=<cursor>
```

**Key fields**:
| Field | Description |
|-------|-------------|
| `ticker` | Unique market ticker |
| `title` | Market question |
| `category` | Market category |
| `result` | YES / NO |
| `yes_bid` / `no_bid` | Current prices |
| `volume` | Total volume |
| `open_interest` | Open interest |
| `close_time` | Settlement timestamp |

**API key setup**: Store in environment variable `KALSHI_API_KEY`. Never hardcode.

---

## Feature Engineering

Every market snapshot is transformed into a feature vector for ML training and strategy input.

### Features Computed Per Market
| Feature | Formula | Purpose |
|---------|---------|---------|
| `time_to_resolution_hours` | `(end_date - now).total_seconds() / 3600` | Urgency / information horizon |
| `liquidity_ratio` | `volume / max(open_interest, 1)` | Market activity relative to size |
| `price_momentum_24h` | `price_now - price_24h_ago` | Directional price pressure |
| `price_volatility_7d` | `std(price_7d_history)` | Uncertainty / information arrival rate |
| `volume_anomaly_score` | `(volume - category_mean_volume) / category_std_volume` | Z-score vs. category peers |
| `spread_pct` | `(ask - bid) / mid_price` | Execution cost / market efficiency |
| `category` | One-hot encoded | Category-specific effects |
| `days_since_market_open` | `(now - start_date).days` | Market maturity |
| `prior_resolution_rate` | From `data/base_rates.json` | Prior probability anchor |

### Label
- `outcome`: 1 if resolved YES, 0 if NO (INVALID markets excluded from training)

---

## Base Rate Database

**File**: `data/base_rates.json`

**Schema**:
```json
{
  "politics": {
    "yes_rate": 0.42,
    "median_resolution_days": 45,
    "price_accuracy_at_7d": 0.78,
    "category_calibration": 0.91,
    "sample_count": 1247
  },
  "crypto": {
    "yes_rate": 0.51,
    "median_resolution_days": 7,
    "price_accuracy_at_7d": 0.83,
    "category_calibration": 0.88,
    "sample_count": 892
  },
  "science": { ... },
  "sports": { ... },
  "other": { ... }
}
```

**How base rates anchor probability estimates**:
```python
# Superforecasting anchor: blend model output with base rate
anchored_p = (1 - anchor_weight) * model_p + anchor_weight * base_rate_yes
# anchor_weight starts high (0.5) for new markets, decays as information accumulates
anchor_weight = max(0.1, 0.5 - (days_since_open / 60))
```

---

## Storage Schema

```
data/
├── polymarket_resolved.parquet     # Raw resolved markets, columns: id, question, category, resolution, volume, liquidity, start_date, end_date
├── kalshi_resolved.parquet         # Same schema, source=kalshi
├── .polymarket_cursor.json         # {"last_offset": 4200, "last_run": "2024-01-15T10:30:00Z"}
├── .kalshi_cursor.json             # {"last_cursor": "eyJhbGciOiJSUzI...", "last_run": "..."}
├── features/
│   └── market_features.parquet    # Engineered features + outcome label, all markets
└── base_rates.json                 # Per-category statistics
```

**Parquet advantages**: Columnar, compressed, fast pandas read, supports append without reload.

---

## Data Quality Rules

1. **Exclude INVALID resolutions** — these have no label (not YES/NO)
2. **Exclude markets with < 1 hour of price history** — insufficient signal
3. **Exclude markets with volume < $1,000** — too illiquid for reliable price signals
4. **Clip features** at 3 standard deviations — removes outliers without dropping rows
5. **Deduplicate by market ID** — Polymarket API sometimes returns duplicates near pagination boundaries

---

## Holdout Set (Critical — YouTube Addition)

The last **20% of resolved markets by date** are RESERVED as a holdout set.
- Training/validation: first 80%
- Holdout: last 20% — NEVER used for feature engineering, model training, or hyperparameter tuning
- Purpose: Final unbiased evaluation of the complete system after all development is done
- Implementation: `build_base_rates.py` computes base rates only from the training 80%

```python
# In build_base_rates.py and backtest_runner.py:
cutoff = df['end_date'].quantile(0.80)
train_df = df[df['end_date'] <= cutoff]
holdout_df = df[df['end_date'] > cutoff]
# Train/backtest only on train_df
# Hold holdout_df for final validation
```

---

## Running Data Collection

```bash
# Step 1: Collect historical resolved markets
python scripts/collect_polymarket.py       # ~5,000–15,000 markets, incremental
python scripts/collect_kalshi.py           # Requires KALSHI_API_KEY env var

# Step 2: Build feature dataset + base rates
python scripts/build_base_rates.py

# Verify output
python -c "
import pandas as pd, json
df = pd.read_parquet('data/features/market_features.parquet')
print(f'Markets: {len(df)}, Features: {df.shape[1]}')
print(df.dtypes)
br = json.load(open('data/base_rates.json'))
print(list(br.keys()))
"
```
