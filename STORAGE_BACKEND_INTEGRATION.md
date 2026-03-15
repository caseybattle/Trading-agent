# StorageBackend Integration Guide

## Overview
The `scripts/storage_backend.py` provides an abstract `StorageBackend` class that routes file I/O to either local filesystem (Windows) or AWS S3 (Lambda).

**Factory Function:**
```python
from storage_backend import get_storage

storage = get_storage()  # Returns LocalStorage or S3Storage based on environment
```

## Integration Pattern

### Step 1: Import and Initialize
```python
from storage_backend import get_storage

# At module load or function entry
storage = get_storage()
```

### Step 2: Replace File Operations

**Replace:** File path construction
```python
# OLD
BANKROLL_FILE = Path(__file__).resolve().parent.parent / "trades" / "bankroll.json"

# NEW (paths are relative to project root)
BANKROLL_PATH = "trades/bankroll.json"
```

**Replace:** JSON operations
```python
# OLD
with open(cfg_path) as f:
    config = json.load(f)

# NEW
config = storage.read_json("backtest/strategy_config.json")
```

**Replace:** JSON writes
```python
# OLD
with open(file_path, 'w') as f:
    json.dump(data, f, indent=2)

# NEW
storage.write_json("backtest/strategy_config.json", data)
```

**Replace:** Parquet operations
```python
# OLD
df = pd.read_parquet(LEDGER_FILE)

# NEW
df = storage.read_parquet("trades/live_trades.parquet")

# Write
storage.write_parquet("trades/live_trades.parquet", df)
```

**Replace:** CSV operations
```python
# OLD
rows = []
with open(SIGNALS_LOG_FILE, newline="") as fh:
    reader = csv.DictReader(fh)
    rows = list(reader)

# NEW
rows = storage.read_csv("trades/signals_log.csv")

# Append single row
storage.append_csv("trades/signals_log.csv", {"id": "123", "outcome": "WIN"})
```

**Replace:** Existence checks
```python
# OLD
if BANKROLL_FILE.exists():

# NEW
if storage.exists("trades/bankroll.json"):
```

## Files to Modify

### 1. kalshi_btc_trader.py
**File I/O Operations:**
- Line ~49: `load_strategy_config()` reads `backtest/strategy_config.json`
- Line ~327: Reads `trades/bankroll.json`

**Changes:**
- Import: `from storage_backend import get_storage`
- Replace `load_strategy_config()` to use `storage.read_json()`
- Replace bankroll.json read to use `storage.read_json()`

### 2. auto_resolver.py
**File I/O Operations:**
- Line ~198: Reads `trades/live_trades.parquet`
- Line ~253-285: Reads/writes `trades/signals_log.csv`

**Changes:**
- Import: `from storage_backend import get_storage`
- Replace parquet read: `storage.read_parquet("trades/live_trades.parquet")`
- Replace CSV operations: `storage.read_csv()` and `storage.append_csv()`

### 3. strategy_optimizer.py
**File I/O Operations:**
- Reads `backtest/strategy_config.json`
- Writes updated `backtest/strategy_config.json`
- Reads/writes optimization logs

**Changes:**
- Import: `from storage_backend import get_storage`
- Replace config read/write with `storage.read_json()` and `storage.write_json()`

### 4. bankroll_tracker.py
**File I/O Operations:**
- Lines ~90-98: Reads/writes `trades/bankroll.json`
- Line ~104: Reads `trades/live_trades.parquet`

**Changes:**
- Import: `from storage_backend import get_storage`
- Replace JSON operations with storage methods
- Replace parquet read with `storage.read_parquet()`

### 5. loss_postmortem.py
**File I/O Operations:**
- Reads `backtest/loss_postmortem.json`
- Writes `backtest/loss_postmortem.json`
- Reads/appends to `backtest/postmortem_log.csv`

**Changes:**
- Import: `from storage_backend import get_storage`
- Replace JSON operations with storage methods
- Replace CSV operations with storage methods

## Key Environment Variables

**Local (Windows):**
```
PROJECT_ROOT=C:\Users\casba\Trading agent\prediction-market-bot
```

**Lambda:**
```
AWS_LAMBDA_FUNCTION_NAME=<function-name>
DATA_BUCKET=kalshi-bot-data-<account-id>
```

The `get_storage()` factory detects these and returns the appropriate backend.

## Testing Modified Scripts

Before deployment, test locally:

```bash
# Verify LocalStorage works (Windows)
python scripts/kalshi_btc_trader.py --bankroll 10 --once

# Check that strategy_config.json is loaded from local filesystem
python scripts/strategy_optimizer.py --dry-run
```

After Lambda deployment, logs will show S3 operations (check CloudWatch).
