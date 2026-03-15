"""
storage_backend.py — Abstract file I/O with local and S3 implementations.

Provides a unified StorageBackend interface so every script can read/write
trades, bankroll, strategy config, and logs without caring whether files
live on the local filesystem or in an S3 bucket.

Usage:
    from storage_backend import get_storage

    store = get_storage()
    cfg = store.read_json("backtest/strategy_config.json")
    store.write_json("backtest/strategy_config.json", cfg)
    df = store.read_parquet("trades/live_trades.parquet")
    store.append_csv("trades/signals_log.csv", row_dict)
"""

from __future__ import annotations

import csv
import io
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class StorageBackend(ABC):
    """Unified interface for file operations used across the trading bot."""

    @abstractmethod
    def read_json(self, path: str) -> dict:
        """Read a JSON file. Returns {} if the file does not exist."""
        ...

    @abstractmethod
    def write_json(self, path: str, data: dict) -> None:
        """Write a dict as pretty-printed JSON."""
        ...

    @abstractmethod
    def read_parquet(self, path: str) -> pd.DataFrame:
        """Read a Parquet file. Returns empty DataFrame if not found."""
        ...

    @abstractmethod
    def write_parquet(self, path: str, df: pd.DataFrame) -> None:
        """Write a DataFrame to Parquet (no index)."""
        ...

    @abstractmethod
    def read_csv(self, path: str) -> list[dict]:
        """Read a CSV into a list of row dicts. Returns [] if not found."""
        ...

    @abstractmethod
    def write_csv(self, path: str, rows: list[dict], fieldnames: list[str] | None = None) -> None:
        """Overwrite a CSV file with the given rows."""
        ...

    @abstractmethod
    def append_csv(self, path: str, row: dict) -> None:
        """Append a single row to a CSV. Creates file + header if missing."""
        ...

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check whether a file exists."""
        ...

    @abstractmethod
    def makedirs(self, path: str) -> None:
        """Ensure a directory exists (no-op for object stores)."""
        ...

    @abstractmethod
    def read_text(self, path: str) -> str | None:
        """Read a file as UTF-8 text. Returns None if not found."""
        ...

    @abstractmethod
    def write_text(self, path: str, text: str) -> None:
        """Write UTF-8 text to a file."""
        ...


# ---------------------------------------------------------------------------
# Local filesystem implementation
# ---------------------------------------------------------------------------

class LocalStorage(StorageBackend):
    """File I/O against the local filesystem, paths relative to project_root."""

    def __init__(self, project_root: str) -> None:
        self.root = Path(project_root)

    def _resolve(self, path: str) -> Path:
        """Resolve a relative path against the project root."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.root / p

    # -- JSON ---------------------------------------------------------------

    def read_json(self, path: str) -> dict:
        fp = self._resolve(path)
        if not fp.exists():
            return {}
        with fp.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def write_json(self, path: str, data: dict) -> None:
        fp = self._resolve(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        with fp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)

    # -- Parquet ------------------------------------------------------------

    def read_parquet(self, path: str) -> pd.DataFrame:
        fp = self._resolve(path)
        if not fp.exists():
            return pd.DataFrame()
        return pd.read_parquet(fp)

    def write_parquet(self, path: str, df: pd.DataFrame) -> None:
        fp = self._resolve(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(fp, index=False)

    # -- CSV ----------------------------------------------------------------

    def read_csv(self, path: str) -> list[dict]:
        fp = self._resolve(path)
        if not fp.exists():
            return []
        with fp.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return list(reader)

    def write_csv(self, path: str, rows: list[dict], fieldnames: list[str] | None = None) -> None:
        fp = self._resolve(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            # Write empty file
            fp.write_text("", encoding="utf-8")
            return
        if fieldnames is None:
            fieldnames = list(rows[0].keys())
        with fp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def append_csv(self, path: str, row: dict) -> None:
        fp = self._resolve(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        file_exists = fp.exists() and fp.stat().st_size > 0
        fieldnames = list(row.keys())
        with fp.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # -- Generic ------------------------------------------------------------

    def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def makedirs(self, path: str) -> None:
        self._resolve(path).mkdir(parents=True, exist_ok=True)

    def read_text(self, path: str) -> str | None:
        fp = self._resolve(path)
        if not fp.exists():
            return None
        return fp.read_text(encoding="utf-8")

    def write_text(self, path: str, text: str) -> None:
        fp = self._resolve(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# S3 implementation
# ---------------------------------------------------------------------------

class S3Storage(StorageBackend):
    """File I/O against an S3 bucket. Keys are prefix + relative path."""

    def __init__(self, bucket: str, prefix: str = "") -> None:
        import boto3
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.s3 = boto3.client("s3")

    def _key(self, path: str) -> str:
        """Build the full S3 key from a relative path."""
        parts = [self.prefix, path] if self.prefix else [path]
        return "/".join(p.strip("/") for p in parts if p)

    def _get_bytes(self, path: str) -> bytes | None:
        """Download an object as bytes. Returns None on NoSuchKey."""
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self._key(path))
            return resp["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None
        except Exception as exc:
            # ClientError for missing keys can also appear as 404
            err_code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if err_code in ("NoSuchKey", "404"):
                return None
            raise

    def _put_bytes(self, path: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        """Upload bytes to S3."""
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self._key(path),
            Body=data,
            ContentType=content_type,
        )

    # -- JSON ---------------------------------------------------------------

    def read_json(self, path: str) -> dict:
        raw = self._get_bytes(path)
        if raw is None:
            return {}
        return json.loads(raw.decode("utf-8"))

    def write_json(self, path: str, data: dict) -> None:
        payload = json.dumps(data, indent=2, default=str).encode("utf-8")
        self._put_bytes(path, payload, content_type="application/json")

    # -- Parquet ------------------------------------------------------------

    def read_parquet(self, path: str) -> pd.DataFrame:
        raw = self._get_bytes(path)
        if raw is None:
            return pd.DataFrame()
        return pd.read_parquet(io.BytesIO(raw))

    def write_parquet(self, path: str, df: pd.DataFrame) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self._put_bytes(path, buf.getvalue(), content_type="application/octet-stream")

    # -- CSV ----------------------------------------------------------------

    def read_csv(self, path: str) -> list[dict]:
        raw = self._get_bytes(path)
        if raw is None:
            return []
        text = raw.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        return list(reader)

    def write_csv(self, path: str, rows: list[dict], fieldnames: list[str] | None = None) -> None:
        if not rows:
            self._put_bytes(path, b"", content_type="text/csv")
            return
        if fieldnames is None:
            fieldnames = list(rows[0].keys())
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        self._put_bytes(path, buf.getvalue().encode("utf-8"), content_type="text/csv")

    def append_csv(self, path: str, row: dict) -> None:
        """Read existing CSV, append row, write back (S3 has no append)."""
        existing = self.read_csv(path)
        existing.append(row)
        # Preserve column order: use existing row keys if available, else new row
        if len(existing) > 1:
            fieldnames = list(existing[0].keys())
        else:
            fieldnames = list(row.keys())
        self.write_csv(path, existing, fieldnames=fieldnames)

    # -- Generic ------------------------------------------------------------

    def exists(self, path: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._key(path))
            return True
        except Exception:
            return False

    def makedirs(self, path: str) -> None:
        # S3 has no directories — this is a no-op.
        pass

    def read_text(self, path: str) -> str | None:
        raw = self._get_bytes(path)
        if raw is None:
            return None
        return raw.decode("utf-8")

    def write_text(self, path: str, text: str) -> None:
        self._put_bytes(path, text.encode("utf-8"), content_type="text/plain")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_storage() -> StorageBackend:
    """
    Return the appropriate storage backend based on the runtime environment.

    - In AWS Lambda (AWS_LAMBDA_FUNCTION_NAME is set): returns S3Storage
      configured from DATA_BUCKET and optional DATA_PREFIX env vars.
    - Otherwise: returns LocalStorage rooted at the project directory
      (one level above the scripts/ folder).
    """
    if os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        bucket = os.environ["DATA_BUCKET"]
        prefix = os.getenv("DATA_PREFIX", "")
        return S3Storage(bucket=bucket, prefix=prefix)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return LocalStorage(project_root=project_root)
