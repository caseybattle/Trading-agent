"""Abstract storage backend for local and cloud (S3) file I/O."""

import os
import json
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
import boto3


class StorageBackend(ABC):
    """Abstract base class for file storage operations."""

    @abstractmethod
    def read_json(self, path: str) -> dict:
        """Read JSON file."""
        pass

    @abstractmethod
    def write_json(self, path: str, data: dict) -> None:
        """Write JSON file."""
        pass

    @abstractmethod
    def read_parquet(self, path: str) -> pd.DataFrame:
        """Read Parquet file."""
        pass

    @abstractmethod
    def write_parquet(self, path: str, df: pd.DataFrame) -> None:
        """Write Parquet file."""
        pass

    @abstractmethod
    def read_csv(self, path: str) -> list:
        """Read CSV file and return list of dicts."""
        pass

    @abstractmethod
    def append_csv(self, path: str, row: dict, fieldnames: list = None) -> None:
        """Append row to CSV file."""
        pass

    @abstractmethod
    def write_csv(self, path: str, rows: list, fieldnames: list) -> None:
        """Write complete CSV file (read-modify-write pattern)."""
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if file exists."""
        pass


class LocalStorage(StorageBackend):
    """Local filesystem storage (Windows/Linux)."""

    def __init__(self, project_root: str = None):
        self.project_root = project_root or Path(__file__).parent.parent
        Path(self.project_root).mkdir(parents=True, exist_ok=True)

    def _get_full_path(self, path: str) -> Path:
        if Path(path).is_absolute():
            return Path(path)
        return Path(self.project_root) / path

    def read_json(self, path: str) -> dict:
        full_path = self._get_full_path(path)
        with open(full_path, 'r') as f:
            return json.load(f)

    def write_json(self, path: str, data: dict) -> None:
        full_path = self._get_full_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w') as f:
            json.dump(data, f, indent=2)

    def read_parquet(self, path: str) -> pd.DataFrame:
        full_path = self._get_full_path(path)
        return pd.read_parquet(full_path)

    def write_parquet(self, path: str, df: pd.DataFrame) -> None:
        full_path = self._get_full_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(full_path, index=False)

    def read_csv(self, path: str) -> list:
        full_path = self._get_full_path(path)
        try:
            df = pd.read_csv(full_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(full_path, encoding='latin-1')
        return df.to_dict('records')

    def append_csv(self, path: str, row: dict, fieldnames: list = None) -> None:
        full_path = self._get_full_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([row])
        if full_path.exists():
            try:
                existing = pd.read_csv(full_path, encoding='utf-8', on_bad_lines='skip')
            except UnicodeDecodeError:
                existing = pd.read_csv(full_path, encoding='latin-1', on_bad_lines='skip')
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(full_path, index=False, encoding='utf-8')

    def write_csv(self, path: str, rows: list, fieldnames: list) -> None:
        full_path = self._get_full_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(full_path, index=False)

    def exists(self, path: str) -> bool:
        return self._get_full_path(path).exists()


class S3Storage(StorageBackend):
    """AWS S3 storage backend."""

    def __init__(self, bucket: str):
        self.bucket = bucket
        self.s3_client = boto3.client('s3')

    def _get_key(self, path: str) -> str:
        # Remove leading slashes, use path as-is for S3 key
        return path.lstrip('/')

    def read_json(self, path: str) -> dict:
        key = self._get_key(path)
        obj = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        return json.loads(obj['Body'].read())

    def write_json(self, path: str, data: dict) -> None:
        key = self._get_key(path)
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(data, indent=2),
            ContentType='application/json'
        )

    def read_parquet(self, path: str) -> pd.DataFrame:
        key = self._get_key(path)
        obj = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        return pd.read_parquet(obj['Body'])

    def write_parquet(self, path: str, df: pd.DataFrame) -> None:
        key = self._get_key(path)
        buffer = df.to_parquet(index=False)
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer
        )

    def read_csv(self, path: str) -> list:
        key = self._get_key(path)
        obj = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        df = pd.read_csv(obj['Body'])
        return df.to_dict('records')

    def append_csv(self, path: str, row: dict, fieldnames: list = None) -> None:
        key = self._get_key(path)
        df_new = pd.DataFrame([row])

        try:
            obj = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            df_existing = pd.read_csv(obj['Body'])
            df = pd.concat([df_existing, df_new], ignore_index=True)
        except self.s3_client.exceptions.NoSuchKey:
            df = df_new

        buffer = df.to_csv(index=False)
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer
        )

    def write_csv(self, path: str, rows: list, fieldnames: list) -> None:
        key = self._get_key(path)
        df = pd.DataFrame(rows)
        buffer = df.to_csv(index=False)
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer
        )

    def exists(self, path: str) -> bool:
        key = self._get_key(path)
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.s3_client.exceptions.NoSuchKey:
            return False


def get_storage() -> StorageBackend:
    """Factory function: return appropriate storage backend."""
    if os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
        bucket = os.getenv('DATA_BUCKET')
        if not bucket:
            raise ValueError('DATA_BUCKET env var required in Lambda')
        return S3Storage(bucket)
    else:
        # Local Windows/Linux
        project_root = os.getenv('PROJECT_ROOT')
        return LocalStorage(project_root)
