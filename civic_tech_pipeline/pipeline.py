"""Civic tech socio-economic ETL pipeline.

Pulls public Open Data REST endpoints (in parallel), caches responses,
engineers socio-economic indicator features, flags statistical disparities,
deduplicates overlapping records, and writes versioned Parquet/CSV extracts
plus a JSON run summary.

See README.md for design notes and known limitations. In particular: the
HUD and FCC source configurations below use response shapes that have NOT
been verified against the live endpoints (see README) -- only the Census
ACS "array of arrays" shape is verified and covered by tests. Treat the
HUD/FCC SourceConfig entries as templates to adapt once you've confirmed
their real payload shape.
"""

import os
import sys
import json
import time
import logging
import hashlib
import sqlite3
import argparse
import functools
import threading
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import requests
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("civic_etl")

DEFAULT_TIMEOUT = 15
MAX_RETRIES = 4
BACKOFF_BASE = 1.6


class PipelineError(Exception):
    """Base class for all pipeline-raised errors."""


class SourceUnavailableError(PipelineError):
    """Raised when a source cannot be fetched after retries are exhausted."""


class SchemaValidationError(PipelineError):
    """Raised when a fetched source's data doesn't match its declared schema."""


RETRYABLE_EXCEPTIONS = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
    json.JSONDecodeError,
)


def _is_retryable_http_error(exc: requests.HTTPError) -> bool:
    """5xx responses are usually transient server-side failures worth
    retrying; 4xx responses (404, 401, 403, ...) mean the request itself
    was rejected and retrying identically won't help -- it just wastes
    time and API quota against a server that already said no.
    """
    status = getattr(exc.response, "status_code", None)
    return status is not None and 500 <= status < 600


def retry(max_attempts: int = MAX_RETRIES, base: float = BACKOFF_BASE,
          exceptions: tuple = RETRYABLE_EXCEPTIONS):
    """Retry a function with exponential backoff, but only for transient
    network/parsing failures (`exceptions`), plus 5xx HTTP errors.
    requests.HTTPError (raised by response.raise_for_status()) is handled
    separately so that non-transient 4xx errors are NOT retried -- they're
    allowed to propagate immediately, same as a programming error or a
    KeyError from bad config, instead of being retried 4x and misreported
    as a transient source outage.
    """
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except requests.HTTPError as exc:
                    if not _is_retryable_http_error(exc):
                        raise
                    last_exc = exc
                except exceptions as exc:
                    last_exc = exc

                if attempt < max_attempts:
                    sleep_for = min(base ** attempt + np.random.uniform(0, 0.5), 8)
                    logger.warning(
                        "%s failed on attempt %d/%d (%s); retrying in %.2fs",
                        fn.__name__, attempt, max_attempts, last_exc, sleep_for,
                    )
                    time.sleep(sleep_for)
            raise SourceUnavailableError(
                f"{fn.__name__} exhausted {max_attempts} attempts"
            ) from last_exc
        return wrapper
    return decorator


def timed(fn: Callable):
    """Log wall-clock duration of the wrapped call at INFO level."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s completed in %.3fs", fn.__name__, elapsed)
        return result
    return wrapper


@dataclass
class SourceConfig:
    """Configuration for a single Open Data source.

    parser:
        "records"     -- response is a dict/list of dicts (optionally
                          nested under `record_path`); parsed with
                          pd.json_normalize. This is the common REST-JSON
                          shape.
        "array_table" -- response is a list of lists where the first row
                          is the header and the rest are data rows (this
                          is the shape the Census ACS API actually
                          returns; see README).
    field_map:
        Maps raw response field names to canonical names used downstream
        by FeatureExtractor (e.g. {"B19013_001E": "median_income"}).
        Renaming happens before schema validation.
    required_fields:
        List of (canonical_field_name, expected_kind) tuples, where
        expected_kind is "numeric" or "string". Validated after
        field_map renaming. A field must be PRESENT, and if "numeric",
        at least half its values must survive pd.to_numeric coercion --
        this catches "the whole column is text/garbage" without being so
        strict that a source with a few missing values is treated as
        totally broken.
    geo_id_fields:
        Canonical field names (post field_map) to concatenate into a
        "geo_id" column, used for cross-source deduplication.
    """
    name: str
    url: str
    params: dict = field(default_factory=dict)
    record_path: Optional[str] = None
    parser: str = "records"
    field_map: dict = field(default_factory=dict)
    required_fields: tuple = ()
    geo_id_fields: tuple = ()
    weight: float = 1.0


@dataclass
class PipelineConfig:
    sources: list
    output_dir: str = "civic_output"
    cache_db_path: str = "civic_pipeline_cache.sqlite3"
    cache_ttl_seconds: int = 3600
    disparity_z_cutoff: float = 1.5
    max_workers: int = 4
    fixture_dir: Optional[str] = None


def _traverse_record_path(data: Any, path: Optional[str]) -> Any:
    """Walk `data` along a dot-separated `path`, supporting both dict keys
    and integer list indices (e.g. "results.0.rows"). Raises PipelineError
    with a precise location instead of silently stopping partway, which is
    what the previous implementation did when a path segment landed on a
    list.
    """
    if not path:
        return data
    cur = data
    parts = path.split(".")
    for i, part in enumerate(parts):
        located = ".".join(parts[: i + 1])
        if isinstance(cur, dict):
            if part not in cur:
                raise PipelineError(
                    f"record_path segment '{part}' not found in dict (path so far: '{located}')"
                )
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                raise PipelineError(
                    f"record_path segment '{part}' is not a valid list index "
                    f"(path so far: '{located}'); use an integer to index into a list"
                )
            try:
                cur = cur[idx]
            except IndexError:
                raise PipelineError(
                    f"record_path index {idx} out of range (path so far: '{located}')"
                )
        else:
            raise PipelineError(
                f"record_path cannot descend into {type(cur).__name__} at segment '{part}' "
                f"(path so far: '{located}')"
            )
    return cur


class CacheStore:
    """Thread-safe (thread-local connections) SQLite cache with WAL mode,
    a busy timeout so concurrent readers/writers don't collide, and
    expiry-based pruning so the cache doesn't grow unbounded.
    """

    def __init__(self, path: str):
        self.path = path
        self._local = threading.local()
        self._init_schema()

    def _configure(self, conn: sqlite3.Connection):
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")

    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self.path, timeout=5)
            self._configure(conn)
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self):
        conn = sqlite3.connect(self.path, timeout=5)
        self._configure(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                fetched_at REAL NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    def get(self, key: str, ttl: int) -> Optional[Any]:
        row = self.conn.execute(
            "SELECT payload, fetched_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        payload, fetched_at = row
        if time.time() - fetched_at > ttl:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def set(self, key: str, value: Any):
        self.conn.execute(
            "REPLACE INTO cache (key, payload, fetched_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), time.time()),
        )
        self.conn.commit()

    def prune_expired(self, ttl: int):
        """Delete cache rows older than `ttl` seconds. Call periodically
        (e.g. once per pipeline run) -- otherwise the cache table grows
        forever even though expired rows are never served.
        """
        cutoff = time.time() - ttl
        cur = self.conn.execute("DELETE FROM cache WHERE fetched_at < ?", (cutoff,))
        self.conn.commit()
        if cur.rowcount:
            logger.info("pruned %d expired cache row(s)", cur.rowcount)

    def close(self):
        """Close and release the CURRENT THREAD's connection. Thread-local
        connections are created lazily and otherwise never closed, which
        is harmless for a short-lived CLI run (the OS reclaims file
        handles at exit) but would leak connections in a long-lived
        process that keeps creating new threads. Call this from each
        worker thread when it's done, and from main() before exit.
        """
        if hasattr(self._local, "conn"):
            try:
                self._local.conn.close()
            finally:
                del self._local.conn


class DataSourceClient:
    """Fetches and parses each configured source, either from the network
    (with caching) or, in offline mode, from a fixture file on disk.
    """

    def __init__(self, cache: CacheStore, ttl: int, fixture_dir: Optional[str] = None):
        self.cache = cache
        self.ttl = ttl
        self.fixture_dir = fixture_dir
        self._local = threading.local()

    @property
    def session(self) -> requests.Session:
        # requests.Session is not guaranteed thread-safe under concurrent
        # use; give each worker thread its own, matching the thread-local
        # pattern already used for SQLite connections in CacheStore.
        if not hasattr(self._local, "session"):
            s = requests.Session()
            s.headers.update({"User-Agent": "civic-tech-etl/1.0"})
            self._local.session = s
        return self._local.session

    @staticmethod
    def _cache_key(cfg: SourceConfig) -> str:
        raw = f"{cfg.url}:{json.dumps(cfg.params, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @retry()
    def _fetch_raw(self, cfg: SourceConfig) -> Any:
        response = self.session.get(cfg.url, params=cfg.params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        return response.json()

    def _load_fixture(self, cfg: SourceConfig) -> Any:
        path = os.path.join(self.fixture_dir, f"{cfg.name}.json")
        if not os.path.exists(path):
            raise SourceUnavailableError(f"no fixture file found for source '{cfg.name}' at {path}")
        with open(path) as fh:
            return json.load(fh)

    def _get_raw(self, cfg: SourceConfig) -> Any:
        if self.fixture_dir:
            logger.info("loading source '%s' from fixture (offline mode)", cfg.name)
            return self._load_fixture(cfg)

        key = self._cache_key(cfg)
        cached = self.cache.get(key, self.ttl)
        if cached is not None:
            logger.info("cache hit for source '%s'", cfg.name)
            return cached

        logger.info("fetching source '%s' from network", cfg.name)
        raw = self._fetch_raw(cfg)
        self.cache.set(key, raw)
        return raw

    def fetch(self, cfg: SourceConfig) -> pd.DataFrame:
        raw = self._get_raw(cfg)
        records = _traverse_record_path(raw, cfg.record_path)

        if cfg.parser == "array_table":
            df = self._parse_array_table(records, cfg)
        elif cfg.parser == "records":
            if isinstance(records, dict):
                records = [records]
            df = pd.json_normalize(records)
        else:
            raise PipelineError(f"unknown parser '{cfg.parser}' for source '{cfg.name}'")

        if cfg.field_map:
            df = df.rename(columns=cfg.field_map)

        if cfg.geo_id_fields:
            missing = [c for c in cfg.geo_id_fields if c not in df.columns]
            if missing:
                raise SchemaValidationError(
                    f"source '{cfg.name}' cannot build geo_id, missing columns: {missing}"
                )
            df["geo_id"] = df[list(cfg.geo_id_fields)].astype(str).agg("-".join, axis=1)

        self._validate_schema(df, cfg)

        df["__source"] = cfg.name
        df["__weight"] = cfg.weight
        df["__fetched_at"] = datetime.now(timezone.utc).isoformat()
        return df

    @staticmethod
    def _parse_array_table(records: Any, cfg: SourceConfig) -> pd.DataFrame:
        """Parse the "array of arrays, header row first" shape returned by
        the Census ACS API: [["NAME","B19013_001E",...], ["Autauga...", "58786", ...], ...]
        """
        if not isinstance(records, list) or len(records) < 1:
            raise SchemaValidationError(
                f"source '{cfg.name}' expected a non-empty list for array_table parsing"
            )
        header, *rows = records
        if not isinstance(header, list):
            raise SchemaValidationError(
                f"source '{cfg.name}' array_table header row is not a list"
            )
        return pd.DataFrame(rows, columns=header)

    @staticmethod
    def _validate_schema(df: pd.DataFrame, cfg: SourceConfig):
        """Presence + light dtype validation. A "numeric" field must exist
        AND at least half its values must survive pd.to_numeric coercion;
        otherwise we've probably parsed the wrong shape (e.g. the
        array-of-arrays bug this pipeline used to have, which produced
        integer column names and silently-empty numeric columns).
        """
        missing = [f for f, _kind in cfg.required_fields if f not in df.columns]
        if missing:
            raise SchemaValidationError(
                f"source '{cfg.name}' missing required fields: {missing}"
            )
        for fname, kind in cfg.required_fields:
            if kind != "numeric":
                continue
            coerced = pd.to_numeric(df[fname], errors="coerce")
            if len(coerced) == 0:
                continue
            valid_frac = coerced.notna().mean()
            if valid_frac < 0.5:
                raise SchemaValidationError(
                    f"source '{cfg.name}' field '{fname}' expected numeric data but only "
                    f"{valid_frac:.0%} of values were coercible -- likely a schema/parsing mismatch"
                )


class FeatureExtractor:
    """Coerces raw fields to numeric, derives socio-economic indicators,
    and flags statistical disparities.

    Design choices on disparity scoring (see also README):
    1. Every indicator column is standardized to a z-score BEFORE being
       combined, so the composite score isn't dominated by whichever raw
       indicator happens to have the largest natural scale.
    2. Indicators are sign-aligned (see INDICATOR_DIRECTIONS) so that a
       higher composite score always means "more disadvantaged", not a
       mix of "more advantaged on some axes, more disadvantaged on
       others" that happens to average out near zero.
    3. Rows are flagged when the composite score's absolute value exceeds
       `disparity_z_cutoff`, a fixed, documented parameter -- NOT a
       data-dependent quantile, so the flagged fraction is a genuine
       property of the data, not a fixed percentage baked in by
       construction.
    """

    NUMERIC_CANDIDATES = (
        "median_income", "unemployment_rate", "poverty_count", "poverty_universe",
        "population", "median_home_value", "internet_access_rate",
        "public_transit_score", "broadband_subscription_rate",
    )

    # Sign convention: +1 means "a higher raw value indicates MORE
    # disadvantage/disparity" (e.g. unemployment_rate); -1 means "a higher
    # raw value indicates LESS disadvantage" (e.g. income_poverty_gap_index
    # is high when a county is relatively well off, and
    # transit_access_per_capita is high when access is good). Every
    # indicator is standardized and then multiplied by its sign before
    # being averaged, so the composite consistently represents "degree of
    # disadvantage" instead of letting "good" and "bad" indicators cancel
    # each other out. (A naive unsigned mean was the original bug here --
    # see tests/test_pipeline.py for a regression test that catches it.)
    INDICATOR_DIRECTIONS = {
        "income_poverty_gap_index": -1,
        "digital_divide_score": 1,
        "unemployment_rate": 1,
        "transit_access_per_capita": -1,
    }
    INDICATOR_COLUMNS = tuple(INDICATOR_DIRECTIONS.keys())

    def __init__(self, disparity_z_cutoff: float):
        self.disparity_z_cutoff = disparity_z_cutoff

    def coerce_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.NUMERIC_CANDIDATES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def derive_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rate fields from raw counts where both halves are
        present (e.g. poverty_rate from poverty_count/poverty_universe).
        """
        df = df.copy()
        if {"poverty_count", "poverty_universe"}.issubset(df.columns):
            with np.errstate(divide="ignore", invalid="ignore"):
                df["poverty_rate"] = (
                    df["poverty_count"] / df["poverty_universe"].replace(0, np.nan) * 100
                )
        return df

    def engineer_operational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if {"median_income", "poverty_rate"}.issubset(df.columns):
            df["income_poverty_gap_index"] = (
                df["median_income"].rank(pct=True) - df["poverty_rate"].rank(pct=True)
            )
        if {"broadband_subscription_rate", "internet_access_rate"}.issubset(df.columns):
            df["digital_divide_score"] = (
                df["internet_access_rate"] - df["broadband_subscription_rate"]
            ).clip(lower=0)
        if {"public_transit_score", "population"}.issubset(df.columns):
            df["transit_access_per_capita"] = (
                df["public_transit_score"] / df["population"].replace(0, np.nan)
            )
        return df

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        std = series.std(ddof=0)
        if not std or np.isnan(std):
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - series.mean()) / std

    def flag_disparities(self, df: pd.DataFrame, z_cutoff: Optional[float] = None) -> pd.DataFrame:
        """`composite_disparity_score` is positive when a row is more
        disadvantaged than average and negative when it's more advantaged
        (see INDICATOR_DIRECTIONS). `disparity_flag` is set on the
        ABSOLUTE value of the composite exceeding `z_cutoff`, i.e. it
        flags rows that are unusual outliers in either direction, not
        only disadvantaged ones. Filter on the sign of
        composite_disparity_score downstream if you only want one side.
        """
        z_cutoff = self.disparity_z_cutoff if z_cutoff is None else z_cutoff
        df = df.copy()
        available = [c for c in self.INDICATOR_COLUMNS if c in df.columns]
        if not available:
            df["composite_disparity_score"] = np.nan
            df["disparity_flag"] = False
            return df

        standardized = pd.DataFrame(
            {
                c: self._zscore(pd.to_numeric(df[c], errors="coerce")) * self.INDICATOR_DIRECTIONS[c]
                for c in available
            },
            index=df.index,
        )
        composite = standardized.mean(axis=1)
        df["composite_disparity_score"] = composite
        df["disparity_flag"] = composite.abs() >= z_cutoff
        return df


class Deduplicator:
    """Resolves duplicate records (same key, e.g. same geography reported
    by more than one source) by keeping whichever row has the higher
    (field-completeness x source-weight) quality score.
    """

    @staticmethod
    def resolve(df: pd.DataFrame, key_cols: list) -> pd.DataFrame:
        if not key_cols or not all(c in df.columns for c in key_cols):
            return df.drop_duplicates()

        df = df.copy()
        weight = df["__weight"] if "__weight" in df.columns else pd.Series(1.0, index=df.index)
        # Internal metadata columns (__source, __weight, __fetched_at, ...)
        # are present on every row, so including them just adds a constant
        # to every score and doesn't change the ranking -- but excluding
        # them makes the score mean exactly what it claims to mean: "how
        # many real data fields are populated", not an inflated count that
        # happens to include bookkeeping columns.
        quality_cols = [c for c in df.columns if not c.startswith("__")]
        df["__quality_score"] = df[quality_cols].notna().sum(axis=1) * weight
        # kind="mergesort" is stable, so rows tied on quality score keep
        # their original relative order instead of an arbitrary one -
        # otherwise which row "wins" a tie would be undefined behavior.
        df = df.sort_values("__quality_score", ascending=False, kind="mergesort")
        return df.drop_duplicates(subset=key_cols, keep="first").drop(
            columns=["__quality_score"], errors="ignore"
        )


class ETLPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.cache = CacheStore(config.cache_db_path)
        self.client = DataSourceClient(self.cache, config.cache_ttl_seconds, config.fixture_dir)
        self.extractor = FeatureExtractor(config.disparity_z_cutoff)
        os.makedirs(config.output_dir, exist_ok=True)

    @timed
    def extract(self) -> list:
        if not self.config.fixture_dir:
            self.cache.prune_expired(self.config.cache_ttl_seconds)

        frames = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = {pool.submit(self._safe_fetch, cfg): cfg for cfg in self.config.sources}
            for future in concurrent.futures.as_completed(futures):
                cfg = futures[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        frames.append(df)
                except PipelineError as exc:
                    logger.error("dropping source '%s': %s", cfg.name, exc)
        if not frames:
            raise PipelineError("all sources failed; nothing to process")
        return frames

    def _safe_fetch(self, cfg: SourceConfig) -> Optional[pd.DataFrame]:
        try:
            return self.client.fetch(cfg)
        except (SourceUnavailableError, SchemaValidationError, PipelineError) as exc:
            logger.error("source '%s' failed: %s", cfg.name, exc)
            return None
        finally:
            # This worker thread won't be reused after the pool created in
            # extract() shuts down, so release its cache connection now
            # rather than leaving it to be silently GC'd.
            self.cache.close()

    @timed
    def transform(self, frames: list) -> tuple:
        """Returns (merged_df, metadata_dict). Metadata is returned
        explicitly rather than stashed on `df.attrs`, since attrs is an
        experimental pandas feature that isn't guaranteed to survive
        operations like concat/copy/to_parquet round-trips.
        """
        merged = pd.concat(frames, ignore_index=True, sort=False)
        merged = self.extractor.coerce_numeric(merged)
        merged = self.extractor.derive_rates(merged)
        merged = self.extractor.engineer_operational_features(merged)
        merged = self.extractor.flag_disparities(merged)

        key_candidates = [c for c in ("geo_id", "fips", "zip_code", "tract_id") if c in merged.columns]
        if key_candidates:
            merged = Deduplicator.resolve(merged, key_candidates)

        metadata = {
            "row_count": len(merged),
            "disparity_flagged": int(merged["disparity_flag"].sum()) if "disparity_flag" in merged.columns else 0,
            "sources": sorted(merged["__source"].unique().tolist()) if "__source" in merged.columns else [],
        }
        return merged, metadata

    @timed
    def load(self, df: pd.DataFrame, metadata: dict) -> dict:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        parquet_path = os.path.join(self.config.output_dir, f"civic_features_{timestamp}.parquet")
        csv_path = os.path.join(self.config.output_dir, f"civic_features_{timestamp}.csv")
        summary_path = os.path.join(self.config.output_dir, f"civic_summary_{timestamp}.json")

        try:
            df.to_parquet(parquet_path, index=False)
        except (ImportError, ModuleNotFoundError) as exc:
            # pyarrow (or fastparquet) isn't installed -- an expected,
            # documented fallback path (see requirements.txt), so this is
            # a warning, not an error.
            logger.warning("parquet unavailable (%s); falling back to csv only", exc)
            parquet_path = None
        except Exception as exc:
            # Anything else (a bad dtype, a corrupt buffer, ...) is a real
            # bug, not an expected missing-dependency case -- still fall
            # back to CSV so the run isn't lost, but log it louder so it
            # doesn't get silently swallowed alongside the expected case.
            logger.error("parquet export failed unexpectedly: %s", exc)
            parquet_path = None

        df.to_csv(csv_path, index=False)

        summary = {
            "generated_at": timestamp,
            "columns": list(df.columns),
            **metadata,
        }
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)

        return {"parquet": parquet_path, "csv": csv_path, "summary": summary_path}

    def run(self) -> dict:
        frames = self.extract()
        transformed_df, metadata = self.transform(frames)
        outputs = self.load(transformed_df, metadata)
        logger.info("pipeline complete: %s", outputs)
        return outputs


def build_config(
    output_dir: str = "civic_output",
    cache_db_path: str = "civic_pipeline_cache.sqlite3",
    cache_ttl: int = 3600,
    z_cutoff: float = 1.5,
    workers: int = 4,
    fixture_dir: Optional[str] = None,
) -> PipelineConfig:
    """Build the default source list plus a PipelineConfig from explicit
    values -- CLI args are threaded straight in here rather than mutating
    a config object after the fact.
    """
    sources = [
        SourceConfig(
            name="census_acs_income",
            url="https://api.census.gov/data/2022/acs/acs5",
            params={
                "get": "NAME,B19013_001E,B17001_002E,B17001_001E",
                "for": "county:*",
            },
            parser="array_table",
            field_map={
                "NAME": "county_name",
                "B19013_001E": "median_income",
                "B17001_002E": "poverty_count",
                "B17001_001E": "poverty_universe",
                "state": "state_fips",
                "county": "county_fips",
            },
            required_fields=(("median_income", "numeric"), ("county_name", "string")),
            geo_id_fields=("state_fips", "county_fips"),
            weight=1.2,
        ),
        # NOTE: the two sources below are TEMPLATES. Their URLs, params,
        # and field_map/geo_id_fields have NOT been verified against the
        # live HUD / FCC APIs -- confirm the real response shape before
        # relying on them (see README "Design notes / known limitations").
        SourceConfig(
            name="hud_fair_market_rents",
            url="https://www.huduser.gov/hudapi/public/fmr/data/2024",
            parser="records",
            field_map={},
            required_fields=(),
            geo_id_fields=(),
            weight=1.0,
        ),
        SourceConfig(
            name="fcc_broadband_map",
            url="https://broadbandmap.fcc.gov/api/public/map/downloads/1",
            parser="records",
            field_map={},
            required_fields=(),
            geo_id_fields=(),
            weight=0.9,
        ),
    ]
    return PipelineConfig(
        sources=sources,
        output_dir=output_dir,
        cache_db_path=cache_db_path,
        cache_ttl_seconds=cache_ttl,
        disparity_z_cutoff=z_cutoff,
        max_workers=workers,
        fixture_dir=fixture_dir,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Civic tech socio-economic ETL pipeline")
    parser.add_argument("--output-dir", default="civic_output")
    parser.add_argument("--cache-db", default="civic_pipeline_cache.sqlite3")
    parser.add_argument("--cache-ttl", type=int, default=3600)
    parser.add_argument("--z-cutoff", type=float, default=1.5,
                         help="Composite disparity z-score cutoff (default: 1.5 std dev)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fixture-dir", default=None,
                         help="Read sources from local JSON fixtures instead of the network")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = build_config(
        output_dir=args.output_dir,
        cache_db_path=args.cache_db,
        cache_ttl=args.cache_ttl,
        z_cutoff=args.z_cutoff,
        workers=args.workers,
        fixture_dir=args.fixture_dir,
    )

    pipeline = ETLPipeline(config)
    try:
        pipeline.run()
    except PipelineError as exc:
        logger.critical("pipeline aborted: %s", exc)
        sys.exit(1)
    finally:
        pipeline.cache.close()


if __name__ == "__main__":
    main()
