import json
import time
import sqlite3

import numpy as np
import pandas as pd
import pytest
import requests

import pipeline as pl


# ---------------------------------------------------------------------------
# 1. Census ACS array-of-arrays parsing
# ---------------------------------------------------------------------------

def test_census_array_table_parsing_produces_canonical_numeric_columns():
    cfg = pl.SourceConfig(
        name="census_acs_income",
        url="https://api.census.gov/data/2022/acs/acs5",
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
    )
    raw = [
        ["NAME", "B19013_001E", "B17001_002E", "B17001_001E", "state", "county"],
        ["Autauga County, Alabama", "58786", "6497", "54983", "01", "001"],
        ["Baldwin County, Alabama", "66129", "19406", "223752", "01", "003"],
    ]

    client = pl.DataSourceClient(cache=None, ttl=0, fixture_dir=None)
    client._get_raw = lambda c: raw  # bypass network/cache entirely

    df = client.fetch(cfg)

    assert list(df["county_name"]) == ["Autauga County, Alabama", "Baldwin County, Alabama"]
    assert df["geo_id"].tolist() == ["01-001", "01-003"]

    numeric = pd.to_numeric(df["median_income"], errors="coerce")
    assert numeric.notna().all()
    assert numeric.tolist() == [58786.0, 66129.0]


def test_array_table_with_bad_header_type_raises_schema_error():
    cfg = pl.SourceConfig(name="bad", url="x", parser="array_table")
    client = pl.DataSourceClient(cache=None, ttl=0, fixture_dir=None)
    client._get_raw = lambda c: [{"not": "a list header"}]
    with pytest.raises(pl.SchemaValidationError):
        client.fetch(cfg)


# ---------------------------------------------------------------------------
# 2. Cache hit / miss / TTL expiry
# ---------------------------------------------------------------------------

def test_cache_miss_then_hit_avoids_second_network_call(tmp_path, monkeypatch):
    cache = pl.CacheStore(str(tmp_path / "cache.sqlite3"))
    client = pl.DataSourceClient(cache=cache, ttl=3600, fixture_dir=None)

    cfg = pl.SourceConfig(name="s1", url="https://example.com/data")
    call_count = {"n": 0}

    def fake_fetch_raw(self, c):
        call_count["n"] += 1
        return {"rows": [{"a": 1}]}

    monkeypatch.setattr(pl.DataSourceClient, "_fetch_raw", fake_fetch_raw)

    raw1 = client._get_raw(cfg)
    raw2 = client._get_raw(cfg)

    assert call_count["n"] == 1
    assert raw1 == raw2 == {"rows": [{"a": 1}]}


def test_cache_expiry_forces_refetch(tmp_path, monkeypatch):
    cache = pl.CacheStore(str(tmp_path / "cache.sqlite3"))
    client = pl.DataSourceClient(cache=cache, ttl=1, fixture_dir=None)
    cfg = pl.SourceConfig(name="s1", url="https://example.com/data")

    call_count = {"n": 0}

    def fake_fetch_raw(self, c):
        call_count["n"] += 1
        return {"n": call_count["n"]}

    monkeypatch.setattr(pl.DataSourceClient, "_fetch_raw", fake_fetch_raw)

    client._get_raw(cfg)
    time.sleep(1.2)
    client._get_raw(cfg)

    assert call_count["n"] == 2


def test_cache_prune_expired_removes_old_rows(tmp_path):
    cache = pl.CacheStore(str(tmp_path / "cache.sqlite3"))
    cache.set("k1", {"v": 1})
    conn = cache.conn
    conn.execute("UPDATE cache SET fetched_at = fetched_at - 10000 WHERE key = ?", ("k1",))
    conn.commit()

    cache.prune_expired(ttl=100)

    row = conn.execute("SELECT * FROM cache WHERE key = ?", ("k1",)).fetchone()
    assert row is None


def test_cache_uses_wal_mode(tmp_path):
    cache = pl.CacheStore(str(tmp_path / "cache.sqlite3"))
    mode = cache.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


# ---------------------------------------------------------------------------
# 3. Retry exhaustion / narrowed exception types
# ---------------------------------------------------------------------------

def test_retry_exhausts_and_wraps_request_exception(monkeypatch):
    attempts = {"n": 0}

    @pl.retry(max_attempts=3, base=1.01)
    def flaky():
        attempts["n"] += 1
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(pl.time, "sleep", lambda s: None)

    with pytest.raises(pl.SourceUnavailableError):
        flaky()
    assert attempts["n"] == 3


def test_retry_does_not_catch_unrelated_exceptions():
    @pl.retry(max_attempts=3, base=1.01)
    def broken():
        raise KeyError("not a network problem")

    with pytest.raises(KeyError):
        broken()


# ---------------------------------------------------------------------------
# 4. Schema validation failure is isolated, not fatal to the whole run
# ---------------------------------------------------------------------------

def test_bad_source_is_dropped_but_pipeline_still_produces_output(tmp_path):
    good_cfg = pl.SourceConfig(
        name="good",
        url="x",
        parser="records",
        required_fields=(("value", "numeric"),),
    )
    bad_cfg = pl.SourceConfig(
        name="bad",
        url="y",
        parser="records",
        required_fields=(("value", "numeric"),),
    )

    config = pl.PipelineConfig(
        sources=[good_cfg, bad_cfg],
        output_dir=str(tmp_path / "out"),
        cache_db_path=str(tmp_path / "cache.sqlite3"),
        fixture_dir=str(tmp_path / "fixtures"),
    )
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "good.json").write_text(json.dumps([{"value": "10"}, {"value": "20"}]))
    (fixtures_dir / "bad.json").write_text(json.dumps([{"value": "not-a-number"}, {"value": "also-not"}]))

    pipeline = pl.ETLPipeline(config)
    frames = pipeline.extract()

    assert len(frames) == 1
    assert frames[0]["__source"].iloc[0] == "good"


def test_missing_required_field_raises_schema_error():
    cfg = pl.SourceConfig(name="s", url="x", required_fields=(("missing_col", "numeric"),))
    client = pl.DataSourceClient(cache=None, ttl=0, fixture_dir=None)
    client._get_raw = lambda c: [{"present_col": 1}]
    with pytest.raises(pl.SchemaValidationError):
        client.fetch(cfg)


# ---------------------------------------------------------------------------
# 5. Dedup tie-breaking
# ---------------------------------------------------------------------------

def test_dedup_keeps_more_complete_higher_weight_row():
    df = pd.DataFrame([
        {"geo_id": "01-001", "median_income": 50000, "poverty_rate": None, "__weight": 1.0},
        {"geo_id": "01-001", "median_income": 50000, "poverty_rate": 12.5, "__weight": 1.2},
        {"geo_id": "02-002", "median_income": 70000, "poverty_rate": 8.0, "__weight": 1.0},
    ])

    result = pl.Deduplicator.resolve(df, key_cols=["geo_id"])

    assert len(result) == 2
    row = result[result["geo_id"] == "01-001"].iloc[0]
    assert row["poverty_rate"] == 12.5
    assert row["__weight"] == 1.2


def test_dedup_falls_back_to_plain_drop_duplicates_without_key_cols():
    df = pd.DataFrame([{"a": 1}, {"a": 1}, {"a": 2}])
    result = pl.Deduplicator.resolve(df, key_cols=["nonexistent"])
    assert len(result) == 2


# ---------------------------------------------------------------------------
# 6. Disparity scoring on a hand-built frame with a known expected output
# ---------------------------------------------------------------------------

def test_flag_disparities_standardizes_and_sign_aligns_before_combining():
    # Row 2 (index 2) is the clear "worse off" outlier on every indicator:
    # lowest income-poverty gap, highest digital divide, highest
    # unemployment, and lowest transit access. income_poverty_gap_index and
    # transit_access_per_capita are "higher = more advantaged" indicators,
    # so they must be sign-flipped before averaging -- a naive unsigned
    # mean would let them cancel out digital_divide/unemployment and hide
    # the disparity entirely (this was a real bug caught by this test).
    df = pd.DataFrame({
        "income_poverty_gap_index": [0.9, 0.5, 0.1],
        "digital_divide_score": [5.0, 10.0, 45.0],
        "unemployment_rate": [3.0, 5.0, 12.0],
        "transit_access_per_capita": [0.002, 0.0015, 0.0002],
    })
    extractor = pl.FeatureExtractor(disparity_z_cutoff=1.0)

    result = extractor.flag_disparities(df)

    expected_composite = pd.DataFrame({
        c: pl.FeatureExtractor._zscore(df[c]) * pl.FeatureExtractor.INDICATOR_DIRECTIONS[c]
        for c in pl.FeatureExtractor.INDICATOR_COLUMNS
    }).mean(axis=1)

    pd.testing.assert_series_equal(
        result["composite_disparity_score"].reset_index(drop=True),
        expected_composite.reset_index(drop=True),
        check_names=False,
    )
    assert result["disparity_flag"].tolist() == (expected_composite.abs() >= 1.0).tolist()
    assert result.loc[2, "disparity_flag"] == True  # noqa: E712 -- row 3 is the clear outlier
    assert result.loc[2, "composite_disparity_score"] > 0  # positive = more disadvantaged


def test_flag_disparities_cutoff_is_not_derived_from_quantile():
    # Regression test for the old bug: quantile(min(threshold/2, 0.99)) always
    # flagged ~37.5% of rows for the default threshold of 1.25, regardless of
    # the underlying data. Uniform, non-outlier data should flag ~nothing at
    # a z-cutoff of 1.5.
    n = 50
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "income_poverty_gap_index": rng.normal(0, 0.01, n),
        "digital_divide_score": rng.normal(0, 0.01, n),
        "unemployment_rate": rng.normal(5, 0.01, n),
        "transit_access_per_capita": rng.normal(0.001, 0.00001, n),
    })
    extractor = pl.FeatureExtractor(disparity_z_cutoff=1.5)
    result = extractor.flag_disparities(df)
    flagged_fraction = result["disparity_flag"].mean()
    assert flagged_fraction < 0.15


def test_flag_disparities_with_no_indicator_columns_present():
    df = pd.DataFrame({"unrelated_column": [1, 2, 3]})
    extractor = pl.FeatureExtractor(disparity_z_cutoff=1.5)
    result = extractor.flag_disparities(df)
    assert result["disparity_flag"].tolist() == [False, False, False]


# ---------------------------------------------------------------------------
# record_path traversal
# ---------------------------------------------------------------------------

def test_record_path_supports_integer_list_index():
    data = {"results": [{"rows": [{"a": 1}, {"a": 2}]}]}
    out = pl._traverse_record_path(data, "results.0.rows")
    assert out == [{"a": 1}, {"a": 2}]


def test_record_path_raises_clear_error_on_bad_segment():
    data = {"results": [1, 2, 3]}
    with pytest.raises(pl.PipelineError):
        pl._traverse_record_path(data, "results.not_an_index")


# ---------------------------------------------------------------------------
# Session is thread-local (regression test for the shared-Session bug)
# ---------------------------------------------------------------------------

def test_session_is_thread_local_not_shared():
    client = pl.DataSourceClient(cache=None, ttl=0, fixture_dir=None)
    sessions = {}

    def grab_session(thread_name):
        sessions[thread_name] = client.session

    import threading as _threading
    t1 = _threading.Thread(target=grab_session, args=("t1",))
    t2 = _threading.Thread(target=grab_session, args=("t2",))
    t1.start(); t1.join()
    t2.start(); t2.join()

    assert sessions["t1"] is not sessions["t2"]
    assert isinstance(sessions["t1"], requests.Session)


# ---------------------------------------------------------------------------
# Retry only retries transient errors, not non-transient 4xx HTTP errors
# ---------------------------------------------------------------------------

def test_retry_does_not_retry_non_transient_4xx(monkeypatch):
    attempts = {"n": 0}
    monkeypatch.setattr(pl.time, "sleep", lambda s: None)

    resp = requests.Response()
    resp.status_code = 404

    @pl.retry(max_attempts=3, base=1.01)
    def flaky():
        attempts["n"] += 1
        raise requests.HTTPError("not found", response=resp)

    with pytest.raises(requests.HTTPError):
        flaky()
    assert attempts["n"] == 1  # no retries -- fails immediately


def test_retry_does_retry_transient_5xx(monkeypatch):
    attempts = {"n": 0}
    monkeypatch.setattr(pl.time, "sleep", lambda s: None)

    resp = requests.Response()
    resp.status_code = 503

    @pl.retry(max_attempts=3, base=1.01)
    def flaky():
        attempts["n"] += 1
        raise requests.HTTPError("service unavailable", response=resp)

    with pytest.raises(pl.SourceUnavailableError):
        flaky()
    assert attempts["n"] == 3


# ---------------------------------------------------------------------------
# CacheStore.close()
# ---------------------------------------------------------------------------

def test_cache_close_releases_thread_local_connection(tmp_path):
    cache = pl.CacheStore(str(tmp_path / "cache.sqlite3"))
    cache.set("k", {"v": 1})
    assert hasattr(cache._local, "conn")

    cache.close()

    assert not hasattr(cache._local, "conn")
    # subsequent use transparently reconnects
    assert cache.get("k", ttl=3600) == {"v": 1}


# ---------------------------------------------------------------------------
# Dedup excludes internal bookkeeping columns from the quality score
# ---------------------------------------------------------------------------

def test_dedup_quality_score_ignores_internal_columns():
    # Both rows have identical real data completeness (one populated field,
    # one missing); only __fetched_at differs between them, which must NOT
    # influence which row wins.
    df = pd.DataFrame([
        {"geo_id": "01-001", "median_income": 50000, "poverty_rate": None,
         "__weight": 1.0, "__source": "a", "__fetched_at": "2026-01-01"},
        {"geo_id": "01-001", "median_income": 50000, "poverty_rate": None,
         "__weight": 1.0, "__source": "b", "__fetched_at": "2026-01-02T00:00:00.123456"},
    ])
    result = pl.Deduplicator.resolve(df, key_cols=["geo_id"])
    assert len(result) == 1
    # both rows tie on real-data completeness x weight; "first" after a
    # stable sort on equal scores keeps original row order -- source "a"
    assert result.iloc[0]["__source"] == "a"
