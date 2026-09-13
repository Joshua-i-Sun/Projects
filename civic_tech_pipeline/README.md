# Civic Tech Data Pipeline

A single-file ETL pipeline (`pipeline.py`) that pulls public Open Data
REST endpoints in parallel, caches responses, engineers socio-economic
indicator features, statistically flags disparities, deduplicates
overlapping records, and writes versioned Parquet/CSV extracts plus a
JSON run summary.

## Quickstart

```bash
pip install -r requirements.txt

# Offline / demo mode -- no network access required, reads from
# tests/fixtures/<source_name>.json instead of the live APIs
python pipeline.py --fixture-dir tests/fixtures --output-dir out

# Live mode (network access required, hits the real Census API and the
# HUD/FCC placeholders -- see "Unverified sources" below)
python pipeline.py --output-dir out
```

Run the tests:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Architecture

- **`CacheStore`** -- thread-safe SQLite response cache (WAL mode + busy
  timeout, so concurrent worker threads don't hit "database is locked").
  Expired rows are pruned once per pipeline run via `prune_expired`.
- **`DataSourceClient`** -- fetches (or, in offline mode, reads a fixture
  file for) each configured source, parses it into a canonical
  DataFrame, and validates its schema.
- **`FeatureExtractor`** -- coerces raw fields to numeric, derives rate
  fields (e.g. poverty_rate), engineers indicator columns, and computes
  the standardized composite disparity score.
- **`Deduplicator`** -- resolves duplicate records (rows sharing the same
  key, e.g. the same geography reported more than once) by keeping the
  row with the higher completeness x source-weight score.
- **`ETLPipeline`** -- orchestrates extract (parallel, per-source failure
  isolated) -> transform -> load.

## Changelog: second-round fixes

A follow-up review caught real issues the first pass missed:

- **`requests.Session` was shared across worker threads.** `Session` isn't
  guaranteed thread-safe under concurrent use; it's now thread-local
  (`DataSourceClient.session` property), matching the pattern already
  used for SQLite connections.
- **Retry no longer retries non-transient HTTP errors.** Previously any
  `requests.RequestException` was retried, which includes `HTTPError`
  from a 404/401/403 -- retrying an already-rejected request 4x wastes
  time and API quota. Only connection errors, timeouts, JSON decode
  errors, and *5xx* HTTP errors are retried now; other HTTP errors
  propagate immediately.
- **`CacheStore.close()`** releases a thread's SQLite connection instead
  of leaving it to be garbage-collected. Called from each worker thread
  after its fetch and from `main()` before exit. Trade-off: if the same
  worker thread handles a second source in the same run, it reconnects
  (cheap for SQLite) rather than staying open indefinitely.
- **Dedup quality score excludes internal bookkeeping columns**
  (`__source`, `__weight`, `__fetched_at`) so it measures exactly "how
  many real data fields are populated," not an inflated count that
  included metadata present on every row anyway. Tie-breaking now also
  uses a stable sort, so ties resolve to original row order instead of
  undefined behavior.
- **Parquet export failure is split into two cases**: missing
  `pyarrow`/`fastparquet` (expected, warning-level, documented fallback)
  versus any other failure (logged at error level, since that likely
  indicates a real bug rather than a missing optional dependency).

## Design notes / known limitations

**Dedup deduplicates; it does not join.** `Deduplicator.resolve` picks
the single best row per key from the concatenated frame -- it does not
merge complementary fields from different sources into one row. This
pipeline's dedup step assumes sources are *overlapping/redundant*
reports of the same entities (e.g. two different Open Data portals both
publishing county median income). If your sources are instead
*complementary* (source A has income, source B has broadband access,
for the same geography), as the bundled demo fixtures deliberately are,
the current design will not combine them into a single row -- whichever
source's row happens to score higher on completeness x weight simply
wins, and the other source's fields are dropped for that key. Run the
bundled offline demo and you'll see exactly this: only `census_acs_income`
survives dedup, because its rows have more populated columns than the
single-purpose HUD/FCC rows. If you need cross-source enrichment, add an
explicit `pd.merge` step on `geo_id` before feature engineering instead
of relying on `Deduplicator`.

**Disparity scoring is intentionally two things: standardized, and
sign-aligned.** Every indicator is first converted to a z-score so no
single indicator's raw scale dominates the composite. Indicators are
also multiplied by a documented direction (`FeatureExtractor.
INDICATOR_DIRECTIONS`) so "higher = more disadvantaged" and "higher =
more advantaged" indicators don't cancel out in the average -- an
earlier version of this pipeline mixed both without sign-aligning them,
which could mask real disparities (see the regression test
`test_flag_disparities_standardizes_and_sign_aligns_before_combining`).
`disparity_flag` is set on the *absolute* value of the composite
exceeding `disparity_z_cutoff` (default 1.5), so it flags outliers in
either direction, not only disadvantaged ones -- filter on the sign of
`composite_disparity_score` downstream if you only want one side.
`disparity_z_cutoff` is a fixed, documented parameter, not a
data-dependent quantile, so the fraction of rows flagged is a genuine
property of the input data.

**Schema validation is presence + light dtype checking, not strict
typing.** A numeric field must exist and at least half its values must
survive `pd.to_numeric` coercion. This catches "wrong shape entirely"
(e.g. the old Census array-of-arrays bug, which produced integer column
names) without failing a source just because it has some missing
values, which is normal in public data.

**Unverified sources.** Only the Census ACS 5-Year adapter
(`array_table` parser) is verified against a real, documented response
shape and covered by a fixture-based test. The `hud_fair_market_rents`
and `fcc_broadband_map` `SourceConfig` entries in `build_config()` are
**templates** -- their URLs are plausible endpoints but their
`field_map`, `required_fields`, and `geo_id_fields` are left empty
because I have not confirmed their actual live response schema. Treat
them as a starting point: point them at a real response, fill in the
mapping, and add a fixture + test the same way `census_acs_income` is
covered before trusting their output.

**No benchmark numbers are claimed anywhere in this repo.** I have not
measured throughput, latency, or query performance against a real
deployment, so none of those claims appear in the code, README, or
resume bullets below. If you run this against real infrastructure and
collect real numbers, add them yourself.

## Offline fixtures

`tests/fixtures/*.json` contain small, synthetic, hand-written sample
payloads:

- `census_acs_income.json` mirrors the **real, documented** Census ACS
  "array of arrays, header row first" response shape.
- `hud_fair_market_rents.json` and `fcc_broadband_map.json` are
  synthetic placeholders (see "Unverified sources" above) shaped to
  exercise the `records` parser and the dedup/disparity logic in tests
  and the offline demo -- they are not real HUD/FCC data.
