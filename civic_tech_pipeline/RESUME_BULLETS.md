CIVIC TECH DATA PIPELINE - Resume bullets (rewritten to match verified behavior)

Civic Tech Data Pipeline | Python, Pandas, NumPy, SQLite, Requests, pytest, Open Data APIs | Coding It Forward

- Built a concurrent, fault-tolerant ETL pipeline ingesting public Open
  Data REST APIs via a thread-safe thread pool, with per-source schema
  validation, failure isolation, exponential-backoff retry scoped to
  transient errors only, and a SQLite-backed TTL cache (WAL mode) to
  avoid redundant network calls.
- Parsed the Census ACS 5-Year API's array-of-arrays response format and
  mapped raw variable codes to canonical fields, deriving rates (e.g.
  poverty rate) from raw counts.
- Designed a standardized, sign-aligned composite disparity score so
  indicators on different scales and directions don't cancel each other
  out, flagged against a fixed, documented cutoff rather than a
  data-dependent quantile.
- Covered the pipeline with a 22-test pytest suite (mocked HTTP,
  fixture-based offline mode) covering parsing, caching, retry
  semantics, schema validation, dedup tie-breaking, and disparity
  scoring against hand-built expected outputs.
- Documented design tradeoffs and known limitations in a README,
  including which source adapters are verified versus templated.

Longer version with full detail lives in README.md -- keep the resume
bullets to 1-2 lines each; a reviewer/interviewer can ask about the
detail live.

NOTES ON WHAT I DELIBERATELY LEFT OUT
- No specific performance/benchmark numbers (e.g. query speedups,
  latency, throughput) are claimed, since none have been measured against
  a real deployment. If you later benchmark this against real
  infrastructure, add the real numbers then.
- No claim of production deployment, PostgreSQL/PostGIS storage, or
  GeoPandas/spatial processing -- none of that is implemented in this
  version. Add those bullets only once (and if) that work actually
  exists in the code.
- No claim about the HUD/FCC sources beyond "configured as templates" --
  their real schemas are unverified in this repo (see README).
