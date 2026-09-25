# monitoring

Records what each pipeline run did and reports it.

## Why

Before this, a run left almost no evidence of itself:

- `run_pipeline.py` caught every per-dataset exception and always exited 0, so cron could
  not tell a clean run from one where every dataset failed.
- `init_pipeline` called bare `exit()` when Solr was down or credentials were missing —
  also exit code 0, and with no Solr trace at all.
- The per-stage status strings (`harvester_status`, the transformation `pipeline_cleanup`,
  `get_agg_status`) are **cumulative**: they count every doc ever written for a dataset, so
  one historical failure makes them read "N failed" forever. They cannot answer "did *this*
  run go well".

## Artifacts

Each run writes:

| Artifact | Role |
|---|---|
| `logs/<run_id>/run_summary.json` | Canonical. Written unconditionally, including on abort — so "Solr was down" leaves a record. |
| `type_s:pipeline_run` Solr doc | Mirror for the dashboard and run history. Best-effort; a Solr failure never changes the exit code. |

`run_id` is the `logs/<run_id>/` directory name (local time). Every `*_dt` field is UTC,
matching the rest of the pipeline. Don't compare the two.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | every stage ran, no per-run failures |
| 1 | one or more dataset/stage failures |
| 2 | precondition abort — Solr unreachable, missing `.netrc` / `OUTPUT_DIR` / grids |
| 3 | unexpected fatal error |

## Counting a run's activity

Two independent sources, deliberately:

- **Timestamp windows (primary)** — `last_attempt_dt`, `transformation_started_dt`,
  `aggregation_time_dt`, bounded at the run's start. Every stage already writes these on
  every doc it writes, so this source *cannot miss a write site*. It can over-attribute
  when two runs overlap.
- **`run_id_s` stamps (cross-check)** — exact attribution, but silently under-counts if a
  write site is ever added without a stamp.

Under-reporting failures is the one direction a monitoring system cannot tolerate, so the
timestamp window is primary and `run_id_s` is the check. Disagreements land in
`totals.count_divergence`.

The comparison is **one-directional**: only `timestamp > run_id` is flagged. The reverse is
expected — `reconstruct_tx_solr_doc` and `reconstruct_agg_solr_doc` write a doc *now* but
backdate its activity timestamp to the output file's mtime, so a reconstructed doc carries
this run's id while falling outside this run's window.

Log ERROR/CRITICAL line counts are collected as a third, weaker signal — they catch a
worker killed outright, which dies before writing its result doc. They are reported but
never drive the exit code, since recoverable conditions are logged at ERROR level in
places.

## Notifications

```
python ecco_pipeline/monitoring/notify.py --run-id latest [--dry-run]
```

Run by cron *after* `run_pipeline.py` (see `scripts/run_pipeline_cron.sh`), never from
inside it: a hung SMTP relay then cannot affect a pipeline run, and a digest can be
re-rendered for any past run. Configure via `conf/global_settings.py` (see the
`.example`); unconfigured, it logs a warning and does nothing.

Keep `NOTIFY_ON = 'always'`. There is no external dead-man's switch, so a run that never
starts produces no signal — a digest you expect every week is the only thing that makes its
own absence noticeable. The digest also reports the gap since the previous run, so a missed
run surfaces on the next one.

## Known gaps

- **No dead-man's switch.** A run that never starts reports nothing until the following
  run notes the gap.
- **Upstream staleness reads as success.** A dataset whose source stops publishing keeps
  reporting "All granules successfully harvested" (cumulative status) while `end_date_dt`
  stops advancing. A zero-activity run likewise reports `ok`.
- **No cron overlap guard or run timeout.** Overlapping runs cross-attribute; the
  divergence check is what surfaces it.
- **`OUTPUT_DIR` free space is unmonitored.**
