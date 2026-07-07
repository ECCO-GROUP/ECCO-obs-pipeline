import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import requests
import yaml
from conf.global_settings import SOLR_COLLECTION, SOLR_HOST

logger = logging.getLogger("pipeline")

_TIMEOUT = 60
_MAX_RETRIES = 3           # retries after the initial attempt
_BACKOFF_BASE = 2          # seconds; exponential backoff: 2, 4, 8, ...
_COMMIT_WITHIN_MS = 15000  # let Solr batch commits instead of one hard commit per write

# One requests.Session per process for connection reuse (keep-alive). Keyed on PID so
# forked transformation/aggregation workers each build their own session rather than
# inheriting — and corrupting — the parent's open sockets.
_session = None
_session_pid = None


def _get_session() -> requests.Session:
    global _session, _session_pid
    pid = os.getpid()
    if _session is None or _session_pid != pid:
        _session = requests.Session()
        _session_pid = pid
    return _session


def _request(method: str, url: str, **kwargs):
    """
    Wrapper for all Solr requests: reuses a per-process keep-alive session and retries
    transient failures (timeouts, connection errors) with exponential backoff. HTTP
    error statuses are returned as-is, matching prior behavior.
    """
    kwargs.setdefault("timeout", _TIMEOUT)
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return _get_session().request(method, url, **kwargs)
        except Exception as e:
            if attempt == _MAX_RETRIES:
                logger.error(f"Solr request failed after {_MAX_RETRIES + 1} attempts: {e}")
                raise
            delay = _BACKOFF_BASE * (2 ** attempt)
            logger.warning(f"Solr request failed ({e}), retry {attempt + 1}/{_MAX_RETRIES} in {delay}s...")
            time.sleep(delay)


_QUERY_BATCH = 10000  # cursorMark page size for full (unbounded) queries


def _query_page(url: str, params: dict) -> dict:
    """GET one Solr result page, raising a clear error on a non-OK response."""
    response = _request("GET", url, params=params)
    if not response.ok:
        raise RuntimeError(
            f"Solr query failed ({response.status_code}) for fq={params.get('fq')}: "
            f"{response.text[:500]}"
        )
    return response.json()


def solr_query(fq: Iterable[str], fl: str = "", rows: int | None = None) -> list[dict]:
    """
    Query Solr and return matching docs.

    By default returns *all* matching docs via cursorMark deep paging, so large result
    sets come back complete and in bounded batches — no silent truncation, no single
    huge response. Pass rows=N for a bounded single-page fetch when only a few docs are
    needed (e.g. a single-doc lookup: solr_query(fq, rows=1)[0]).
    """
    url = f"{SOLR_HOST}{SOLR_COLLECTION}/select?"

    if rows is not None:
        params = {"q": "*:*", "fq": fq, "fl": fl, "rows": rows}
        return _query_page(url, params)["response"]["docs"]

    params = {
        "q": "*:*",
        "fq": fq,
        "fl": fl,
        "sort": "id asc",  # cursorMark requires a deterministic sort on the uniqueKey
        "rows": _QUERY_BATCH,
        "cursorMark": "*",
    }
    docs = []
    while True:
        body = _query_page(url, params)
        docs.extend(body["response"]["docs"])
        next_cursor = body.get("nextCursorMark")
        # cursorMark is exhausted when the returned mark stops advancing
        if not next_cursor or next_cursor == params["cursorMark"]:
            break
        params["cursorMark"] = next_cursor
    return docs


def solr_count(fq: Iterable[str]) -> int:
    """
    Return the number of documents matching fq without fetching any docs.
    """
    params = {"q": "*:*", "fq": fq, "rows": 0}
    url = f"{SOLR_HOST}{SOLR_COLLECTION}/select?"
    return _query_page(url, params)["response"]["numFound"]


def solr_update(update_body: Iterable[dict], r: bool = False, commit: bool = True):
    """
    Submit update to Solr.

    commit=True forces an immediate hard commit — required when the caller reads the
    written doc back in the same run (e.g. prepopulate_solr followed by solr_query).
    commit=False uses commitWithin so Solr batches the commit; use it for high-frequency
    writes with no read-your-write dependency to avoid a commit/searcher-warming storm.
    """
    if commit:
        url = f"{SOLR_HOST}{SOLR_COLLECTION}/update?commit=true"
    else:
        url = f"{SOLR_HOST}{SOLR_COLLECTION}/update?commitWithin={_COMMIT_WITHIN_MS}"
    response = _request("POST", url, json=update_body)
    if r:
        return response


def commit_solr():
    """
    Force an immediate hard commit, flushing any writes made with commitWithin
    (commit=False) that Solr has not yet committed.

    Call this at a run boundary before reading back state that depends on those
    deferred writes — e.g. after the transformation Pool finishes and before
    pipeline_cleanup counts success_b:false docs. Without it, a still-pending
    commitWithin batch makes just-completed transformations transiently read as
    failed. It is a single commit, not a per-write one, so it does not reintroduce
    the commit/searcher-warming storm that commitWithin exists to avoid.
    """
    url = f"{SOLR_HOST}{SOLR_COLLECTION}/update?commit=true"
    _request("POST", url, json={"commit": {}})


def ping_solr():
    """
    Ping Solr to ensure it is running
    """
    url = f"{SOLR_HOST}{SOLR_COLLECTION}/admin/ping"
    requests.get(url)


def collection_check() -> bool:
    """
    Check if collection exists on Solr
    """
    url = f"{SOLR_HOST}admin/collections?action=LIST"
    response = requests.get(url).json()
    if SOLR_COLLECTION in response["collections"]:
        return True
    return False


def check_grids():
    """
    Check if grids have been written to Solr
    """
    if solr_count(["type_s:grid"]) == 0:
        return True
    return False


def validate_granules():
    granules = solr_query(["type_s:granule"], fl="id,pre_transformation_file_path_s")
    docs_to_remove = []

    for granule in granules:
        file_path = granule["pre_transformation_file_path_s"]
        if os.path.exists(file_path):
            continue
        else:
            docs_to_remove.append(granule["id"])

    if docs_to_remove:
        solr_update({"delete": docs_to_remove})


def clean_solr(config):
    """
    Remove harvested and transformed entries in Solr for dates
    outside of config date range. Also remove related aggregations, and force
    aggregation rerun for those years.
    """
    dataset_name = config["ds_name"]
    config_start = config["start"]
    config_end = config["end"]

    if config_end == "NOW":
        config_end = datetime.utcnow().strftime("%Y%m%dT%H:%M:%SZ")

    # Convert config dates to Solr format
    config_start = f"{config_start[:4]}-{config_start[4:6]}-{config_start[6:]}"
    config_end = f"{config_end[:4]}-{config_end[4:6]}-{config_end[6:]}"

    fq = ["type_s:dataset", f"dataset_s:{dataset_name}"]
    dataset_metadata = solr_query(fq)

    if not dataset_metadata:
        return
    else:
        dataset_metadata = dataset_metadata[0]

    # Remove entries earlier than config start date
    fq = f"dataset_s:{dataset_name} AND date_dt:[* TO {config_start}T00:00:00Z}}"
    url = f"{SOLR_HOST}{SOLR_COLLECTION}/update?commit=true"
    requests.post(url, json={"delete": {"query": fq}})

    # Remove entries later than config end date
    fq = f"dataset_s:{dataset_name} AND date_dt:{{{config_end}T00:00:00Z TO *]"
    url = f"{SOLR_HOST}{SOLR_COLLECTION}/update?commit=true"
    requests.post(url, json={"delete": {"query": fq}})


def delete_mismatch_transformations():
    """
    Function called when using the wipe_transformations pipeline argument. Queries
    Solr for all transformation entries for the given dataset and compares the
    transformation version in Solr and in the config YAML. If they differ, the
    function deletes the transformed file from disk and the entry from Solr.
    """
    datasets = [
        os.path.splitext(ds)[0]
        for ds in os.listdir("conf/ds_configs")
        if ds != ".DS_Store" and "tpl" not in ds
    ]
    datasets.sort()

    for ds in datasets:
        with open(Path(f"conf/ds_configs/{ds}.yaml"), "r") as stream:
            config = yaml.load(stream, yaml.Loader)
        dataset_name = config["ds_name"]
        config_version = config["t_version"]

        # Query for existing transformations
        fq = [f"dataset_s:{dataset_name}", "type_s:transformation"]
        transformations = solr_query(fq)

        for transformation in transformations:
            if transformation["transformation_version_f"] != config_version:
                # Remove file from disk
                if os.path.exists(transformation["transformation_file_path_s"]):
                    os.remove(transformation["transformation_file_path_s"])

                # Remove transformation entry from Solr
                url = f"{SOLR_HOST}{SOLR_COLLECTION}/update?commit=true"
                requests.post(url, json={"delete": [transformation["id"]]})
