"""
Thin wrapper around solr_utils that returns pandas DataFrames.
All Solr field names are normalised here so views don't need to know them.
"""
import json
from datetime import datetime, timedelta, timezone
from glob import glob
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

_CONF_DIR = Path(__file__).resolve().parent.parent / "conf" / "ds_configs"

SOLR_HOST = None  # set at import time by calling configure()
SOLR_COLLECTION = None


def configure(host: str, collection: str):
    global SOLR_HOST, SOLR_COLLECTION
    SOLR_HOST = host
    SOLR_COLLECTION = collection


def get_active_dataset_names() -> set[str]:
    """
    Names of datasets with an active (non-deprecated) config. The pipeline
    uses the same top-level glob to drive harvest/transform/aggregate, so
    presence here is the canonical "is this dataset active" signal — Solr
    docs for deprecated datasets stay around but aren't shown in the dashboard.
    """
    return {Path(f).stem for f in glob(str(_CONF_DIR / "*.yaml"))}


def _filter_active(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "dataset_s" not in df.columns:
        return df
    return df[df["dataset_s"].isin(get_active_dataset_names())]


def _query(fq: list[str], fl: str = "") -> list[dict]:
    params = {"q": "*:*", "fq": fq, "rows": 300000}
    if fl:
        params["fl"] = fl
    url = f"{SOLR_HOST}{SOLR_COLLECTION}/select?"
    try:
        resp = requests.get(url, params=params, headers={"Connection": "close"}, timeout=10)
        resp.raise_for_status()
        return resp.json()["response"]["docs"]
    except Exception as e:
        return []


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def get_datasets() -> pd.DataFrame:
    docs = _query(["type_s:dataset"])
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    # Normalise expected columns that may not exist yet
    for col in [
        "dataset_s", "harvest_status_s", "transformation_status_s",
        "aggregation_status_s", "last_checked_dt", "last_transformation_dt",
        "last_aggregation_dt", "n_granules_i", "n_granules_success_i",
        "n_granules_failed_i", "harvester_type_s", "ecco_variable_s",
    ]:
        if col not in df.columns:
            df[col] = None
    for col in ["last_checked_dt", "last_transformation_dt", "last_aggregation_dt"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return _filter_active(df)


# ---------------------------------------------------------------------------
# Pipeline-wide aggregates
# ---------------------------------------------------------------------------

def _count(fq: list[str]) -> int:
    """Number of docs matching fq, without fetching any docs."""
    params = {"q": "*:*", "fq": fq, "rows": 0}
    url = f"{SOLR_HOST}{SOLR_COLLECTION}/select?"
    try:
        resp = requests.get(url, params=params, headers={"Connection": "close"}, timeout=10)
        resp.raise_for_status()
        return resp.json()["response"]["numFound"]
    except Exception:
        return 0


def get_total_counts() -> dict:
    """All-time document counts per pipeline stage."""
    return {
        "granules": _count(["type_s:granule"]),
        "transformations": _count(["type_s:transformation"]),
        "aggregations": _count(["type_s:aggregation"]),
        "datasets": _count(["type_s:dataset"]),
    }


def get_harvest_coverage() -> pd.DataFrame:
    """
    First and last successfully-harvested granule date per dataset, plus a
    granule count. Resolved in a single Solr JSON Facet query rather than
    pulling every granule doc.
    """
    facet = {
        "datasets": {
            "type": "terms",
            "field": "dataset_s",
            "limit": -1,
            "facet": {"first": "min(date_dt)", "last": "max(date_dt)"},
        }
    }
    params = {
        "q": "*:*",
        "fq": ["type_s:granule", "harvest_success_b:true"],
        "rows": 0,
        "json.facet": json.dumps(facet),
    }
    url = f"{SOLR_HOST}{SOLR_COLLECTION}/select?"
    cols = ["dataset_s", "start", "end", "count"]
    try:
        resp = requests.get(url, params=params, headers={"Connection": "close"}, timeout=10)
        resp.raise_for_status()
        buckets = resp.json()["facets"]["datasets"]["buckets"]
    except Exception:
        return pd.DataFrame(columns=cols)
    if not buckets:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(
        [
            {
                "dataset_s": b["val"],
                "start": b.get("first"),
                "end": b.get("last"),
                "count": b.get("count", 0),
            }
            for b in buckets
        ]
    )
    df["start"] = pd.to_datetime(df["start"], errors="coerce", utc=True)
    df["end"] = pd.to_datetime(df["end"], errors="coerce", utc=True)
    return _filter_active(df)


# ---------------------------------------------------------------------------
# Recent activity  (last N days, all datasets)
# ---------------------------------------------------------------------------

def _since(days: int) -> str:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")


def get_recent_granules(days: int = 7) -> pd.DataFrame:
    since = _since(days)
    # Filter on last_attempt_dt (set unconditionally) rather than download_time_dt
    # (set only on success) so first-time failures are visible.
    docs = _query(
        ["type_s:granule", f"last_attempt_dt:[{since} TO *]"],
        fl="dataset_s,date_dt,filename_s,harvest_success_b,error_message_s,download_time_dt,last_attempt_dt,download_duration_i",
    )
    if not docs:
        return pd.DataFrame(columns=["dataset_s", "date_dt", "filename_s",
                                      "harvest_success_b", "error_message_s",
                                      "download_time_dt", "last_attempt_dt"])
    df = pd.DataFrame(docs)
    df = _ensure_columns(df, {
        "dataset_s": "", "filename_s": "", "error_message_s": "",
        "harvest_success_b": True, "download_time_dt": None,
        "last_attempt_dt": None,
    })
    df["download_time_dt"] = pd.to_datetime(df["download_time_dt"], errors="coerce", utc=True)
    df["last_attempt_dt"] = pd.to_datetime(df["last_attempt_dt"], errors="coerce", utc=True)
    df["date_dt"] = pd.to_datetime(df["date_dt"], errors="coerce", utc=True)
    df["harvest_success_b"] = df["harvest_success_b"].astype(bool)
    return _filter_active(df)


def get_recent_transformations(days: int = 7) -> pd.DataFrame:
    since = _since(days)
    docs = _query(
        ["type_s:transformation", f"transformation_started_dt:[{since} TO *]"],
        fl="dataset_s,date_dt,grid_name_s,field_s,success_b,error_message_s,transformation_started_dt",
    )
    if not docs:
        return pd.DataFrame(columns=["dataset_s", "date_dt", "grid_name_s",
                                      "field_s", "success_b", "error_message_s",
                                      "transformation_started_dt"])
    df = pd.DataFrame(docs)
    df = _ensure_columns(df, {
        "dataset_s": "", "grid_name_s": "", "field_s": "",
        "error_message_s": "", "success_b": True,
        "transformation_started_dt": None,
    })
    df["transformation_started_dt"] = pd.to_datetime(
        df["transformation_started_dt"], errors="coerce", utc=True
    )
    df["success_b"] = df["success_b"].astype(bool)
    return _filter_active(df)


def get_recent_aggregations(days: int = 7) -> pd.DataFrame:
    since = _since(days)
    docs = _query(
        ["type_s:aggregation", f"aggregation_time_dt:[{since} TO *]"],
        fl="dataset_s,year_i,grid_name_s,field_s,aggregation_success_b,error_message_s,aggregation_time_dt",
    )
    if not docs:
        return pd.DataFrame(columns=["dataset_s", "year_i", "grid_name_s",
                                      "field_s", "aggregation_success_b",
                                      "error_message_s", "aggregation_time_dt"])
    df = pd.DataFrame(docs)
    df = _ensure_columns(df, {
        "dataset_s": "", "year_i": None, "grid_name_s": "", "field_s": "",
        "error_message_s": "", "aggregation_success_b": True,
        "aggregation_time_dt": None,
    })
    df["aggregation_time_dt"] = pd.to_datetime(
        df["aggregation_time_dt"], errors="coerce", utc=True
    )
    df["aggregation_success_b"] = df["aggregation_success_b"].astype(bool)
    return _filter_active(df)


# ---------------------------------------------------------------------------
# Per-dataset detail
# ---------------------------------------------------------------------------

def _ensure_columns(df: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """Add any missing columns with a default value so views can assume they exist."""
    for col, default in columns.items():
        if col not in df.columns:
            df[col] = default
    return df


def get_granules(ds_name: str) -> pd.DataFrame:
    docs = _query(
        ["type_s:granule", f"dataset_s:{ds_name}"],
        fl="date_dt,filename_s,harvest_success_b,error_message_s,download_duration_i,last_update_dt",
    )
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    df = _ensure_columns(df, {
        "filename_s": "",
        "error_message_s": "",
        "download_duration_i": None,
        "last_update_dt": None,
        "harvest_success_b": True,
    })
    df["date_dt"] = pd.to_datetime(df["date_dt"], errors="coerce", utc=True)
    df["last_update_dt"] = pd.to_datetime(df["last_update_dt"], errors="coerce", utc=True)
    df["harvest_success_b"] = df["harvest_success_b"].astype(bool)
    return df.sort_values("date_dt")


def get_transformations(ds_name: str) -> pd.DataFrame:
    docs = _query(
        ["type_s:transformation", f"dataset_s:{ds_name}"],
        fl="date_dt,grid_name_s,field_s,success_b,error_message_s,transformation_completed_dt",
    )
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    df = _ensure_columns(df, {
        "grid_name_s": "",
        "field_s": "",
        "error_message_s": "",
        "transformation_completed_dt": None,
        "success_b": True,
    })
    df["date_dt"] = pd.to_datetime(df["date_dt"], errors="coerce", utc=True)
    df["success_b"] = df["success_b"].astype(bool)
    return df.sort_values("date_dt")


def get_aggregations(ds_name: str) -> pd.DataFrame:
    docs = _query(
        ["type_s:aggregation", f"dataset_s:{ds_name}"],
        fl="year_i,grid_name_s,field_s,aggregation_success_b,error_message_s,aggregation_time_dt",
    )
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    df = _ensure_columns(df, {
        "year_i": None,
        "grid_name_s": "",
        "field_s": "",
        "error_message_s": "",
        "aggregation_time_dt": None,
        "aggregation_success_b": True,
    })
    df["aggregation_success_b"] = df["aggregation_success_b"].astype(bool)
    df["year_i"] = pd.to_numeric(df.get("year_i"), errors="coerce")
    return df.sort_values("year_i")
