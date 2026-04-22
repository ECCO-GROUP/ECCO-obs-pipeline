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

_HEADERS = {"Connection": "close"}
_TIMEOUT = 60
_RETRY_DELAY = 5


def _request(method: str, url: str, **kwargs):
    """Single retry wrapper for all Solr requests."""
    kwargs.setdefault("headers", _HEADERS)
    kwargs.setdefault("timeout", _TIMEOUT)
    try:
        return requests.request(method, url, **kwargs)
    except Exception as e:
        logger.warning(f"Solr request failed ({e}), retrying in {_RETRY_DELAY}s...")
        time.sleep(_RETRY_DELAY)
        return requests.request(method, url, **kwargs)


def solr_query(fq: Iterable[str], fl: str = "") -> Iterable[dict]:
    """
    Submit query to Solr
    """
    query_params = {"q": "*:*", "fq": fq, "fl": fl, "rows": 300000}
    url = f"{SOLR_HOST}{SOLR_COLLECTION}/select?"
    response = _request("GET", url, params=query_params)
    return response.json()["response"]["docs"]


def solr_count(fq: Iterable[str]) -> int:
    """
    Return the number of documents matching fq without fetching any docs.
    """
    query_params = {"q": "*:*", "fq": fq, "rows": 0}
    url = f"{SOLR_HOST}{SOLR_COLLECTION}/select?"
    response = _request("GET", url, params=query_params)
    return response.json()["response"]["numFound"]


def solr_update(update_body: Iterable[dict], r: bool = False):
    """
    Submit update to Solr
    """
    url = f"{SOLR_HOST}{SOLR_COLLECTION}/update?commit=true"
    response = _request("POST", url, json=update_body)
    if r:
        return response


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
    if not solr_query(["type_s:grid"]):
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
