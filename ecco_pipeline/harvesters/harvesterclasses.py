import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests

from baseclasses import Dataset
from conf.global_settings import OUTPUT_DIR
from utils.pipeline_utils import file_utils, solr_utils

logger = logging.getLogger("pipeline")

CHUNK_SIZE = 1024 * 1024  # 1 MB

# Number of concurrent download workers. Overridable per remote box via the
# HARVEST_MAX_WORKERS env var without a code change. PO.DAAC/EDL-gated archives
# generally tolerate well more than the old default of 3.
DEFAULT_DOWNLOAD_WORKERS = 8

# Buffered granule docs are flushed to Solr every SOLR_FLUSH_INTERVAL granules
# during a harvest (instead of a single write at the very end), so large
# harvests are durable and visible mid-run.
SOLR_FLUSH_INTERVAL = 500

# One requests.Session per worker thread, holding the Earthdata Login cookie and
# a keep-alive connection so each granule download skips the TCP/TLS handshake
# and the full EDL OAuth redirect chain after the first. netrc creds are picked
# up automatically (trust_env is on by default).
_thread_local = threading.local()


class Granule:
    def __init__(
        self,
        ds_name: str,
        local_fp: str,
        date: datetime,
        modified_time: datetime,
        url: str,
    ):
        self.ds_name = ds_name
        self.local_fp = local_fp
        self.filename = local_fp.split("/")[-1]
        self.datetime = date
        self.modified_time = modified_time
        self.url = url
        self.gen_granule_doc()

    def gen_granule_doc(self):
        item = {}
        item["type_s"] = "granule"
        item["date_dt"] = datetime.strftime(self.datetime, "%Y-%m-%dT00:00:00Z")
        item["dataset_s"] = self.ds_name
        item["filename_s"] = self.filename
        item["source_s"] = self.url
        item["modified_time_dt"] = self.modified_time.strftime("%Y-%m-%dT00:00:00Z")
        item["error_message_s"] = ""
        self.solr_item = item

    def update_item(self, solr_docs, success, error_message="", download_duration=0):
        if self.filename in solr_docs.keys():
            self.solr_item["id"] = solr_docs[self.filename]["id"]

        # last_attempt_dt records every harvest attempt (success or failure) so the
        # dashboard can surface first-time failures that have no download_time_dt.
        self.solr_item["last_attempt_dt"] = datetime.utcnow().strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        if success:
            # calculate checksum and expected file size
            self.solr_item["checksum_s"] = file_utils.md5(self.local_fp)
            self.solr_item["pre_transformation_file_path_s"] = self.local_fp
            self.solr_item["harvest_success_b"] = True
            self.solr_item["file_size_l"] = os.path.getsize(self.local_fp)
            self.solr_item["error_message_s"] = ""
            self.solr_item["download_time_dt"] = datetime.fromtimestamp(
                os.path.getmtime(self.local_fp)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            self.solr_item["harvest_success_b"] = False
            self.solr_item["pre_transformation_file_path_s"] = ""
            self.solr_item["file_size_l"] = 0
            self.solr_item["error_message_s"] = error_message
            existing = solr_docs.get(self.filename, {})
            if "download_time_dt" in existing:
                self.solr_item["download_time_dt"] = existing["download_time_dt"]
        self.solr_item["download_duration_i"] = download_duration

    def get_solr_docs(self):
        return [self.solr_item]


class Harvester(Dataset):
    solr_format: str = "%Y-%m-%dT%H:%M:%SZ"

    def __init__(self, config: dict):
        super().__init__(config)
        self._harvester_parsing(config)
        self.target_dir: str = os.path.join(
            OUTPUT_DIR, self.ds_name, "harvested_granules"
        )
        self.updated_solr_docs: list = []
        # Granule docs written to Solr but not yet flushed. Guarded by
        # _flush_lock, which also serializes worker appends against a flush.
        self._pending_solr_docs: list = []
        self._flush_lock = threading.Lock()
        self.max_workers: int = int(
            os.environ.get("HARVEST_MAX_WORKERS", DEFAULT_DOWNLOAD_WORKERS)
        )

        self.ensure_target_dir()
        solr_utils.clean_solr(config)

        self.solr_docs = self.get_solr_docs()
        self.config: dict = config

    def _harvester_parsing(self, config: dict):
        if self.harvester_type == "cmr":
            self.cmr_concept_id = config.get("cmr_concept_id")
            self.provider = config.get("provider")
        else:
            self.ddir = config.get("ddir")

    def fetch(self):
        raise NotImplementedError

    def get_mod_time(self):
        raise NotImplementedError

    def dl_file(self):
        raise NotImplementedError

    def _get_download_session(self) -> requests.Session:
        """
        Return this worker thread's persistent download Session.

        A per-thread Session keeps the EDL auth cookie and a pooled keep-alive
        connection, so downloads after the first skip the ~1-2s TCP/TLS +
        EDL OAuth redirect tax that a fresh requests.get() pays every call.
        """
        session = getattr(_thread_local, "download_session", None)
        if session is None:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=self.max_workers,
                pool_maxsize=self.max_workers,
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            _thread_local.download_session = session
        return session

    def _stream_download(self, src: str, dst: str):
        """
        Stream a file to disk and verify the transfer completed in full.

        The server's Content-Length is the source of truth for expected size:
        if the bytes written don't match it, the download was truncated and we
        raise so the caller records a harvest failure. When the server sends no
        Content-Length (e.g. chunked encoding) we can't know the expected size,
        so we fall back to rejecting only a genuinely empty file.
        """
        session = self._get_download_session()
        with session.get(src, stream=True, timeout=120) as r:
            r.raise_for_status()
            declared = r.headers.get("Content-Length")
            expected_size = int(declared) if declared is not None else None
            bytes_written = 0
            with open(dst, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    bytes_written += len(chunk)

        if expected_size is not None:
            if bytes_written != expected_size:
                raise IOError(
                    f"Truncated download: wrote {bytes_written} of "
                    f"{expected_size} bytes for {src}"
                )
        elif bytes_written == 0:
            raise IOError(f"Empty download: server returned no data for {src}")

    def ensure_target_dir(self):
        os.makedirs(self.target_dir, exist_ok=True)

    def get_solr_docs(self) -> dict:
        docs = {}

        # Query for existing harvested docs — only fetch fields needed for
        # check_update (harvest_success_b, download_time_dt) and upserts (id)
        fq = ["type_s:granule", f"dataset_s:{self.ds_name}"]
        harvested_docs = solr_utils.solr_query(
            fq, fl="id,filename_s,harvest_success_b,download_time_dt"
        )
        for doc in harvested_docs:
            docs[doc["filename_s"]] = doc

        return docs

    def make_ds_doc(self, source: str, chk_time: str):
        ds_meta = {}
        ds_meta["type_s"] = "dataset"
        ds_meta["dataset_s"] = self.ds_name
        ds_meta["short_name_s"] = self.og_ds_metadata["original_dataset_short_name"]
        ds_meta["source_s"] = source
        ds_meta["data_time_scale_s"] = self.data_time_scale
        ds_meta["last_checked_dt"] = chk_time
        ds_meta["original_dataset_title_s"] = self.og_ds_metadata[
            "original_dataset_title"
        ]
        ds_meta["original_dataset_short_name_s"] = self.og_ds_metadata[
            "original_dataset_short_name"
        ]
        ds_meta["original_dataset_url_s"] = self.og_ds_metadata["original_dataset_url"]
        ds_meta["original_dataset_reference_s"] = self.og_ds_metadata[
            "original_dataset_reference"
        ]
        ds_meta["original_dataset_doi_s"] = self.og_ds_metadata["original_dataset_doi"]
        ds_meta["harvester_type_s"] = self.harvester_type
        ds_meta["ecco_variable_s"] = self.ecco_variable
        ds_meta["t_version_f"] = self.t_version
        return ds_meta

    def check_update(self, filename, mod_time):
        return (
            (filename not in self.solr_docs.keys())
            or (not self.solr_docs[filename]["harvest_success_b"])
            or (self.solr_docs[filename]["download_time_dt"] < str(mod_time))
        )

    def need_to_download(self, granule: Granule) -> bool:
        if not os.path.exists(granule.local_fp):
            return True
        # If file exists locally, but is out of date, download it
        elif (
            datetime.fromtimestamp(os.path.getmtime(granule.local_fp))
            <= granule.modified_time
        ):
            return True
        return False

    def flush_solr_docs(self, force: bool = False):
        """
        Write buffered granule docs to Solr using commitWithin batching.

        Called periodically during a harvest so large runs are durable and
        visible mid-run instead of a single write at the end. Only flushes once
        SOLR_FLUSH_INTERVAL docs have accumulated unless force=True, which
        flushes whatever remains (used at the end of the drain loop).

        The pending list is copied and cleared under _flush_lock, then the
        network write happens outside the lock so workers aren't blocked on it.
        """
        with self._flush_lock:
            if not self._pending_solr_docs:
                return
            if not force and len(self._pending_solr_docs) < SOLR_FLUSH_INTERVAL:
                return
            batch = self._pending_solr_docs
            self._pending_solr_docs = []
            # Keep the complete set for post_fetch (last_download_dt, date ranges).
            self.updated_solr_docs.extend(batch)

        # commit=False -> commitWithin, so batched flushes don't trigger a
        # commit/searcher-warming storm on every call. post_fetch forces a hard
        # commit at the run boundary.
        solr_utils.solr_update(batch, commit=False)

    def drain_futures(self, process_granule, to_process, max_workers=None):
        """
        Run process_granule(*args) for each tuple in to_process across a thread
        pool, buffering returned granule docs and flushing to Solr every
        SOLR_FLUSH_INTERVAL granules. Flushes the remainder before returning.

        Centralizes the accumulate + batched-flush pattern so each fetch_*
        method only builds to_process and defines its per-granule worker.
        """
        if max_workers is None:
            max_workers = self.max_workers

        total = len(to_process)
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_granule, *args) for args in to_process]
            for future in as_completed(futures):
                docs = future.result()
                completed += 1
                if completed % SOLR_FLUSH_INTERVAL == 0:
                    logger.info(
                        f"{self.ds_name}: processed {completed}/{total} granules"
                    )
                if not docs:
                    continue
                with self._flush_lock:
                    self._pending_solr_docs.extend(docs)
                self.flush_solr_docs()

        self.flush_solr_docs(force=True)
        logger.info(f"Downloading {self.ds_name} complete")

    def post_fetch(self, source: str) -> str:
        check_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        if self.updated_solr_docs:
            # Batched flushes used commitWithin; force a hard commit so the
            # count/date read-backs below see the latest state.
            solr_utils.commit_solr()
            logger.debug("Committed batched Solr harvested documents")
        else:
            logger.debug("No downloads required.")

        harvesting_status = self.harvester_status()

        # Query for Solr dataset level document
        fq = ["type_s:dataset", f"dataset_s:{self.ds_name}"]
        ds_doc = solr_utils.solr_query(fq)

        if ds_doc:
            # -----------------------------------------------------
            # Update Solr dataset entry
            # -----------------------------------------------------
            dataset_metadata = ds_doc[0]

            # Query for dates of all harvested docs
            fq = [
                f"dataset_s:{self.ds_name}",
                "type_s:granule",
                "harvest_success_b:true",
            ]
            dates_query = solr_utils.solr_query(fq, fl="date_dt")
            dates = [x["date_dt"] for x in dates_query]

            # Granule counts for dashboard
            fq_base = ["type_s:granule", f"dataset_s:{self.ds_name}"]
            n_failed = solr_utils.solr_count(fq_base + ["harvest_success_b:false"])
            n_success = solr_utils.solr_count(fq_base + ["harvest_success_b:true"])

            # Build update document body
            ds_meta = {}
            ds_meta["id"] = dataset_metadata["id"]
            ds_meta["last_checked_dt"] = {"set": check_time}
            ds_meta["ecco_variable_s"] = {"set": self.ecco_variable}
            ds_meta["n_granules_i"] = {"set": n_failed + n_success}
            ds_meta["n_granules_success_i"] = {"set": n_success}
            ds_meta["n_granules_failed_i"] = {"set": n_failed}
            if dates:
                ds_meta["start_date_dt"] = {"set": min(dates)}
                ds_meta["end_date_dt"] = {"set": max(dates)}

            if self.updated_solr_docs:
                ds_meta["harvest_status_s"] = {"set": harvesting_status}
                dl_solr_docs = [
                    doc
                    for doc in self.updated_solr_docs
                    if "download_time_dt" in doc.keys()
                ]
                last_dl_item = sorted(
                    dl_solr_docs, key=lambda d: d["download_time_dt"]
                )[-1]
                ds_meta["last_download_dt"] = {"set": last_dl_item["download_time_dt"]}
        else:
            # -----------------------------------------------------
            # Create Solr Dataset-level Document if doesn't exist
            # -----------------------------------------------------
            ds_meta = self.make_ds_doc(source, check_time)

            # Only include start_date and end_date if there was at least one successful download
            if self.updated_solr_docs:
                ds_meta["harvest_status_s"] = {"set": harvesting_status}
                dl_solr_docs = [
                    doc
                    for doc in self.updated_solr_docs
                    if "download_time_dt" in doc.keys()
                ]
                dl_items = sorted(dl_solr_docs, key=lambda d: d["date_dt"])
                ds_meta["start_date_dt"] = dl_items[0]["date_dt"]
                ds_meta["end_date_dt"] = dl_items[-1]["date_dt"]
                ds_meta["last_download_dt"] = sorted(
                    dl_solr_docs, key=lambda d: d["download_time_dt"]
                )[-1]["download_time_dt"]

            ds_meta["harvest_status_s"] = harvesting_status

        # Update Solr with modified dataset entry
        r = solr_utils.solr_update([ds_meta], r=True)

        if r.status_code == 200:
            logger.debug("Successfully updated Solr dataset document")
        else:
            logger.exception("Failed to update Solr dataset document")
        return harvesting_status

    def harvester_status(self) -> str:
        fq_base = ["type_s:granule", f"dataset_s:{self.ds_name}"]
        failed_count = solr_utils.solr_count(fq_base + ["harvest_success_b:false"])
        successful_count = solr_utils.solr_count(fq_base + ["harvest_success_b:true"])

        if not successful_count:
            return (
                "No usable granules harvested (either all failed or no data collected)"
            )
        elif failed_count:
            return f"{failed_count} harvested granules failed"
        return "All granules successfully harvested"

    def ds_doc_update(self) -> bool:
        # Query for Solr dataset level document
        fq = ["type_s:dataset", f"dataset_s:{self.ds_name}"]
        dataset_query = solr_utils.solr_query(fq)

        # If dataset entry exists on Solr
        return len(dataset_query) == 1
