from dataclasses import dataclass
from datetime import datetime
from typing import Iterable


class Config:
    """
    Global runtime configuration shared across the pipeline.
    """

    user_cpus = 1
    grids_to_use = []


def set_grids(grids: Iterable[str]):
    Config.grids_to_use = grids


def set_cpus(cpus: int):
    Config.user_cpus = cpus


@dataclass
class Field:
    name: str
    long_name: str
    standard_name: str
    units: str
    pre_transformations: Iterable[str]
    post_transformations: Iterable[str]


class Dataset:
    """
    Represents a single dataset configuration entry.
    """

    def __init__(self, config: dict) -> None:
        self.ds_name: str = config.get("ds_name")
        self.start: datetime = datetime.strptime(config.get("start"), "%Y%m%dT%H:%M:%SZ")
        self.end: datetime = (
            datetime.now() if config.get("end") == "NOW" else datetime.strptime(config["end"], "%Y%m%dT%H:%M:%SZ")
        )
        self.harvester_type: str = config.get("harvester_type")
        self.filename_date_fmt: str = config.get("filename_date_fmt")
        self.filename_date_regex: str = config.get("filename_date_regex")
        self.data_time_scale: str = config.get("data_time_scale")
        self.hemi_pattern: dict = config.get("hemi_pattern")
        self.fields: Iterable[Field] = [Field(**fld) for fld in config.get("fields", [])]
        self.og_ds_metadata: dict[str, any] = {k: v for k, v in config.items() if "original" in k}
        self.preprocessing_function: str = config.get("preprocessing")
        self.t_version = config.get("t_version")
        self.a_version = config.get("a_version")
        self.note: str = config.get("notes")
