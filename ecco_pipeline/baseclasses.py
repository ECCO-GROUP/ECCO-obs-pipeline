from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
import numpy as np
import xarray as xr


class Config:
    user_cpus = 1
    grids_to_use = []


def set_grids(grids: Iterable[str]):
    Config.grids_to_use = grids


def set_cpus(cpus: int):
    Config.user_cpus = cpus


class Dataset:
    """
    Base class for working with a dataset's config values.
    """

    def __init__(self, config: dict) -> None:
        self.ds_name: str = config.get("ds_name")
        self.start: datetime = datetime.strptime(config.get("start"), "%Y%m%dT%H:%M:%SZ")
        self.end: datetime = (
            datetime.now() if config.get("end") == "NOW" else datetime.strptime(config.get("end"), "%Y%m%dT%H:%M:%SZ")
        )
        self.harvester_type: str = config.get("harvester_type")
        self.filename_date_fmt: str = config.get("filename_date_fmt")
        self.filename_date_regex: str = config.get("filename_date_regex")
        self.data_time_scale: str = config.get("data_time_scale")
        self.hemi_pattern: dict = config.get("hemi_pattern")
        self.fields: Iterable[Field] = [Field(**config_field) for config_field in config.get("fields")]
        self.og_ds_metadata: dict = {k: v for k, v in config.items() if "original" in k}
        self.preprocessing_function: str = config.get("preprocessing")
        self.t_version = config.get("t_version")
        self.a_version = config.get("a_version")
        self.note: str = config.get("notes")


@dataclass
class Field:
    name: str
    long_name: str
    standard_name: str
    units: str
    pre_transformations: Iterable[str]
    post_transformations: Iterable[str]


class Grid:
    def __init__(self, ds: xr.Dataset):
        self.ds = ds
        self.name = ds.name
        self.type = ds.type

    @property
    def radius(self) -> np.ndarray:
        for key in ("effective_grid_radius", "effective_radius", "RAD"):
            if key in self.ds:
                return self.ds[key].values.ravel()

        if "rA" in self.ds:
            return 0.5 * np.sqrt(self.ds["rA"].values.ravel())

        raise KeyError("No valid grid radius variable found in dataset")
