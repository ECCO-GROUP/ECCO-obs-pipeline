"""
Single source of truth for target grid identity: name, file path and type.

A grid's identity is its ``name`` global attribute, not its filename — the two are
allowed to differ (``grids/polar_stereo_n_25km.nc`` is named
``polar_stereo_n_25km_demo``). Before this module, the transformation path rebuilt
paths from the name as ``grids/{grid_name}.nc``, which raises FileNotFoundError deep
inside a worker for any grid where they diverge, while the aggregation path resolved
them correctly from Solr's ``grid_path_s``. Both now go through ``load_registry``.

Scanning is cheap (~9ms for 5 grid files) and derives purely from disk, so the
registry behaves identically under the "fork" and "spawn" multiprocessing start
methods — a worker that re-imports this module rebuilds the same mapping rather than
depending on inherited parent state.
"""

import logging
import os
from dataclasses import dataclass
from glob import glob
from typing import Dict, Optional

import xarray as xr

logger = logging.getLogger("pipeline")

# Permitted values of a grid's ``type`` global attribute (see grids/README.md).
VALID_GRID_TYPES = frozenset({"llc", "latlon"})

GRIDS_DIR = "grids"


class GridRegistryError(Exception):
    """
    A grid file on disk does not meet the metadata standard in grids/README.md, or two
    grid files claim the same ``name``.

    Raised rather than logged because every one of these makes a grid unusable or
    ambiguous. A duplicate name is the dangerous case: Solr grid docs are keyed by
    name, and update_solr_grid deletes every transformation and aggregation for a grid
    whose checksum changed, so two files sharing a name would make alternating runs
    wipe each other's work. init_pipeline converts this into a
    PipelinePreconditionError so the run aborts with a recorded summary.
    """


@dataclass(frozen=True)
class GridFile:
    """A target grid file on disk, identified by its ``name`` global attribute."""

    name: str
    path: str
    type: str


# Per-process cache. Module-level, so each forked or spawned worker gets its own copy;
# there is no cross-process sharing to synchronise. Keyed by grids directory so a test
# pointing at a fixture directory does not poison the real entry.
_registry_cache: Dict[str, Dict[str, GridFile]] = {}


def load_registry(
    grids_dir: str = GRIDS_DIR, use_cache: bool = True
) -> Dict[str, GridFile]:
    """
    Scan ``grids_dir`` for grid files and return ``{name: GridFile}``.

    Raises GridRegistryError if any grid file is missing its ``name`` or ``type``
    global attribute, declares a ``type`` outside VALID_GRID_TYPES, or shares a
    ``name`` with another file.

    Paths are relative to the current working directory, matching the existing
    ``glob("grids/*.nc")`` convention; conf/global_settings.py chdirs to ROOT_DIR on
    import, so this resolves the same way in workers as in the parent.
    """
    if use_cache and grids_dir in _registry_cache:
        return _registry_cache[grids_dir]

    registry: Dict[str, GridFile] = {}
    problems = []

    for grid_file_path in sorted(glob(os.path.join(grids_dir, "*.nc"))):
        try:
            with xr.open_dataset(grid_file_path) as ds:
                attrs = dict(ds.attrs)
        except Exception as e:
            problems.append(f"{grid_file_path}: could not be opened ({e})")
            continue

        # Checked explicitly rather than via ds.attrs[...]: grids_to_solr used bare
        # subscripting, so a grid missing either attribute raised KeyError inside a
        # broad try/except that logged and continued — leaving the run to proceed with
        # that grid silently absent.
        missing = [key for key in ("name", "type") if not attrs.get(key)]
        if missing:
            problems.append(
                f"{grid_file_path}: missing required global attribute(s) "
                f"{', '.join(missing)} (see grids/README.md)"
            )
            continue

        grid_name = attrs["name"]
        grid_type = attrs["type"]

        if grid_type not in VALID_GRID_TYPES:
            problems.append(
                f'{grid_file_path}: type "{grid_type}" is not one of '
                f"{', '.join(sorted(VALID_GRID_TYPES))}"
            )
            continue

        if grid_name in registry:
            problems.append(
                f'duplicate grid name "{grid_name}" claimed by both '
                f"{registry[grid_name].path} and {grid_file_path}"
            )
            continue

        registry[grid_name] = GridFile(
            name=grid_name, path=grid_file_path, type=grid_type
        )
        logger.debug(f"Registered grid {grid_name} ({grid_type}) at {grid_file_path}")

    if problems:
        raise GridRegistryError(
            f"Invalid grid files in {grids_dir}/:\n  " + "\n  ".join(problems)
        )

    if use_cache:
        _registry_cache[grids_dir] = registry
    return registry


def grid_path(grid_name: str, grids_dir: str = GRIDS_DIR) -> str:
    """
    Resolve a grid name to the path of its file on disk.

    Raises GridRegistryError naming the known grids if ``grid_name`` is not registered,
    which is a clearer failure than the FileNotFoundError the old
    ``f"grids/{grid_name}.nc"`` produced.
    """
    registry = load_registry(grids_dir)
    grid = registry.get(grid_name)
    if grid is None:
        known = ", ".join(sorted(registry)) or "none"
        raise GridRegistryError(
            f'No grid file found for "{grid_name}" in {grids_dir}/. Known grids: {known}'
        )
    return grid.path


def clear_cache(grids_dir: Optional[str] = None):
    """
    Drop cached registries. For tests that write grid fixtures between loads.
    """
    if grids_dir is None:
        _registry_cache.clear()
    else:
        _registry_cache.pop(grids_dir, None)
