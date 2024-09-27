import os
import xarray as xr
import numpy as np
from typing import Iterable
from dateutil.relativedelta import relativedelta
from datetime import datetime
import netCDF4 as nc4
from utils.processing_utils.llc_array_conversion import llc_tiles_to_compact

DTYPE = np.float32
BINARY_DTYPE = ">f4"
BINARY_FILL_VALUE = -9999
NETCDF_FILL_VALUE = nc4.default_fillvals[BINARY_DTYPE.replace(">", "")]


def make_empty_record(record_date: str, model_grid: xr.Dataset) -> xr.DataArray:
    """
    Creates xarray DataArray filled with nans.
    """
    # model_grid must contain the corrdinates XC and YC

    # make an empty data array to hold the interpolated 2D field
    # all values are nans.
    # dimensions are the same as model_grid.XC
    nan_array = np.full(model_grid.XC.values.shape, np.nan, DTYPE)
    data_DA = xr.DataArray(nan_array, dims=model_grid.XC.dims)

    data_DA = data_DA.assign_coords(time=np.datetime64(record_date, "ns"))
    data_DA = data_DA.expand_dims(dim="time", axis=0)

    # add start and end time records. default is same value as record date
    data_DA = data_DA.assign_coords(
        {
            "time_start": ("time", data_DA.time.data.copy()),
            "time_end": ("time", data_DA.time.data.copy()),
        }
    )

    for dim in model_grid.XC.dims:
        data_DA = data_DA.assign_coords({dim: model_grid[dim]})

    try:
        data_DA = data_DA.assign_coords(
            {
                "XC": (model_grid.XC.dims, model_grid.XC.data),
                "YC": (model_grid.YC.dims, model_grid.YC.data),
            }
        )
    except Exception:
        print("Unsupported model grid format")
        return []

    data_DA.XC.attrs["coverage_content_type"] = "coordinate"
    data_DA.YC.attrs["coverage_content_type"] = "coordinate"

    # copy over the attributes from XC and YC to the dataArray
    data_DA.XC.attrs = model_grid.XC.attrs
    data_DA.YC.attrs = model_grid.YC.attrs

    data_DA.name = "Default empty model grid record"

    return data_DA


class TimeBound:
    """
    Class for computing time bounds and center time for a given date and coverage period.

    Supports both looking forward (ie: bounds computed from a start date) and looking backward (ie: bounds computed from an end date)
    """

    freq_mapping = {
        "AVG_MON": relativedelta(months=1),
        "AVG_DAY": relativedelta(days=1),
        "AVG_WEEK": relativedelta(weeks=1),
        "AVG_YEAR": relativedelta(years=1),
    }

    def __init__(
        self,
        rec_avg_start: np.datetime64 | None = None,
        rec_avg_end: np.datetime64 | None = None,
        period: str = "AVG_DAY",
    ):
        if all([rec_avg_start, rec_avg_end]) or None not in [
            rec_avg_end,
            rec_avg_start,
        ]:
            raise ValueError(
                "One of rec_avg_start or rec_avg_end must be provided, but not both."
            )

        if period not in ["AVG_MON", "AVG_DAY", "AVG_WEEK", "AVG_YEAR"]:
            raise ValueError(
                f"{period} is invalid output_freq_code. Must be one of AVG_MON, AVG_DAY, AVG_WEEK, OR AVG_YEAR"
            )

        if rec_avg_end:
            time_dt: datetime = rec_avg_end.astype("datetime64[s]").astype(object)
            rec_avg_start = time_dt - self.freq_mapping[period]
            rec_avg_start = np.datetime64(rec_avg_start).astype("datetime64[ns]")
        elif rec_avg_start:
            time_dt: datetime = rec_avg_start.astype("datetime64[s]").astype(object)
            rec_avg_end = time_dt + self.freq_mapping[period]
            rec_avg_end = np.datetime64(rec_avg_end).astype("datetime64[ns]")

        rec_avg_delta = rec_avg_end - rec_avg_start
        rec_avg_middle = rec_avg_start + rec_avg_delta / 2

        self._start: np.datetime64 = rec_avg_start
        self.center: np.datetime64 = rec_avg_middle
        self._end: np.datetime64 = rec_avg_end
        self.bounds: Iterable[np.datetime64] = np.array([rec_avg_start, rec_avg_end])


def save_binary(data, output_filename, binary_output_dir, model_grid_type, data_var=""):
    if data_var:
        data_values = data[data_var].values
    else:
        data_values = data.values

    # define binary file output filetype
    dt_out = np.dtype(BINARY_DTYPE)

    # create directory
    os.makedirs(binary_output_dir, exist_ok=True)

    # define binary output filename
    binary_output_filename = os.path.join(binary_output_dir, output_filename)

    # replace nans with the binary fill value (something like -9999)
    tmp_fields = np.where(np.isnan(data_values), BINARY_FILL_VALUE, data_values)

    # SAVE FLAT BINARY
    # loop through each record of the year, save binary fields one at a time
    # appending each record as we go
    fd1 = open(str(binary_output_filename), "wb")
    fd1 = open(str(binary_output_filename), "ab")

    for i in range(len(data.time)):
        # print('saving binary record: ', str(i))

        # if we have an llc grid, then we have to reform to compact
        if model_grid_type == "llc":
            tmp_field = llc_tiles_to_compact(tmp_fields[i, :], less_output=True)

        # otherwise assume grid is x,y (2 dimensions)
        elif model_grid_type == "latlon":
            tmp_field = tmp_fields[i, :]

        else:
            print("unknown model grid type!")
            tmp_field = []
            return []

        # make sure we have something to save...
        if len(tmp_field) > 0:
            # if this is the first record, create new binary file
            tmp_field.astype(dt_out).tofile(fd1)

    # close the file at the end of the operation
    fd1.close()


def save_netcdf(data: xr.Dataset, output_filename: str, netcdf_output_dir: str):
    os.makedirs(netcdf_output_dir, exist_ok=True)
    nc_output_path = os.path.join(netcdf_output_dir, output_filename)

    try:
        data = data.fillna(NETCDF_FILL_VALUE)
        data_DS = data.to_dataset()
    except Exception:
        data_DS = data

    coord_encoding = {}
    for coord in data_DS.coords:
        coord_encoding[coord] = {"_FillValue": None, "dtype": "float32"}

        if coord == "time" or coord == "time_bnds":
            coord_encoding[coord] = {"dtype": "int32"}
    coord_encoding["time"] = {"units": "hours since 1980-01-01"}

    var_encoding = {}
    for var in data_DS.data_vars:
        var_encoding[var] = {
            "zlib": True,
            "complevel": 5,
            "shuffle": True,
            "_FillValue": NETCDF_FILL_VALUE,
        }

    encoding = {**coord_encoding, **var_encoding}
    data_DS.to_netcdf(nc_output_path, encoding=encoding)
    data_DS.close()
