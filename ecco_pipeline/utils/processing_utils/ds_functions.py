import logging
from dataclasses import Field
from multiprocessing import current_process
from typing import Iterable

import numpy as np
import xarray as xr

logger = logging.getLogger(str(current_process().pid))


class FuncNotFound(Exception):
    """Raise for processing func not found"""


class PreprocessingFuncs:
    """
    Preprocessing functions, used for handling opening irregular files
    """

    def call_function(self, function_name: str, file_path: str, fields: Iterable[Field]):
        # Get the function dynamically by name
        func = getattr(self, function_name, None)
        if func:
            # Call the function if found
            logger.info(f"Applying preprocessing function {function_name} to data.")
            return func(file_path, fields)
        else:
            raise FuncNotFound(f"Function '{function_name}' not found")

    def ATL20_V004_monthly(self, file_path: str, fields: Iterable[Field]) -> xr.Dataset:
        """
        Handle data groups
        """
        vars = [field.name for field in fields]

        ds = xr.open_dataset(file_path, decode_times=True)
        ds = ds[["grid_x", "grid_y", "crs"]]

        var_ds = xr.open_dataset(file_path, group="monthly")[vars]
        merged_ds = xr.merge([ds, var_ds])
        return merged_ds


class PretransformationFuncs:
    """
    Pre transformation functions, to be performed on xr.Datasets
    """

    def call_functions(self, function_names: Iterable[str], ds: xr.Dataset) -> xr.Dataset:
        for func_name in function_names:
            ds = self.call_function(func_name, ds)
        return ds

    def call_function(self, function_name: str, ds: xr.Dataset) -> xr.Dataset:
        # Get the function dynamically by name
        func = getattr(self, function_name, None)
        if func:
            try:
                ds = func(ds)
            except Exception as e:
                logger.exception(f"Error running {function_name}. {e}")
        else:
            raise FuncNotFound(f"Function '{function_name}' not found")
        return ds

    def AVHRR_remove_ice_or_near_ice(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Replaces SST values < -0.5 or sea_ice_fraction > 0 to NaN
        """
        # nonzero sea ice fraction
        if "sea_ice_fraction" in ds:
            ds.analysed_sst.values = np.where(ds.sea_ice_fraction > 0, np.nan, ds.analysed_sst.values)

        # colder than -0.5C
        ds.analysed_sst.values = np.where(ds.analysed_sst <= 273.15 - 0.5, np.nan, ds.analysed_sst.values)

        return ds

    def RDEFT4_remove_negative_values(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Replaces negative values with nans for all data vars
        """
        for field in ds.data_vars:
            if field in ["lat", "lon"]:
                continue
            ds[field].values = np.where(ds[field].values < 0, np.nan, ds[field].values)
        return ds

    def G02202v4_mask_flagged_conc(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Masks out values greater than 1 in nsidc_nt_seaice_conc and cdr_seaice_conc
        """
        logger.debug(f'G2202 masking flagged nt pre   : {np.sum(ds["nsidc_nt_seaice_conc"].values.ravel() > 1)}')
        tmpNT = np.where(ds["nsidc_nt_seaice_conc"].values.ravel() > 1, 1, 0)
        tmpCDR = np.where(ds["cdr_seaice_conc"].values.ravel() > 1, 1, 0)
        logger.debug(f"G2202 masking flagged NDR, CDR pre: {np.sum(tmpNT), np.sum(tmpCDR)}")

        ds["nsidc_nt_seaice_conc"] = ds["nsidc_nt_seaice_conc"].where(ds["nsidc_nt_seaice_conc"] <= 1)
        ds["cdr_seaice_conc"] = ds["cdr_seaice_conc"].where(ds["cdr_seaice_conc"] <= 1)

        # nan all spatial interpolation (removes  pole hole)
        ds["nsidc_nt_seaice_conc"] = ds["nsidc_nt_seaice_conc"].where(np.isnan(ds["spatial_interpolation_flag"].values))
        ds["cdr_seaice_conc"] = ds["cdr_seaice_conc"].where(np.isnan(ds["spatial_interpolation_flag"].values))

        tmpNT = np.where(ds["nsidc_nt_seaice_conc"].values.ravel() > 1, 1, 0)
        tmpCDR = np.where(ds["cdr_seaice_conc"].values.ravel() > 1, 1, 0)
        logger.debug(f"G02202 masking flagged NDR, CDR post: {np.sum(tmpNT), np.sum(tmpCDR)}")

        return ds

    def G02202v5_mask_flagged_conc(self, ds: xr.Dataset) -> xr.Dataset:
        """
        2025-09-15 rewrite based on v5 version of G02202
        https://nsidc.org/data/g02202/versions/5

        This subroutine masks out some sea ice concentration values from the
        'cdr_seaice_conc' field in G02202 v5 datasets, based on
        their quality assurance (QA) flags in the 'cdr_seaice_conc_qa_flag' field.

        Specifically, we set sea ice concentration values to NaN where any of the following are true:
        spatial interpolation, temporal interpolation, no observations

        where there are no obs, the data producers sometimes fill the gaps
        via interpolation. we're not going to use their interpolated values
        where there are no obs or interpolation, we mask to nan.

        For detailed description of QA flags see
        https://nsidc.org/sites/default/files/documents/user-guide/g02202-v005-userguide.pdf

        QA Flag Meanings

        flag_meanings = [
            "BT_weather_filter_applied",        # 2**0 = 1
            "NT_weather_filter_applied",        # 2**1 = 2
            "land_spillover_applied",           # 2**2 = 4
            "no_input_data",                    # 2**3 = 8
            "invalid_ice_mask_applied",         # 2**4 = 16
            "spatial_interpolation_applied",    # 2**5 = 32
            "temporal_interpolation_applied",   # 2**6 = 64
            "melt_start_detected"               # 2**7 = 128
        ]

        The BT and NT weather filters and land spillover filters set the sea ice concentration to zero (open ocean)
        so we do not mask out those values
        The invalid ice mask is where ice is never present, so we do not mask those values (they are already zero)
        The melt start detected flag indicates melting ice, which is not a dealbreaker for us
        but we could have a higher uncertainty there.

        Parameters
        ----------
        ds : xr.Dataset
            Input dataset containing 'cdr_seaice_conc' and 'cdr_seaice_conc_qa_flag' variables.
        debug_plotting : bool, optional
            If True, generates debug plots before and after QA flag application, by default False.
        Returns
        -------
        xr.Dataset
            Dataset with 'cdr_seaice_conc' values masked to NaN based on QA flags.
        """

        # the NASA cdr sea ice concentration data array
        cdr_seaice_conc = ds["cdr_seaice_conc"]

        # log the sum of all conc values > 1 before any masking
        logger.debug(f"G2202 masking flagged CDR pre : {np.sum(cdr_seaice_conc.values.ravel() > 1)}")

        # Create a QA Flag DataArray
        # ---------------------------------
        # make a new data array object of dimension [n_flags, y, x]
        # where n_flags = 8, the number of different qa flags

        # the cdf qa flags
        qa_flags = ds["cdr_seaice_conc_qa_flag"]

        # first find the bit values
        # 8 flags, 2^0, 2^1, ..., 2^7
        n = 8
        field = qa_flags.values.astype(np.uint64)  # ensure integer dtype
        bits = 1 << np.arange(n, dtype=np.uint64)  # [1,2,4,...,2**(n-1)]

        # flags has shape (n, *field.shape); flags[k] is 1 where bit 2**k is set
        # .. thank you chatgpt, I don't know how this works        
        index = (slice(None), None) + (None,) * qa_flags.ndim
        flags = ((field[None, ...] & bits[index]) != 0).astype(np.uint8)

        # then make a list of data array objects, one for each flag of dimension [y,x]
        flag_das = []
        for n in range(8):
            flag_da_tmp = xr.DataArray(flags[n, 0, 0], dims=["y", "x"], coords={"flag": 2**n})
            flag_da_tmp.name = f"qa_flag{n}"
            flag_das.append(flag_da_tmp)

        # then concat the qa flag arrays along the 'flag' dimension
        flag_das = xr.concat(flag_das, dim="flag")

        # finally, set to NaN any qa flag values > 0 (indicating the qa flag is set)
        flag_das.values = np.where(flag_das.values > 0, np.nan, 1)

        # Apply QA flags
        # -------------------------
        # apply some QA flags, specifically
        flags_to_nan = [3, 5, 6]  # 2^3=8, 2^5=32, 2^6=64

        # flag 2^3 = 8  : No input data
        # flag 2^5 = 32 : spatial interpolation applied
        # flag 2^6 = 64 : temporal interpolation applied

        # do not nan out flags 2^0, 2^1, 2^2 : BT and NT weather filter applied
        #    and land spillover applied.  these indicate that the sea ice conc was
        #    set to zero (open ocean). without these filters, spurious nonzero sea ice
        #    concentrations could be present due to weather effects or land spillover
        #
        # do not nan out flag 16 (2**4) "invalid ice mask" (ice is never present there)
        #
        # do not nan out flag 128 (2**7) "melt start detected" (ice is melting)
        #                                 melting ice doesn't mean bad data not a dealbreaker
        #                                 although we could have a higher uncertainty

        # make a copy of the original conc field to apply the QA flags to
        cdr_seaice_conc_post_qa = cdr_seaice_conc.copy(deep=True)

        # loop through the flags_to_nan, and multiply the conc field with the
        # corresponding 'flag_das' field, which is labelled on the 'flag' dimension
        # as 2**n
        for n in flags_to_nan:
            cdr_seaice_conc_post_qa = cdr_seaice_conc_post_qa * flag_das.sel(flag=2**n)

        # replace the original conc field with the post-QA field
        ds["cdr_seaice_conc"].values[:] = cdr_seaice_conc_post_qa.values[:]

        logger.debug(f"G2202 masking flagged CDR post: {np.sum(ds['cdr_seaice_conc'].values.ravel())}")
        # finally, drop the qa flag fields and other unneeded fields
        ds = ds.drop_vars(
            [
                "cdr_seaice_conc_interp_spatial_flag",
                "cdr_seaice_conc_qa_flag",
                "cdr_seaice_conc_interp_temporal_flag",
                "cdr_seaice_conc_stdev",
            ]
        )
        return ds

    def GRACE_MASCON(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Mask out land, setting land points to NaN.
        """
        ds["lwe_thickness"] = ds.lwe_thickness.where(ds.land_mask == 0.0)
        ds["uncertainty"] = ds.uncertainty.where(ds.land_mask == 0.0)
        return ds


class PosttransformationFuncs:
    """
    Post transformation functions, to be performed on xr.DataArrays
    """

    def call_functions(self, function_names: Iterable[str], da: xr.DataArray) -> xr.DataArray:
        for func_name in function_names:
            da = self.call_function(func_name, da)
        return da

    def call_function(self, function_name: str, da: xr.DataArray) -> xr.DataArray:
        # Get the function dynamically by name
        func = getattr(self, function_name, None)
        if func:
            try:
                da = func(da)
            except Exception as e:
                logger.exception(f"Error running {function_name}. {e}")
        else:
            raise FuncNotFound(f"Function '{function_name}' not found")
        return da

    def meters_to_cm(self, da: xr.DataArray) -> xr.DataArray:
        """
        Converts meters to centimeters
        """
        da.attrs["units"] = "cm"
        da.values *= 100
        return da

    def kelvin_to_celsius(self, da: xr.DataArray) -> xr.DataArray:
        """
        Converts Kelvin values to Celsius
        """
        da.attrs["units"] = "Celsius"
        da.values -= 273.15
        return da

    def seaice_concentration_to_fraction(self, da: xr.DataArray) -> xr.DataArray:
        """
        Converts seaice concentration values to a fraction by dividing them by 100
        """
        da.attrs["units"] = "1"
        da.values /= 100.0
        return da

    def MEaSUREs_fix_time(self, da: xr.DataArray) -> xr.DataArray:
        """
        time_start and time_end for MEaSUREs_1812 is not acceptable
        this function takes the provided center time, removes the hours:minutes:seconds.ns
        and sets the new time_start and time_end based on that new time
        """

        # remove time from date
        today = str(da.time.values[0])[:10]
        tomorrow = np.datetime64(today, "D") + 1

        da.time_start.values[0] = str(np.datetime64(today, "ns"))
        da.time_end.values[0] = str(np.datetime64(str(tomorrow), "ns"))

        return da
