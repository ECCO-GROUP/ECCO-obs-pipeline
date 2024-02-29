from dataclasses import Field
import logging
from multiprocessing import current_process

from typing import Iterable

import numpy as np
import xarray as xr

logger = logging.getLogger(str(current_process().pid))

# Preprocessing functions
# Requirements: function parameters MUST be file_path, Transformation object
# -----------------------------------------------------------------------------------------------------------------------------------------------

def ATL20_V004_monthly(file_path: str, fields: Iterable[Field]) -> xr.Dataset:
    '''
    Handle data groups
    '''
    vars = [field.name for field in fields]

    ds = xr.open_dataset(file_path, decode_times=True)
    ds = ds[['grid_x', 'grid_y', 'crs']]
    
    var_ds = xr.open_dataset(file_path, group='monthly')[vars]
    merged_ds = xr.merge([ds, var_ds])
    return merged_ds



# Pre-transformation (on Datasets only)
# -----------------------------------------------------------------------------------------------------------------------------------------------

def AVHRR_remove_ice_or_near_ice(ds):
    '''
    Replaces SST values < -0.5 or sea_ice_fraction > 0 to NaN
    '''
    # nonzero sea ice fraction
    if 'sea_ice_fraction' in ds:
        ds.analysed_sst.values = np.where(ds.sea_ice_fraction > 0, np.nan, ds.analysed_sst.values)
        
    # colder than -0.5C
    ds.analysed_sst.values = np.where(ds.analysed_sst <= 273.15-0.5, np.nan, ds.analysed_sst.values)

    return ds


def RDEFT4_remove_negative_values(ds: xr.Dataset) -> xr.Dataset:
    '''
    Replaces negative values with nans for all data vars
    '''
    for field in ds.data_vars:
        if field in ['lat', 'lon']:
            continue
        ds[field].values = np.where(ds[field].values < 0, np.nan, ds[field].values)
    return ds


def G2202_mask_flagged_conc(ds: xr.Dataset) -> xr.Dataset:
    '''
    Masks out values greater than 1 in nsidc_nt_seaice_conc and cdr_seaice_conc
    '''
    logger.debug(f'G2202 masking flagged nt pre   : {np.sum(ds["nsidc_nt_seaice_conc"].values.ravel() > 1)}')
    tmpNT = np.where(ds["nsidc_nt_seaice_conc"].values.ravel() > 1, 1, 0)
    tmpCDR = np.where(ds["cdr_seaice_conc"].values.ravel() > 1, 1, 0)
    logger.debug(f'G2202 masking flagged NDR, CDR pre: {np.sum(tmpNT), np.sum(tmpCDR)}')

    ds['nsidc_nt_seaice_conc'] = ds['nsidc_nt_seaice_conc'].where(ds['nsidc_nt_seaice_conc'] <= 1)
    ds['cdr_seaice_conc'] = ds['cdr_seaice_conc'].where(ds['cdr_seaice_conc'] <= 1)

    # nan all spatial interpolation (removes  pole hole)
    ds['nsidc_nt_seaice_conc'] = ds['nsidc_nt_seaice_conc'].where(np.isnan(ds['spatial_interpolation_flag'].values))
    ds['cdr_seaice_conc'] = ds['cdr_seaice_conc'].where(np.isnan(ds['spatial_interpolation_flag'].values))

    tmpNT = np.where(ds["nsidc_nt_seaice_conc"].values.ravel() > 1, 1, 0)
    tmpCDR = np.where(ds["cdr_seaice_conc"].values.ravel() > 1, 1, 0)
    logger.debug(f'G2202 masking flagged NDR, CDR post: {np.sum(tmpNT), np.sum(tmpCDR)}')

    return ds


def GRACE_MASCON(ds: xr.Dataset) -> xr.Dataset:
    '''
    Mask out land, setting land points to NaN.
    '''
    ds['lwe_thickness'] = ds.lwe_thickness.where(ds.land_mask == 0.0)
    ds['uncertainty'] = ds.uncertainty.where(ds.land_mask == 0.0)
    return ds


# Post-transformations (on DataArrays only)
# -----------------------------------------------------------------------------------------------------------------------------------------------

def meters_to_cm(da: xr.DataArray) -> xr.DataArray:
    '''
    Converts meters to centimeters
    '''
    da.attrs['units'] = 'cm'
    da.values *= 100
    return da

def kelvin_to_celsius(da: xr.DataArray) -> xr.DataArray:
    '''
    Converts Kelvin values to Celsius
    '''
    da.attrs['units'] = 'Celsius'
    da.values -= 273.15
    return da


def seaice_concentration_to_fraction(da: xr.DataArray) -> xr.DataArray:
    '''
    Converts seaice concentration values to a fraction by dividing them by 100
    '''
    da.attrs['units'] = "1"
    da.values /= 100.
    return da


def MEaSUREs_fix_time(da: xr.DataArray) -> xr.DataArray:
    '''
    time_start and time_end for MEaSUREs_1812 is not acceptable
    this function takes the provided center time, removes the hours:minutes:seconds.ns
    and sets the new time_start and time_end based on that new time
    '''

    # remove time from date
    today = str(da.time.values[0])[:10]
    tomorrow = np.datetime64(today, 'D') + 1

    da.time_start.values[0] = str(np.datetime64(today, 'ns'))
    da.time_end.values[0] = str(np.datetime64(str(tomorrow), 'ns'))

    return da
