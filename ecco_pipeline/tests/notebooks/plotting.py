import numpy as np
import ecco_v4_py as ecco
from matplotlib import pyplot as plt
import netCDF4 as nc4


def plot_latlon(da, vmin, vmax):
    if 'time' in da.dims:
        data = da.isel(time=0)
    else:
        data = da
    if not vmin:
        vmin = da.attrs['valid_min']
    if not vmax:
        vmax = da.attrs['valid_max']
    data = data.where(data != nc4.default_fillvals['f4'], np.nan)
    ecco.plot_tiles(data,
                    cmin=vmin,
                    cmax=vmax,
                    cmap='RdYlBu_r',
                    show_colorbar=True,
                    layout='latlon',
                    show_tile_labels=False,
                    rotate_to_latlon=True, Arctic_cap_tile_location=10)
    plt.suptitle(da.name + '\n' + str(data.time.values)[0:10])
    plt.show()


def plot_nh(da, vmin, vmax):
    if 'time' in da.dims:
        data = da.isel(time=0)
    else:
        data = da
    if not vmin:
        vmin = da.attrs['valid_min']
    if not vmax:
        vmax = da.attrs['valid_max']
    data = data.where(data != nc4.default_fillvals['f4'], np.nan)

    ecco.plot_proj_to_latlon_grid(da.XC, da.YC,
                                  data,
                                  projection_type='stereo',
                                  plot_type='contourf',
                                  show_colorbar=True,
                                  cmap='RdYlBu_r',
                                  dx=1, dy=1, cmin=da.attrs['valid_min'], cmax=da.attrs['valid_max'],
                                  lat_lim=35)

    plt.suptitle(da.name + '\n' + str(data.time.values)[0:10])
    plt.show()


def plot_sh(da, vmin, vmax):
    if 'time' in da.dims:
        data = da.isel(time=0)
    else:
        data = da
    if not vmin:
        vmin = da.attrs['valid_min']
    if not vmax:
        vmax = da.attrs['valid_max']
    data = data.where(data != nc4.default_fillvals['f4'], np.nan)

    ecco.plot_proj_to_latlon_grid(da.XC, da.YC,
                                  data,
                                  projection_type='stereo',
                                  plot_type='contourf',
                                  show_colorbar=True,
                                  cmap='RdYlBu_r',
                                  dx=1, dy=1, cmin=da.attrs['valid_min'], cmax=da.attrs['valid_max'],
                                  lat_lim=-35)

    plt.suptitle(da.name + '\n' + str(data.time.values)[0:10])
    plt.show()


def plot_tpose(da, vmin, vmax):
    if 'time' in da.dims:
        data = da.isel(time=0)
    else:
        data = da
    if not vmin:
        vmin = da.attrs['valid_min']
    if not vmax:
        vmax = da.attrs['valid_max']
    p = plt.pcolormesh(da.XC, da.YC, data, vmin=vmin, vmax=vmax)
    plt.colorbar(p)
    plt.show()


def make_plot(da, plot_type='default', vmin='', vmax=''):
    if plot_type == 'default':
        return plot_latlon(da, vmin, vmax)
    elif plot_type == 'nh':
        return plot_nh(da, vmin, vmax)
    elif plot_type == 'sh':
        return plot_sh(da, vmin, vmax)
    elif plot_type == 'tpose':
        return plot_tpose(da, vmin, vmax)
