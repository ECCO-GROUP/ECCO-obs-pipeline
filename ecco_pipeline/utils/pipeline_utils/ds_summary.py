from glob import glob
import os
from typing import Iterable

import yaml
import numpy as np
import xarray as xr
import ecco_v4_py as ecco
from matplotlib import pyplot as plt


# from conf.global_settings import OUTPUT_DIR
OUTPUT_DIR = '/Users/marlis/Developer/ECCO/ecco_output'
# from utils.pipeline_utils import file_utils 

import matplotlib.pyplot as plt


class InsufficientFilesError(Exception):
    pass

class DSStatus():
    
    def __init__(self, config_path: str) -> None:
        self.name: str = os.path.splitext(os.path.basename(config_path))[0]
        with open(config_path, 'r') as f:
            self.config: dict = yaml.load(f, Loader=yaml.SafeLoader)
        self.expected_variables:Iterable[str] = [field['name'] for field in sorted(self.config['fields'], key=lambda x: x['name'])]
        self.agg_stats_daily = []
        self.agg_stats_monthly = []
        self.tx_stats = []
        self.harvest_stats = {}
        
        self.glob_outputs()

    def glob_outputs(self) -> Iterable[str]:
        glob_path = os.path.join(OUTPUT_DIR, self.name, 'transformed_products', '**', 'aggregated', '**', 'netCDF', '*nc')
        self.aggregated_filepaths = sorted(glob(glob_path))
        self.output_agg_vars, self.output_agg_grids = self.get_vars_grids(self.aggregated_filepaths, 'aggregated')

    def get_vars_grids(self, filepaths, stage):
        unique_grids = set()
        unique_variables = set()
        for tokenized_fp in [fp.split('/') for fp in filepaths]:
            agg_index = tokenized_fp.index(stage)
            grid = tokenized_fp[agg_index - 1]
            unique_grids.add(grid)
            variable = tokenized_fp[agg_index + 1]
            unique_variables.add(variable)
        return sorted(list(unique_variables)), sorted(list(unique_grids))
    
    def make_plots(self, grid, var, time_res: str, ds_name: str, map_type: str):
        scenario_filepaths = [fp for fp in self.aggregated_filepaths
                              if (grid in fp) and
                              (var in fp) and
                              (time_res in fp)]
        if not scenario_filepaths:
            raise InsufficientFilesError(f'No aggregated files found for {time_res} {grid}, {var}')
        
        scenario_filepaths.sort()
        if len(scenario_filepaths) > 1:
            opened_files = [xr.open_dataset(file) for file in scenario_filepaths]
            ds = xr.concat(opened_files, dim='time')
        else:
            ds = xr.open_dataset(scenario_filepaths[0])
            
        maps_title = f'{time_res.lower().capitalize()} {ds_name}\n{var} interpolated to {grid}'
        var_name = f'{var}_interpolated_to_{grid}'
        make_maps(ds, var_name, maps_title, map_type)
    
def make_maps(ds: xr.Dataset, var_name: str, title: str, map_type: str):

    mins = np.nanmin(ds[var_name].values, axis=(0))
    means = np.nanmean(ds[var_name].values, axis=(0))
    maxs = np.nanmax(ds[var_name].values, axis=(0))
    
    min_mean = np.nanmin(means.ravel())
    max_mean = np.nanmax(means.ravel())
    
    if map_type == 'robinson':
        fig, axs = plt.subplots(2,3, figsize=(18,10), dpi=120)
        
        ecco.plot_proj_to_latlon_grid(
            ds.XC,
            ds.YC,
            mins,
            plot_type='pcolormesh',
            cmap='RdBu_r',
            cmin=min_mean, cmax=max_mean,
            subplot_grid=[2,3,1]
        )
        plt.title('Min')
        
        ecco.plot_proj_to_latlon_grid(
            ds.XC,
            ds.YC,
            means,
            plot_type='pcolormesh',
            cmap='RdBu_r',
            cmin=min_mean, cmax=max_mean,
            subplot_grid=[2,3,2]
        )
        plt.title('Mean')
        
        ecco.plot_proj_to_latlon_grid(
            ds.XC,
            ds.YC,
            maxs,
            plot_type='pcolormesh',
            cmap='RdBu_r',
            cmin=min_mean, cmax=max_mean,
            subplot_grid=[2,3,3]
        )
        plt.title('Max')
        
    elif map_type == 'hemisphere':
        fig, axs = plt.subplots(3,3, figsize=(18,10), dpi=120)
        
        ecco.plot_proj_to_latlon_grid(
            ds.XC,
            ds.YC,
            mins,
            projection_type='stereo',
            plot_type='pcolormesh',
            cmap='RdBu_r',
            cmin=min_mean, cmax=max_mean,
            subplot_grid=[3,3,1]
        )
        plt.title('Min')
        
        ecco.plot_proj_to_latlon_grid(
            ds.XC,
            ds.YC,
            means,
            projection_type='stereo',
            plot_type='pcolormesh',
            cmap='RdBu_r',
            cmin=min_mean, cmax=max_mean,
            subplot_grid=[3,3,2]
        )
        plt.title('Mean')
        
        ecco.plot_proj_to_latlon_grid(
            ds.XC,
            ds.YC,
            maxs,
            projection_type='stereo',
            plot_type='pcolormesh',
            cmap='RdBu_r',
            cmin=min_mean, cmax=max_mean,
            subplot_grid=[3,3,3]
        )
        plt.title('Max')
        
        ecco.plot_proj_to_latlon_grid(
            ds.XC,
            ds.YC,
            mins,
            lat_lim=-40,
            projection_type='stereo',
            plot_type='pcolormesh',
            cmap='RdBu_r',
            cmin=min_mean, cmax=max_mean,
            subplot_grid=[3,3,4]
        )
        plt.title('Min')
        
        ecco.plot_proj_to_latlon_grid(
            ds.XC,
            ds.YC,
            means,
            lat_lim=-40,
            projection_type='stereo',
            plot_type='pcolormesh',
            cmap='RdBu_r',
            cmin=min_mean, cmax=max_mean,
            subplot_grid=[3,3,5]
        )
        plt.title('Mean')
        
        ecco.plot_proj_to_latlon_grid(
            ds.XC,
            ds.YC,
            maxs,
            lat_lim=-40,
            projection_type='stereo',
            plot_type='pcolormesh',
            cmap='RdBu_r',
            cmin=min_mean, cmax=max_mean,
            subplot_grid=[3,3,6]
        )
        plt.title('Max')
    
    axs[-1][0].grid()
    axs[-1][0].plot(ds['time'].values, np.nanmin(ds[var_name].values, axis=(1,2,3)))
    axs[-1][0].set_title('Min')
    axs[-1][0].set_ylabel(ds[var_name].attrs['units'])
    
    axs[-1][1].grid()
    mean = np.nanmean(ds[var_name].values, axis=(1,2,3))
    std = np.nanstd(ds[var_name].values, axis=(1,2,3))
    axs[-1][1].fill_between(ds['time'], mean - std, mean + std, alpha=.25)
    axs[-1][1].plot(ds['time'].values, mean)
    axs[-1][1].set_title('Mean')
    axs[-1][1].set_ylabel(ds[var_name].attrs['units'])
    
    axs[-1][2].grid()
    axs[-1][2].plot(ds['time'].values, np.nanmax(ds[var_name].values, axis=(1,2,3)))
    axs[-1][2].set_title('Max')
    axs[-1][2].set_ylabel(ds[var_name].attrs['units'])
    
    fig.autofmt_xdate()
    plt.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()
    
def generate_comprehensive(export_filename: str):
    config_paths = sorted(glob(f'ecco_pipeline/conf/ds_configs/*.yaml'))

    for config_path in config_paths[-6:]:
        ds_status = DSStatus(config_path)
        print(f'Making dataset analysis report for {ds_status.name}')
        if not ds_status.output_agg_grids:
            raise InsufficientFilesError(f'No aggregation files for {ds_status.name}')
        for grid in ds_status.output_agg_grids:
            for var in ds_status.output_agg_vars:
                for time_res in ['DAILY', 'MONTHLY']:
                    try:
                        if 'hemi_pattern' in ds_status.config:
                            map_type = 'hemisphere'
                        else:
                            map_type = 'robinson'
                        ds_status.make_plots(grid, var, time_res, ds_status.name, map_type)
                    except InsufficientFilesError as e:
                        print(e)
                break
            break
        break
        
def generate_reports():
    generate_comprehensive('')

if __name__ == '__main__':
    generate_reports()