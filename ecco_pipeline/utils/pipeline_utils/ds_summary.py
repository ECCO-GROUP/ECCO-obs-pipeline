from glob import glob
import logging
import os
from typing import Iterable, Tuple

import yaml
import numpy as np
import xarray as xr
import ecco_v4_py as ecco
from matplotlib import pyplot as plt

from conf.global_settings import OUTPUT_DIR

logger = logging.getLogger('pipeline')


class InsufficientFilesError(Exception):
    pass

class DSStatus():
    
    def __init__(self, config_path: str) -> None:
        self.name: str = os.path.splitext(os.path.basename(config_path))[0]
        self.config: dict = self.load_config(config_path)
        self.expected_variables:Iterable[str] = [field['name'] for field in sorted(self.config['fields'], key=lambda x: x['name'])]
        self.agg_stats_daily = []
        self.agg_stats_monthly = []
        self.tx_stats = []
        self.harvest_stats = {}
        
        self.glob_outputs()
        
    def load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    def glob_outputs(self) -> Iterable[str]:
        glob_path = os.path.join(OUTPUT_DIR, self.name, 'transformed_products', '**', 'aggregated', '**', 'netCDF', '*nc')
        self.aggregated_filepaths = sorted(glob(glob_path))
        self.output_agg_vars, self.output_agg_grids = self.get_vars_grids(self.aggregated_filepaths, 'aggregated')

    def get_vars_grids(self, filepaths: Iterable[str], stage: str) -> Tuple[Iterable[str], Iterable[str]]:
        unique_grids = set()
        unique_variables = set()
        for tokenized_fp in [fp.split('/') for fp in filepaths]:
            agg_index = tokenized_fp.index(stage)
            grid = tokenized_fp[agg_index - 1]
            unique_grids.add(grid)
            variable = tokenized_fp[agg_index + 1]
            unique_variables.add(variable)
        return sorted(list(unique_variables)), sorted(list(unique_grids))
    
    def make_stats(self, grid: str, var: str, time_res: str) -> Tuple[xr.Dataset, str]:
        
        scenario_filepaths = [fp for fp in self.aggregated_filepaths
                              if (grid in fp) and
                              (f'aggregated/{var}/' in fp) and
                              (time_res in fp)]
        if not scenario_filepaths:
            raise InsufficientFilesError(f'No aggregated files found for {time_res} {grid}, {var}')
        scenario_filepaths.sort()
        logger.info(f'Creating report for {grid} {var} {time_res}')
        var_name = f'{var}_interpolated_to_{grid}'
        
        ds = xr.open_mfdataset(scenario_filepaths, combine='nested', concat_dim='time')
        units = ds[var_name].attrs['units']
        logger.info(f'Completed opening {len(scenario_filepaths)} agg files')  
        
        valid_times = ds[var_name].notnull().any(dim=['tile', 'j', 'i'])
        ds = ds.sel(time=valid_times)
        logger.info('Completed filtering nan time slices')
        
        mins = ds[var_name].min(dim='time', skipna=True)
        mins.name = mins.name + '_min'
        means = ds[var_name].mean(dim='time', skipna=True)
        means.name = means.name + '_mean'
        maxs = ds[var_name].max(dim='time', skipna=True)
        maxs.name = maxs.name + '_max'
        
        
        global_mins = ds[var_name].min(dim=['tile', 'j', 'i' ], skipna=True)
        global_mins.name = 'global_' + global_mins.name + '_min'
        global_means = ds[var_name].mean(dim=['tile', 'j', 'i' ], skipna=True)
        global_means.name = 'global_' + global_means.name + '_mean'
        global_std = ds[var_name].std(dim=['tile', 'j', 'i' ], skipna=True)
        global_std.name = 'global_' + global_std.name + '_std'
        global_maxs = ds[var_name].max(dim=['tile', 'j', 'i' ], skipna=True)
        global_maxs.name = 'global_' + global_maxs.name + '_max'
        global_counts = ds[var_name].count(dim=['tile', 'j', 'i' ])
        global_counts.name = f'global_{var_name}_nn_count'
        global_counts_lat = ds[var_name].count(dim=['tile', 'i' ])
        global_counts_lat.name = f'global_{var_name}_nn_count_lat'
        global_counts_lon = ds[var_name].count(dim=['tile', 'j',])
        global_counts_lon.name = f'global_{var_name}_nn_count_lon'
        
        stats_ds = xr.merge([mins, means, maxs, global_mins, global_means, global_maxs, global_std, global_counts, global_counts_lat, global_counts_lon])
        logger.info(f'Completed computing stats of agg files')    

        coord_encoding = {}
        for coord in stats_ds.coords:
            coord_encoding[coord] = {'_FillValue': None, 'dtype': 'float32'}
            if coord == 'time' or coord == 'time_bnds':
                coord_encoding[coord] = {'dtype': 'int32'}
        coord_encoding['time'] = {'units': 'hours since 1980-01-01'}

        var_encoding = {}
        for variable in stats_ds.data_vars:
            var_encoding[variable] = {'zlib': True,
                                'complevel': 5,
                                'shuffle': True}

        encoding = {**coord_encoding, **var_encoding}
        
        filename = f'{self.name}_{grid}_{var}_{time_res}.nc'
        output_path = os.path.join(OUTPUT_DIR, 'ds_stats', self.name, 'stats', filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        stats_ds.to_netcdf(output_path, encoding=encoding)
        logger.info(f'Completed saving stats netcdf')    
        
        stats_ds = xr.open_dataset(output_path)
        logger.info(f'Completed loading up stats netcdf')
        
        self.stats_ds = stats_ds
        self.units = units
    
    def make_maps(self, var_name: str, title: str, map_type: str, filename: str):
        
        if map_type == 'robinson':
            nrows = 3
        elif map_type == 'hemisphere':
            nrows = 4
            
        fig = plt.figure(figsize=(18,10), dpi=120)
        
        plot_i = 1
        if map_type == 'robinson':
            for i, stat in enumerate(['_min', '_mean', '_max'], start=plot_i):
                ecco.plot_proj_to_latlon_grid(
                    self.stats_ds.XC,
                    self.stats_ds.YC,
                    self.stats_ds[var_name + stat],
                    plot_type='pcolormesh',
                    cmap='RdBu_r',
                    show_colorbar=True,
                    subplot_grid=[nrows,3,i]
                )
                plt.title(stat.replace('_','').capitalize())
                plot_i += 1
        elif map_type == 'hemisphere':
            # Northern hemisphere
            for i, stat in enumerate(['_min', '_mean', '_max'], start=plot_i):
                ecco.plot_proj_to_latlon_grid(
                    self.stats_ds.XC,
                    self.stats_ds.YC,
                    self.stats_ds[var_name + '_min'],
                    projection_type='stereo',
                    plot_type='pcolormesh',
                    cmap='RdBu_r',
                    show_colorbar=True,
                    subplot_grid=[nrows,3,i]
                )
                plt.title(stat.replace('_','').capitalize())
                plot_i += 1
                
            # Southern hemisphere
            for i, stat in enumerate(['_min', '_mean', '_max'], start=plot_i):
                ecco.plot_proj_to_latlon_grid(
                    self.stats_ds.XC,
                    self.stats_ds.YC,
                    self.stats_ds[var_name + '_min'],
                    lat_lim=-40,
                    projection_type='stereo',
                    plot_type='pcolormesh',
                    cmap='RdBu_r',
                    show_colorbar=True,
                    subplot_grid=[nrows,3,i]
                )
                plt.title(stat.replace('_','').capitalize())
                plot_i += 1
                
        for i, stat in enumerate(['_min', '_mean', '_max'], start=plot_i):
            ax = plt.subplot(nrows, 3, i)
            ax.grid()
            if 'mean' in stat:
                lower_bound = self.stats_ds[f'global_{var_name}{stat}'] - self.stats_ds[f'global_{var_name}{stat}']
                upper_bound = self.stats_ds[f'global_{var_name}{stat}'] + self.stats_ds[f'global_{var_name}{stat}']
                ax.fill_between(self.stats_ds['time'], lower_bound, upper_bound, alpha=.25)
                ax.plot(self.stats_ds['time'], self.stats_ds[f'global_{var_name}{stat}'])
            else:
                ax.plot(self.stats_ds['time'], self.stats_ds[f'global_{var_name}{stat}'])
            ax.set_title(stat.replace('_','').capitalize())
            ax.set_ylabel(self.units)
            plot_i += 1
        
        for i, stat in enumerate(['_nn_count'], start=plot_i):
            ax = plt.subplot(nrows, 3, i)
            ax.grid()
            ax.plot(self.stats_ds['time'], self.stats_ds[f'global_{var_name}{stat}'])
            ax.set_title(stat.replace('_','').capitalize())
            ax.set_ylabel(self.units)
            plot_i += 1
            
        fig.autofmt_xdate()
        plt.suptitle(title, fontsize=20)
        plt.tight_layout()

        output_path = os.path.join(OUTPUT_DIR, 'ds_stats', self.name, 'plots', filename.replace('.nc', '.png'))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f'Saving maps to {output_path}')
        plt.savefig(output_path)
        
    
def generate_comprehensive():
    config_paths = sorted(glob('conf/ds_configs/*.yaml'))

    for config_path in config_paths:
        if 'ATL21_V003_monthly' not in config_path:
            continue
        ds_status = DSStatus(config_path)
        logger.info(f'Making dataset analysis report for {ds_status.name}')
        try:
            if not ds_status.output_agg_grids:
                raise InsufficientFilesError(f'No aggregation files for {ds_status.name}')
        except InsufficientFilesError as e:
            logger.warning(e)
        for grid in ds_status.output_agg_grids:
            for var in ds_status.output_agg_vars:
                for time_res in ['DAILY', 'MONTHLY']:
                    try:
                            
                        ds_status.make_stats(grid, var, time_res)
                        
                        if 'hemi_pattern' in ds_status.config:
                            map_type = 'hemisphere'
                        else:
                            map_type = 'robinson'
                        maps_title = f'{time_res.lower().capitalize()} {ds_status.name}\n{var} interpolated to {grid}'
                        var_name = f'{var}_interpolated_to_{grid}'
                        filename = f'{ds_status.name}_{grid}_{var}_{time_res}.nc'
                        
                        ds_status.make_maps(var_name, maps_title, map_type, filename)

                    except InsufficientFilesError as e:
                        logger.warning(e)

        
def generate_reports():
    generate_comprehensive()