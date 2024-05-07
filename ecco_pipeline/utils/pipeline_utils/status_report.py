from dataclasses import dataclass
from datetime import datetime
from glob import glob
import os
from typing import Iterable
from itertools import product
import pandas as pd
import yaml
import numpy as np
import json
from conf.global_settings import OUTPUT_DIR
from utils.pipeline_utils import file_utils 

import matplotlib.pyplot as plt


@dataclass
class AggStats():
    ds: str
    grid: str
    var: str
    resolution: str
    first_year: int
    last_year: int
    output_count: int
    
@dataclass
class TxStats():
    ds: str
    grid: str
    var: str
    first_year: int
    last_year: int
    output_count: int

@dataclass
class HarvestStats():
    ds: str
    first_year: int
    last_year: int
    output_count: int

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
        
        glob_path = os.path.join(OUTPUT_DIR, self.name, 'transformed_products', '**', 'transformed', '**', '*nc')
        self.transformed_filepaths = sorted(glob(glob_path))
        self.output_tx_vars, self.output_tx_grids = self.get_vars_grids(self.transformed_filepaths, 'transformed')

        glob_path = os.path.join(OUTPUT_DIR, self.name, 'harvested_granules', '**', '*.*')
        self.harvested_filepaths = sorted(glob(glob_path))

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

    def analyze_outputs(self, stage):
        
        if stage == 'aggregation':
            if not self.output_agg_grids and not self.output_agg_vars:
                return
            for grid, var in sorted(product(self.output_agg_grids, self.output_agg_vars)):
                self.agg_stats_daily.append(self.get_agg_stats(grid, var, 'DAILY'))
                self.agg_stats_monthly.append(self.get_agg_stats(grid, var, 'MONTHLY'))
                
        if stage == 'transformation':
            if not self.output_tx_vars and not self.output_tx_grids:
                return
            for grid, var in sorted(product(self.output_tx_grids, self.output_tx_vars)):
                stats = self.get_tx_stats(grid, var)
                self.tx_stats.append(stats)
        
        if stage == 'harvesting':
            years = [fp.split('/')[-2] for fp in self.harvested_filepaths]
            if years:
                self.harvest_stats = HarvestStats(self.name, min(years, default=np.nan), max(years, default=np.nan), len(self.harvested_filepaths))
        
    def get_agg_stats(self, grid, var, resolution):
        files = list(filter(lambda x: all(token in x for token in [grid, var, resolution]), self.aggregated_filepaths))
        years = [os.path.splitext(os.path.basename(fp))[0][-4:] for fp in files]
        return AggStats(self.name, grid, var, resolution.lower(), min(years, default=np.nan), max(years, default=np.nan), len(files))
    
    def get_tx_stats(self, grid, var):
        files = list(filter(lambda x: all(token in x for token in [grid, var]), self.transformed_filepaths))
        years = [file_utils.get_date(self.config['filename_date_regex'], os.path.basename(fp))[:4] for fp in files]
        return TxStats(self.name, grid, var, min(years, default=np.nan), max(years, default=np.nan), len(files))

    def find_stats(self, stage, grid, var, resolution=''):
        if stage == 'aggregation':
            if resolution == 'daily':
                agg_stats = self.agg_stats_daily
            if resolution == 'monthly':
                agg_stats = self.agg_stats_monthly
            for stats in agg_stats:
                if stats.grid == grid and stats.var == var:
                    return stats
        if stage == 'transformation':
            for stats in self.tx_stats:
                if stats.grid == grid and stats.var == var:
                    return stats


    def package_csv(self):
        pass
    
    def package_json(self):
        agg_stats = dict()
        for grid in self.output_agg_grids:
            grid_stats = dict()
            for var in self.output_agg_vars:
                daily_stats = self.find_stats('aggregation', grid, var, 'daily')
                monthly_stats = self.find_stats('aggregation', grid, var, 'monthly')
                var_stats = {
                    'daily': {
                        'first_year': daily_stats.first_year,
                        'last_year': daily_stats.last_year,
                        'file_count': daily_stats.output_count
                    },
                    'monthly': {
                        'first_year': monthly_stats.first_year,
                        'last_year': monthly_stats.last_year,
                        'file_count': monthly_stats.output_count
                    }
                }
                grid_stats[var] = var_stats
            agg_stats[grid] = grid_stats
            
        tx_stats = dict()
        for grid in self.output_tx_grids:
            grid_stats = dict()
            for var in self.output_tx_vars:
                stats = self.find_stats('transformation', grid, var)
                var_stats = {
                    'first_year': stats.first_year,
                    'last_year': stats.last_year,
                    'file_count': stats.output_count
                }
                grid_stats[var] = var_stats
            tx_stats[grid] = grid_stats
        
        if self.harvest_stats:      
            harvesting_stats = {
                'first_year': self.harvest_stats.first_year,
                'last_year': self.harvest_stats.last_year,
                'file_count': self.harvest_stats.output_count
            }
        else:
            harvesting_stats = {}
        
        ds_dump = {
                'aggregation': agg_stats,
                'transformation': tx_stats,
                'harvesting': harvesting_stats
            }
        
        return ds_dump
    
def generate_comprehensive(export_filename: str):
    config_paths = sorted(glob(f'conf/ds_configs/*.yaml'))
    
    comprehensive_report = {}
    for config_path in config_paths:
        ds_status = DSStatus(config_path)
        for stage in ['aggregation', 'transformation', 'harvesting']:
            ds_status.analyze_outputs(stage)
        comprehensive_report[ds_status.name] = ds_status.package_json()
    report_dir = os.path.join(OUTPUT_DIR, 'reports', 'comprehensive')
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, export_filename), 'w') as f:
        json.dump(comprehensive_report, f)
    return comprehensive_report
        
def to_df(report):
    flattened_data = []
    for dataset, stages in report.items():
        for stage, stage_data in stages.items():
            if stage == 'aggregation':
                for grid, variables in stage_data.items():
                    for variable, resolutions in variables.items():
                        for resolution, years_info in resolutions.items():
                            flattened_data.append({
                                'dataset': dataset,
                                'stage': stage,
                                'grid': grid,
                                'variable': variable,
                                'time_resolution': resolution,
                                'first_year': years_info.get('first_year'),
                                'last_year': years_info.get('last_year'),
                                'file_count': years_info.get('file_count')
                            })
            elif stage == 'transformation':
                for grid, variables in stage_data.items():
                    for variable, years_info in variables.items():
                        flattened_data.append({
                                'dataset': dataset,
                                'stage': stage,
                                'grid': grid,
                                'variable': variable,
                                'first_year': years_info.get('first_year'),
                                'last_year': years_info.get('last_year'),
                                'file_count': years_info.get('file_count')
                            })
            elif stage == 'harvesting':
                flattened_data.append({
                        'dataset': dataset,
                        'stage': stage,
                        'first_year': stage_data.get('first_year'),
                        'last_year': stage_data.get('last_year'),
                        'file_count': stage_data.get('file_count')
                    })

    df = pd.DataFrame(flattened_data).set_index(['dataset', 'stage', 'grid', 'variable', 'time_resolution'])
    return df

def check_delta(new_report: pd.DataFrame):
    reports = sorted(glob(os.path.join(OUTPUT_DIR, 'reports', 'comprehensive', '*.json')))
    if len(reports) > 1:
        old_report_path = reports[-2]
        print(f'Comparing new report with {os.path.basename(old_report_path)}')
    else:
        print(new_report)
        return
    
    with open(old_report_path, 'r') as f:
        old_report = to_df(json.load(f))

    if new_report.index.equals(old_report.index):
        difference_df = new_report.compare(old_report)
        # This gets us the differences already when there are no new rows (datasets or stages for existing datasets)
    else:
        # Get difference in index
        difference_df = new_report.index.difference(old_report.index)
        # Need to compare index and then individual items
        # Subset new index on old index to compare existing items
        # Use difference_df to get new items
        
    if difference_df.size > 0:
        print(difference_df)
    else:
        plt.figure(figsize=(10, 6))
        grouped_agg = new_report.groupby(['dataset'])['file_count'].sum().unstack('dataset').fillna(0)
        grouped_agg.plot(kind='bar', stacked=True)
        plt.title('File Counts by Variable and Stage')
        plt.xlabel('Variable')
        plt.ylabel('File Count')
        plt.show()
        print('No change between reports.')

def generate_reports():
    export_filename = f'pipeline_report_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}.json'
        
    comprehensive_report = generate_comprehensive(export_filename)
    comprehensive_report_df = to_df(comprehensive_report)

    check_delta(comprehensive_report_df)
    
    