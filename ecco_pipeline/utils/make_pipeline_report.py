#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import numpy as np
import yaml
import sys
import argparse, subprocess
from datetime import date
import datetime
#================================================================================================
def main(observations_dir:Path, conf_dir:Path, export_filename=''):

    yamls = load_yamls(conf_dir)

    ds_found = read_dataset_dirs(observations_dir)

    #def export_ds_found_dict(ds_found:dict, yamls:dict, observations_dir:Path, export_filename=''):
    export_ds_found_dict(ds_found, yamls, observations_dir, conf_dir, export_filename)

    return ds_found


#================================================================================================
def load_yamls(conf_dir:Path):

    all_yaml_files = np.sort(list(conf_dir.glob('*.yaml')))
    yamls = dict()

    print('YAML FILES FOUND:')
    print('-----------------')
    for yf in all_yaml_files:
        print(f' yf name {yf.name}')
        
    for yf in all_yaml_files:
        # safely load the yaml file into a dictionary
        with open(yf,'r') as f:
            tmp = yaml.load(f, Loader=yaml.FullLoader)

        yaml_name = yf.name[:-5]
        yamls[yaml_name] = tmp

    return yamls


#================================================================================================
def read_dataset_dirs(observations_dir:Path):
    datasets = np.sort(list(observations_dir.glob('*')))

    ds_found = dict()
    print('parsing observations_dir')
    
    for ds in datasets:
        print(f'found {ds.name}')
        ds_found[ds.name] = dict()
        grids = list((ds / 'transformed_products').glob('*'))
        
        for g in grids:
            ds_found[ds.name][g.name] = dict()
            
            print(f'...found   {g.name}')
            vars= list((ds / 'transformed_products' / g/ 'aggregated').glob('*'))

            for var in vars:
                ds_found[ds.name][g.name][var.name] = dict()
                print(f'... found      {var.name}')
                fp = Path(ds / 'transformed_products' / g/ 'aggregated' / var / 'netCDF')
                mon_files =  list(fp.glob(f'*{g.name}_MONTHLY*nc'))
                day_files =  list(fp.glob(f'*{g.name}_DAILY*nc'))

                if len(mon_files) > 0:
                    mon_fns = [file.name for file in mon_files]
                else:
                    mon_fns = []

                if len(day_files) > 0:
                    day_fns = [file.name for file in day_files]            
                else:
                    day_fns = []

                ds_found[ds.name][g.name][var.name]['file_path']  = fp
                ds_found[ds.name][g.name][var.name]['mon_files']  = mon_fns
                ds_found[ds.name][g.name][var.name]['day_files']  = day_fns


    return ds_found
            

#================================================================================================
def export_ds_found_dict(ds_found:dict, yamls:dict, observations_dir:Path, 
                         conf_dir:Path, export_filename=''):

    # save the current stdout setting (screen)
    # we'll reset stdout to tmp_stdout at the end of the function
    tmp_stdout= sys.stdout

    # if an export_filename is provided, then redirect stdout to that file
    # otherwise, just print to the screen
    if type(export_filename) == str and len(export_filename) >0:
        try:
            sys.stdout = open(export_filename,'wt')
        except:
            print('could not make filename')
            return ''
    
    today = date.today()

    # print header containing the name of all datasets found in the observations_dir
    print(f'===================================================')
    print(f'ECCO-pipeline Aggregated Observations Report')
    print("report date: ", datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))
    print(f'===================================================')

    print(f'\nobservations dir  : {observations_dir}')
    print(f'ds_conf (yaml) dir: {conf_dir}')

    print('\nDatasets found in observations directory:')
    print('---------------------------------------------------')

    for ds in ds_found.keys():
        print(f'{ds.ljust(50)}    YAML:{ds in yamls}')

    yamls_not_in_ds_found = [y for y in yamls.keys() if y not in ds_found.keys()]
    if len(yamls_not_in_ds_found) > 0:
        print('\nYAMLs found without a matching dataset:')
        print('--------------------------------------')
        for y in yamls_not_in_ds_found:
            print(y)



    print('\n\nAggregated files found in observations directory:')
    print('===================================================')

    # loop through each dataset in ds_found
    for ds in ds_found.keys():
        print('---------------------------------------------------')
        
        # print the dataset name and the title and doi from the yaml
        if ds in yamls:
            print(f"{ds}\n{yamls[ds]['original_dataset_title']}")
            print(f"doi:{yamls[ds]['original_dataset_doi']}")
        else:
            print(f'{ds}\nYAML not founds in ds_conf')
    
        # print the grid names and the variables in each grid
        print('---------------------------------------------------')
        if len(ds_found[ds].keys()) == 0:
            print ('\nNO DATA!\n')
        
        # loop through each grid in the dataset
        for g in ds_found[ds].keys():
            print(f'\n => {g}  ')
    
            # print the number of monthly and daily files for each variable in this grid
            for v in ds_found[ds][g].keys():
                tmp = ds_found[ds][g][v];
    
                # monthly files
                t2 = len(tmp['mon_files'])
                if t2  > 0:
                    mfy = tmp['mon_files'][0][-7:-3]
                    mly = tmp['mon_files'][-1][-7:-3]
                    mflys = f'[{mfy}-{mly}]'
                else:
                    mflys = ''
                    
                # daily files
                t3 = len(tmp['day_files'])
                if t3 > 0:
                    dfy = tmp['day_files'][0][-7:-3]
                    dly = tmp['day_files'][-1][-7:-3]
                    dflys = f'[{dfy}-{dly}]'
                else:
                    dflys = ''
                    
                print(f"     * {v.ljust(28)}   MON:{str(len(tmp['mon_files'])).ljust(2)} {mflys}  DAY:{str(len(tmp['day_files'])).ljust(2)} {dflys}")
        print('\n')
    # return stdout to the original setting (screen )
    sys.stdout = tmp_stdout


#================================================================================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-od' , '--observations_dir', help="root 'observations' directory", required=True)
    parser.add_argument('-dcd', '--ds_conf_dir',      help="directory with ds_conf yamls", required=True)    
    parser.add_argument('-ef' , '--export_to_file',  help="export report to file", required=False, \
                        type=bool, default=False)
    parser.add_argument('-rfn', '--export_filename',  help="filename to export report to", required=False,
                        default='pipeline_report.txt')
    parser.add_argument('-ds' , '--date_stamp',    help="add date stamp to export_filename", required=False,\
                         type=bool, default=True)
    

    args = parser.parse_args()

    # make observations_dir and conf_dir into Path objects
    try:
        observations_dir = Path(args.observations_dir)
    except:
        print('could not find observations_dir')
        sys.exit()

    try:
        conf_dir = Path(args.ds_conf_dir)
    except:
        print('could not find ds_conf_dir')

    if args.export_to_file:
        export_filename = args.export_filename
        print(export_filename, type(export_filename))
        if args.date_stamp:
            export_filename = export_filename.rsplit('.', 1)[0]
            export_filename = f'{export_filename}_{datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}.txt'
            
        print(f'exporting report to {export_filename}')

    else:
        export_filename = ''
        print('printing report to screen')



    ds_found = main(observations_dir, conf_dir, export_filename)