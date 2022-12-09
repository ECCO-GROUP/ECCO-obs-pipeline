import os
from pathlib import Path

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
#OUTPUT_DIR = Path('/Users/marlis/Developer/ECCO/ecco_output')
OUTPUT_DIR = Path('/ecco_nfs_1/shared/ECCO-pipeline/observations')
GRIDS = ['ECCO_llc90']

SOLR_HOST = 'http://localhost:8983/solr/'
SOLR_COLLECTION = 'ecco_datasets'

os.chdir(ROOT_DIR)
