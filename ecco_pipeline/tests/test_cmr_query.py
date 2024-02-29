from typing import Iterable
import unittest
from datetime import datetime
import yaml
from harvesters.enumeration.cmr_enumerator import CMRQuery, CMRGranule
from harvesters.harvesterclasses import Harvester

class EndToEndCMRQueryTestCase(unittest.TestCase):   
    granules: Iterable[CMRGranule]
    
    @classmethod
    def setUpClass(cls) -> None:       
        with open('conf/ds_configs/SEA_SURFACE_HEIGHT_ALT_GRIDS_L4_2SATS_5DAY_6THDEG_V_JPL2205.yaml', 'r') as stream:
            config = yaml.load(stream, yaml.Loader)
        cls.harvester = Harvester(config)
        cmr_query = CMRQuery(cls.harvester)
        
        cls.granules = cmr_query.query()

    def test_correct_collection(self):
        for granule in self.granules:
            self.assertEqual(self.harvester.cmr_concept_id, granule.collection_id)
            
    def test_granule_properties(self):
        for granule in self.granules:
            self.assertIsInstance(granule.mod_time, datetime)
            
    def test_urls(self):
        for granule in self.granules:
            self.assertTrue(granule.url.startswith('http'))
            self.assertTrue(self.harvester.provider in granule.url)