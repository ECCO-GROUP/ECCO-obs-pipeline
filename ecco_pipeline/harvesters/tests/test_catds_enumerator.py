import unittest
from datetime import datetime
from typing import Iterable

import yaml
from harvesters.enumeration.catds_enumerator import CATDSGranule, CATDS_URL
from harvesters.catds_harvester import CATDS_Harvester


class EndToEndCATDSEnumeratorTestCase(unittest.TestCase):
    granules: Iterable[CATDSGranule]

    @classmethod
    def setUpClass(cls) -> None:
        with open(
            "conf/ds_configs/L3_DEBIAS_LOCEAN_v8_q09.yaml",
            "r",
        ) as stream:
            config = yaml.load(stream, yaml.Loader)
        cls.harvester = CATDS_Harvester(config)
        
    def test_found_granules(self):
        self.assertGreater(len(self.harvester.catds_granules), 0)

    def test_granule_properties(self):
        for granule in self.harvester.catds_granules:
            self.assertIsInstance(granule.mod_time, datetime)

    def test_urls(self):
        for granule in self.harvester.catds_granules:
            self.assertTrue(granule.url.startswith(CATDS_URL))
