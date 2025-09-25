import unittest
from datetime import datetime
from typing import Iterable

import yaml
from harvesters.enumeration.nsidc_enumerator import NSIDCGranule, NSIDC_URL
from harvesters.nsidc_harvester import NSIDC_Harvester


class EndToEndNSIDCEnumeratorTestCase(unittest.TestCase):
    granules: Iterable[NSIDCGranule]

    @classmethod
    def setUpClass(cls) -> None:
        
        with open(
            "conf/ds_configs/G02202_V5.yaml",
            "r",
        ) as stream:
            config = yaml.load(stream, yaml.Loader)
        cls.harvester = NSIDC_Harvester(config)
        cls.granules = cls.harvester.nsidc_granules

    def test_granule_properties(self):
        for granule in self.granules:
            self.assertIsInstance(granule.mod_time, datetime)

    def test_urls(self):
        for granule in self.granules:
            self.assertTrue(granule.url.startswith(NSIDC_URL))
