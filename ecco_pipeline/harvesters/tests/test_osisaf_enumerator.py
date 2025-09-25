import unittest
from datetime import datetime
from typing import Iterable

import yaml
from harvesters.enumeration.osisaf_enumerator import OSISAFGranule
from harvesters.osisaf_harvester import OSISAF_Harvester


class EndToEndOSISAFEnumeratorTestCase(unittest.TestCase):
    granules: Iterable[OSISAFGranule]

    @classmethod
    def setUpClass(cls) -> None:
        
        with open(
            "conf/ds_configs/SSMIS_OSI-430-a_daily.yaml",
            "r",
        ) as stream:
            config = yaml.load(stream, yaml.Loader)
        cls.harvester = OSISAF_Harvester(config)
        cls.granules = cls.harvester.osisaf_granules

    def test_granule_properties(self):
        for granule in self.granules:
            self.assertIsInstance(granule.mod_time, datetime)

    def test_urls(self):
        for granule in self.granules:
            self.assertTrue(granule.url.startswith("http"))
