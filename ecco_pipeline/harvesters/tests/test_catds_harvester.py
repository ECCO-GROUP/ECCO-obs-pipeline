import unittest
import tempfile
from datetime import datetime
import os
import shutil
import yaml
from glob import glob
from harvesters.enumeration.catds_enumerator import CATDS_URL
from harvesters.catds_harvester import CATDS_Harvester


class EndToEndCATDSHarvesterTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open(
            "conf/ds_configs/L3_DEBIAS_LOCEAN_v8_q09.yaml",
            "r",
        ) as stream:
            config = yaml.load(stream, yaml.Loader)
        config["start"] = "2010208T00:00:00Z"
        config["end"] = "20100210T00:00:00Z"

        cls.tempdir = tempfile.mkdtemp()
        cls.harvester = CATDS_Harvester(config)

        cls.harvester.target_dir = cls.tempdir
        cls.harvester.fetch()


    def test_granule_properties(self):
        for granule in self.harvester.catds_granules:
            self.assertIsInstance(granule.mod_time, datetime)

    def test_urls(self):
        for granule in self.harvester.catds_granules:
            self.assertTrue(granule.url.startswith(CATDS_URL))
            self.assertTrue(self.harvester.provider in granule.url)

    def test_dls(self):
        dl_files = glob(f"{self.tempdir}/**/*.nc", recursive=True)
        self.assertEqual(len(dl_files), 1)
        self.assertEqual(os.path.basename(dl_files[0]), "SMOS_L3_DEBIAS_LOCEAN_AD_20100209_EASE_09d_25km_v10.nc")

    @classmethod
    def tearDownClass(self) -> None:
        shutil.rmtree(self.tempdir)
        return super().tearDown(self)
