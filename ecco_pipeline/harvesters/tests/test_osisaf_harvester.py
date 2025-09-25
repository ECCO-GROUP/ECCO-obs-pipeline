import unittest
import tempfile
from datetime import datetime
import os
import shutil
import yaml
from glob import glob
from harvesters.osisaf_harvester import OSISAF_Harvester


class EndToEndCATDSHarvesterTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open(
            "conf/ds_configs/SSMIS_OSI-430-a_daily.yaml",
            "r",
        ) as stream:
            config = yaml.load(stream, yaml.Loader)
        config["start"] = "20241212T00:00:00Z"
        config["end"] = "20241213T00:00:00Z"

        cls.tempdir = tempfile.mkdtemp()
        cls.harvester = OSISAF_Harvester(config)

        cls.harvester.target_dir = cls.tempdir
        cls.harvester.fetch()


    def test_granule_properties(self):
        for granule in self.harvester.osisaf_granules:
            self.assertIsInstance(granule.mod_time, datetime)

    def test_urls(self):
        for granule in self.harvester.osisaf_granules:
            self.assertTrue(granule.url.startswith("http"))

    def test_dls(self):
        dl_files = glob(f"{self.tempdir}/**/*.nc", recursive=True)
        filenames = [os.path.basename(fp) for fp in dl_files]
        self.assertEqual(len(dl_files), 4)
        self.assertTrue("ice_conc_sh_ease2-250_icdr-v3p0_202412121200.nc" in filenames)

    @classmethod
    def tearDownClass(self) -> None:
        shutil.rmtree(self.tempdir)
        return super().tearDown(self)
