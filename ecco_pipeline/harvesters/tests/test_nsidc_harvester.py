import unittest
import tempfile
from datetime import datetime
import os
import shutil
import yaml
from glob import glob
from harvesters.enumeration.nsidc_enumerator import NSIDC_URL
from harvesters.nsidc_harvester import NSIDC_Harvester


class EndToEndNSIDCHarvesterTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open(
            "conf/ds_configs/G02202_V5.yaml",
            "r",
        ) as stream:
            config = yaml.load(stream, yaml.Loader)
        config["start"] = "19900101T00:00:00Z"
        config["end"] = "19900102T00:00:00Z"

        cls.tempdir = tempfile.mkdtemp()
        cls.harvester = NSIDC_Harvester(config)

        cls.harvester.target_dir = cls.tempdir
        cls.harvester.fetch()


    def test_granule_properties(self):
        for granule in self.harvester.nsidc_granules:
            self.assertIsInstance(granule.mod_time, datetime)

    def test_urls(self):
        for granule in self.harvester.nsidc_granules:
            self.assertTrue(granule.url.startswith(NSIDC_URL))

    def test_dls(self):
        dl_files = glob(f"{self.tempdir}/**/*.nc", recursive=True)
        filenames = [os.path.basename(fp) for fp in dl_files]
        self.assertEqual(len(dl_files), 4)
        self.assertTrue("sic_psn25_19900101_F08_v05r00.nc" in filenames)

    @classmethod
    def tearDownClass(self) -> None:
        shutil.rmtree(self.tempdir)
        return super().tearDown(self)
