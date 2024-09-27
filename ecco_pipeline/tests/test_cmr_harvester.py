import unittest
import tempfile
from datetime import datetime
import os
import shutil
import yaml
from glob import glob
from harvesters.cmr_harvester import CMR_Harvester


class EndToEndCMRHarvesterTestCase(unittest.TestCase):
    # granules: Iterable[CMRGranule]

    @classmethod
    def setUpClass(cls) -> None:
        with open(
            "conf/ds_configs/SEA_SURFACE_HEIGHT_ALT_GRIDS_L4_2SATS_5DAY_6THDEG_V_JPL2205.yaml",
            "r",
        ) as stream:
            config = yaml.load(stream, yaml.Loader)
        config["start"] = "20200101T00:00:00Z"
        config["end"] = "20200105T00:00:00Z"

        cls.tempdir = tempfile.mkdtemp()
        cls.harvester = CMR_Harvester(config)

        cls.harvester.target_dir = cls.tempdir
        cls.harvester.fetch()

    def test_correct_collection(self):
        for granule in self.harvester.cmr_granules:
            self.assertEqual(self.harvester.cmr_concept_id, granule.collection_id)

    def test_granule_properties(self):
        for granule in self.harvester.cmr_granules:
            self.assertIsInstance(granule.mod_time, datetime)

    def test_urls(self):
        for granule in self.harvester.cmr_granules:
            self.assertTrue(granule.url.startswith("http"))
            self.assertTrue(self.harvester.provider in granule.url)

    def test_dls(self):
        dl_files = glob(f"{self.tempdir}/**/*.nc", recursive=True)
        self.assertEqual(len(dl_files), 1)
        self.assertEqual(os.path.basename(dl_files[0]), "ssh_grids_v2205_2020010212.nc")

    @classmethod
    def tearDownClass(self) -> None:
        shutil.rmtree(self.tempdir)
        return super().tearDown(self)
