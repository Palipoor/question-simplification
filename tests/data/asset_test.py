from src.test_util import FIXTURES_DATA_ROOT, PROJECT_ROOT
from src.data.asset import Asset
from datasets import load_dataset_builder


def test_asset_dataset():
    reader = load_dataset_builder(str(PROJECT_ROOT.joinpath("src/data/asset.py")))
    orig = FIXTURES_DATA_ROOT.joinpath("asset.orig")
    simp = FIXTURES_DATA_ROOT.joinpath("asset.simp")

    filepaths = {"asset.test.orig": str(orig), "asset.test.simp.0": str(simp)}

    result = list(reader._generate_examples(filepaths, "test", 1))
    assert len(result) == 3
