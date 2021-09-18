import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parents[1].resolve()
TEST_ROOT = PROJECT_ROOT.joinpath("tests")
FIXTURES_ROOT = PROJECT_ROOT.joinpath("test_fixtures")
FIXTURES_DATA_ROOT = PROJECT_ROOT.joinpath("test_fixtures", "data")
