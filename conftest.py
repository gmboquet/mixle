"""Root conftest: command-line options that must exist before collection begins.

`pytest_addoption` is honored only in early-loaded conftest files -- the rootdir one, or the
ancestors of an explicit positional path argument. The shard options were first defined in
mixle/tests/conftest.py and worked locally only because local runs happened to pass `mixle/tests/`
positionally; the hosted tier runner passes no path (testpaths comes from pyproject), so pytest
rejected `--num-shards` as unrecognized there. Options live here; the filtering logic stays in
mixle/tests/conftest.py, where `config.getoption` can read them from anywhere.
"""


def pytest_addoption(parser) -> None:
    parser.addoption("--shard-id", type=int, default=0, help="0-based shard index")
    parser.addoption("--num-shards", type=int, default=1, help="total shard count")
