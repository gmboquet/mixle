"""automatic.get_estimator accepts a DataFrame / DataSource / bare list (not only a list)."""

import unittest

import numpy as np

from pysp.data import MaterializedSource
from pysp.inference.estimation import fit
from pysp.utils.automatic import get_estimator
from pysp.utils.automatic.profiling import normalize_input


class AutomaticInputTypesTest(unittest.TestCase):
    def test_dataframe_is_profiled_per_column(self):
        pd = __import__("pytest").importorskip("pandas")
        rng = np.random.RandomState(0)
        df = pd.DataFrame({"x": rng.normal(0, 1, 1500), "k": rng.poisson(3, 1500)})
        self.assertEqual(type(get_estimator(df)).__name__, "CompositeEstimator")
        # a single-column frame collapses to the scalar leaf
        self.assertEqual(len(normalize_input(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))), 3)

    def test_datasource_is_profiled_as_records(self):
        rng = np.random.RandomState(1)
        src = MaterializedSource(list(rng.normal(5.0, 2.0, 1500)))
        m = fit(src, get_estimator(src), max_its=15, out=None)
        self.assertEqual(type(m).__name__, "GaussianDistribution")

    def test_bare_list_unchanged(self):
        data = [1.0, 2.0, 3.0]
        self.assertIs(normalize_input(data), data)


if __name__ == "__main__":
    unittest.main()
