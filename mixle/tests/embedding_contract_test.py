"""Fast immutable-spec checks for shared categorical embeddings."""

import unittest
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.models.embedding import CategoricalEmbedding, resolve_embedding  # noqa: E402


class SharedEmbeddingContractTest(unittest.TestCase):
    def test_dimensions_must_be_exact_positive_integers(self):
        for value in (0, -1, 2.5, True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                CategoricalEmbedding(value, 4)
            with self.assertRaisesRegex(ValueError, "positive integer"):
                CategoricalEmbedding(4, value)

    def test_initialization_is_bound_to_an_immutable_explicit_spec(self):
        first = CategoricalEmbedding(5, 3, init_seed=17)
        second = CategoricalEmbedding(5, 3, init_seed=17)
        torch.testing.assert_close(first.module().weight, second.module().weight)
        with self.assertRaises(FrozenInstanceError):
            first.spec.dim = 10

    def test_every_attachment_rechecks_shape_device_dtype_and_finiteness(self):
        embedding = CategoricalEmbedding(5, 3)
        module = resolve_embedding(embedding, 5, 3)
        self.assertIs(module, resolve_embedding(embedding, 5, 3))
        module.to(dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "dtype"):
            resolve_embedding(embedding, 5, 3)

    def test_external_embeddings_are_type_shape_and_finiteness_checked(self):
        with self.assertRaisesRegex(TypeError, "torch.nn.Embedding"):
            resolve_embedding(object(), 5, 3)
        with self.assertRaisesRegex(ValueError, "shape"):
            resolve_embedding(torch.nn.Embedding(4, 3), 5, 3)
        module = torch.nn.Embedding(5, 3)
        module.weight.data[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            resolve_embedding(module, 5, 3)


if __name__ == "__main__":
    unittest.main()
