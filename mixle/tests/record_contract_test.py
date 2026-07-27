import numpy as np
import pytest

from mixle.stats import (
    GaussianDistribution,
    GaussianEstimator,
    RecordDataEncoder,
    RecordDistribution,
    RecordEstimator,
    field,
)


def _two_field_model():
    return RecordDistribution(
        {
            "x": GaussianDistribution(0.0, 1.0),
            "y": GaussianDistribution(1.0, 2.0),
        }
    )


def test_record_requires_unique_fields_and_sources():
    with pytest.raises(ValueError, match="record sources must be unique"):
        RecordDistribution(
            {
                field("left", source="x"): GaussianDistribution(0.0, 1.0),
                field("right", source="x"): GaussianDistribution(0.0, 1.0),
            }
        )
    with pytest.raises(ValueError, match="field names must be unique"):
        RecordDistribution(
            (
                field("same", source="x"),
                field("same", source="y"),
            ),
            (
                GaussianDistribution(0.0, 1.0),
                GaussianDistribution(0.0, 1.0),
            ),
        )


def test_record_scoring_and_encoding_require_exact_source_fields():
    model = _two_field_model()
    exact = {"x": 0.0, "y": 1.0}
    assert np.isfinite(model.log_density(exact))
    assert model.log_density({"x": 0.0}) == -np.inf
    assert model.log_density({"x": 0.0, "y": 1.0, "z": 2.0}) == -np.inf
    assert model.log_density((0.0, 1.0)) == -np.inf

    encoder = model.dist_to_encoder()
    encoded = encoder.seq_encode([exact])
    assert model.seq_log_density(encoded).shape == (1,)
    with pytest.raises(ValueError, match="missing=.*y"):
        encoder.seq_encode([{"x": 0.0}])
    with pytest.raises(ValueError, match="extra=.*z"):
        encoder.seq_encode([{"x": 0.0, "y": 1.0, "z": 2.0}])


def test_empty_record_is_a_proper_singleton_mapping_law():
    model = RecordDistribution({})
    assert model.log_density({}) == 0.0
    assert model.log_density({"extra": 1}) == -np.inf
    assert model.sampler(seed=0).sample() == {}
    assert model.sampler(seed=0).sample(size=3) == [{}, {}, {}]
    assert list(model.enumerator()) == [({}, 0.0)]
    encoded = model.dist_to_encoder().seq_encode([{}, {}])
    np.testing.assert_array_equal(model.seq_log_density(encoded), np.zeros(2))


def test_record_encoder_identity_includes_source_layout():
    child = GaussianDistribution(0.0, 1.0).dist_to_encoder()
    from_x = RecordDataEncoder((field("view", source="x"),), (child,))
    from_y = RecordDataEncoder((field("view", source="y"),), (child,))

    assert from_x != from_y
    assert from_x.__mixle_cache_key__ != from_y.__mixle_cache_key__
    assert "source('x')" in str(from_x)


def test_record_state_arity_is_validated_before_mutation():
    estimator = RecordEstimator(
        {
            "x": GaussianEstimator(),
            "y": GaussianEstimator(),
        }
    )
    accumulator = estimator.accumulator_factory().make()
    before_ids = tuple(map(id, accumulator.accumulators))
    before = accumulator.value()

    with pytest.raises(ValueError, match="exactly 2"):
        accumulator.from_value(before[:1])
    assert tuple(map(id, accumulator.accumulators)) == before_ids
    assert accumulator.count == 2
    assert accumulator.value() == before

    with pytest.raises(ValueError, match="exactly 2"):
        accumulator.combine(before[:1])
    with pytest.raises(ValueError, match="exactly 2"):
        estimator.estimate(1.0, before[:1])


def test_record_raw_accumulation_rejects_partial_or_extra_rows():
    estimator = _two_field_model().estimator()
    accumulator = estimator.accumulator_factory().make()
    with pytest.raises(ValueError, match="missing=.*y"):
        accumulator.update({"x": 0.0}, 1.0, None)
    with pytest.raises(ValueError, match="extra=.*z"):
        accumulator.update({"x": 0.0, "y": 1.0, "z": 2.0}, 1.0, None)
