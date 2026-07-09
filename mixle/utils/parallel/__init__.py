"""Distributed-estimation planning and model-parallel entry points.

Importing this namespace registers the model-parallel encoded-data backend and
exposes resource planning, sharding, and estimator wrappers used by
multi-process or multi-device estimation workflows.
"""

# Load the model-parallel backend so it registers with the encoded-data registry on package import.
from mixle.utils.parallel import model_parallel as _model_parallel  # noqa: E402,F401
from mixle.utils.parallel.model_parallel import (  # noqa: E402
    ModelParallelEncodedData,
    ModelParallelEstimator,
    auto_parallel_estimator,
    model_parallel_fold,
)
from mixle.utils.parallel.planner import (  # noqa: E402
    Resources,
    encoded_data,
    is_encoded_data_handle,
    model_sharding_plan,
    plan,
)

__all__ = [
    "ModelParallelEstimator",
    "ModelParallelEncodedData",
    "model_parallel_fold",
    "auto_parallel_estimator",
    "Resources",
    "encoded_data",
    "is_encoded_data_handle",
    "model_sharding_plan",
    "plan",
]
