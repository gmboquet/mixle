"""Distributed-estimation backends (multiprocessing, MPI, torchrun)."""

# Load the model-parallel backend so it registers with the encoded-data registry on package import.
from pysp.utils.parallel import model_parallel as _model_parallel  # noqa: E402,F401
from pysp.utils.parallel.model_parallel import (  # noqa: E402
    ModelParallelEncodedData,
    ModelParallelEstimator,
    model_parallel_fold,
)

__all__ = ["ModelParallelEstimator", "ModelParallelEncodedData", "model_parallel_fold"]
