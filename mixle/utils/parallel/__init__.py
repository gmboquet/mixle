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
    register_encoded_data_backend,
)
from mixle.utils.parallel.training_contracts import (  # noqa: E402
    BackendCapabilities,
    CollectiveKind,
    DistributedUpdate,
    ParallelAxis,
    ParallelPlan,
    ParameterLayout,
    PayloadKind,
    StateLayout,
    StepReceipt,
    available_training_backends,
    get_training_backend,
    register_training_backend,
)
from mixle.utils.parallel.training_launchers import LightningFabricLauncher, RayTrainLauncher  # noqa: E402

register_training_backend(
    "torch_native",
    lambda: __import__(
        "mixle.utils.parallel.torch_training", fromlist=["TorchDistributedBackend"]
    ).TorchDistributedBackend(),
    override=True,
)
register_training_backend(
    "megatron",
    lambda: __import__(
        "mixle.utils.parallel.megatron_training", fromlist=["MegatronBridgeBackend"]
    ).MegatronBridgeBackend(),
    override=True,
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
    "register_encoded_data_backend",
    "BackendCapabilities",
    "CollectiveKind",
    "DistributedUpdate",
    "ParallelAxis",
    "ParallelPlan",
    "ParameterLayout",
    "PayloadKind",
    "StateLayout",
    "StepReceipt",
    "available_training_backends",
    "get_training_backend",
    "register_training_backend",
    "LightningFabricLauncher",
    "RayTrainLauncher",
]
