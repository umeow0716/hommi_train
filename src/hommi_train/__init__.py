"""HoMMI training, evaluation, and deployment utilities."""

from .config import (
    DDIMConfig,
    DatasetConfig,
    DiTModelConfig,
    DiTTrainConfig,
    EvaluationConfig,
    ExportConfig,
    HommiTrainConfig,
    RuntimeConfig,
    hommi_train_config_from_mapping,
)
from .dataset import (
    EpisodeSplit,
    HommiHDF5Dataset,
    HommiHDF5Info,
    inspect_hommi_hdf5,
    split_episode_keys,
)
from .evaluation import EvaluationResult, evaluate_policy, run_evaluation
from .export import (
    PolicyInferenceModule,
    build_inference_module,
    export_policy_pt2,
    export_portable_model_pt2,
    load_portable_payload,
    load_portable_policy,
    run_export,
    save_portable_checkpoint_model,
)
from .normalization import build_hommi_normalizer
from .policy import build_ddim_scheduler, build_dit_policy
from .runner import run_training
from .training import (
    HommiEMAModel,
    Trainer,
    TrainerState,
    build_dataloaders,
    build_lr_scheduler,
    build_optimizer,
    load_training_checkpoint,
    seed_everything,
)

__all__ = [
    "DDIMConfig",
    "DatasetConfig",
    "DiTModelConfig",
    "DiTTrainConfig",
    "EpisodeSplit",
    "EvaluationConfig",
    "EvaluationResult",
    "ExportConfig",
    "HommiEMAModel",
    "HommiHDF5Dataset",
    "HommiHDF5Info",
    "HommiTrainConfig",
    "PolicyInferenceModule",
    "RuntimeConfig",
    "Trainer",
    "TrainerState",
    "build_dataloaders",
    "build_ddim_scheduler",
    "build_dit_policy",
    "build_hommi_normalizer",
    "build_inference_module",
    "build_lr_scheduler",
    "build_optimizer",
    "evaluate_policy",
    "export_policy_pt2",
    "export_portable_model_pt2",
    "hommi_train_config_from_mapping",
    "inspect_hommi_hdf5",
    "load_portable_payload",
    "load_portable_policy",
    "load_training_checkpoint",
    "run_evaluation",
    "run_export",
    "run_training",
    "save_portable_checkpoint_model",
    "seed_everything",
    "split_episode_keys",
]

__version__ = "0.5.0"
