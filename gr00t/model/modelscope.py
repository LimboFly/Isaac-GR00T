# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_MODEL_DIR = REPO_ROOT / "model"

HF_TO_MODELSCOPE_REPO = {
    "nvidia/GR00T-N1.7-3B": "nv-community/GR00T-N1.7-3B",
    "nvidia/Cosmos-Reason2-2B": "nv-community/Cosmos-Reason2-2B",
}

REQUIRED_MODEL_FILES = {
    "nvidia/GR00T-N1.7-3B": (
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
    ),
    "nvidia/Cosmos-Reason2-2B": (
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "preprocessor_config.json",
    ),
}


def _local_model_dir() -> Path:
    return Path(os.environ.get("GROOT_LOCAL_MODEL_DIR", DEFAULT_LOCAL_MODEL_DIR)).expanduser()


def _local_repo_path(model_name_or_path: str) -> Path:
    return _local_model_dir() / model_name_or_path


def _has_model_files(model_name_or_path: str, path: Path) -> bool:
    required_files = REQUIRED_MODEL_FILES.get(model_name_or_path)
    if required_files is not None:
        return all((path / filename).exists() for filename in required_files)

    return (path / "config.json").exists() and any(
        path.glob(pattern) for pattern in ("*.safetensors", "*.bin")
    )


def _uses_modelscope(model_name_or_path: str) -> bool:
    source = os.environ.get("GROOT_MODEL_SOURCE", "").strip().lower()
    if source in {"huggingface", "hf"}:
        return False
    return source == "modelscope" or model_name_or_path in HF_TO_MODELSCOPE_REPO


def _download_from_modelscope(
    modelscope_repo: str,
    local_dir: Path,
    *,
    revision: str | None,
) -> str:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Downloading models from ModelScope requires the `modelscope` package. "
            "Install it with: uv pip install modelscope"
        ) from exc

    kwargs = {"local_dir": str(local_dir)}
    if revision:
        kwargs["revision"] = revision

    return snapshot_download(modelscope_repo, **kwargs)


def resolve_model_path(model_name_or_path: str | Path) -> str:
    """Resolve repo IDs to the project-local model directory when possible.

    Known pretrained model IDs are loaded from ``<repo>/model/<repo_id>`` first.
    If the model is missing locally, it is downloaded from ModelScope into that
    same directory so Docker images can bake the files into ``/workspace/model``.
    """
    model_name_or_path = str(model_name_or_path)
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    local_path = _local_repo_path(model_name_or_path)
    if local_path.is_dir() and _has_model_files(model_name_or_path, local_path):
        return str(local_path)

    if not _uses_modelscope(model_name_or_path):
        return model_name_or_path

    modelscope_repo = HF_TO_MODELSCOPE_REPO.get(model_name_or_path, model_name_or_path)
    revision = os.environ.get("GROOT_MODELSCOPE_REVISION")
    local_path.mkdir(parents=True, exist_ok=True)
    return _download_from_modelscope(modelscope_repo, local_path, revision=revision)
