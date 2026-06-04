# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path


HF_TO_MODELSCOPE_REPO = {
    "nvidia/GR00T-N1.7-3B": "nv-community/GR00T-N1.7-3B",
    "nvidia/Cosmos-Reason2-2B": "nv-community/Cosmos-Reason2-2B",
}


def resolve_model_path(model_name_or_path: str | Path) -> str:
    """Resolve known Hugging Face repo IDs through ModelScope when requested."""
    model_name_or_path = str(model_name_or_path)
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    if os.environ.get("GROOT_MODEL_SOURCE", "").strip().lower() != "modelscope":
        return model_name_or_path

    modelscope_repo = HF_TO_MODELSCOPE_REPO.get(model_name_or_path, model_name_or_path)
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "GROOT_MODEL_SOURCE=modelscope requires the `modelscope` package. "
            "Install it with: uv pip install modelscope"
        ) from exc

    revision = os.environ.get("GROOT_MODELSCOPE_REVISION")
    cache_dir = os.environ.get("MODELSCOPE_CACHE")
    kwargs = {}
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    return snapshot_download(modelscope_repo, **kwargs)
