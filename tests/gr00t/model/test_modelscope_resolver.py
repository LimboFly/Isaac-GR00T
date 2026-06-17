# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from gr00t.model import modelscope as modelscope_resolver


def test_resolve_model_path_prefers_project_local_model_dir(tmp_path, monkeypatch):
    local_model_root = tmp_path / "model"
    local_model = local_model_root / "nvidia" / "GR00T-N1.7-3B"
    local_model.mkdir(parents=True)
    for filename in modelscope_resolver.REQUIRED_MODEL_FILES["nvidia/GR00T-N1.7-3B"]:
        (local_model / filename).write_text("{}")
    monkeypatch.setenv("GROOT_LOCAL_MODEL_DIR", str(local_model_root))

    assert modelscope_resolver.resolve_model_path("nvidia/GR00T-N1.7-3B") == str(local_model)


def test_resolve_model_path_downloads_known_models_to_project_model_dir(tmp_path, monkeypatch):
    local_model_root = tmp_path / "model"
    monkeypatch.setenv("GROOT_LOCAL_MODEL_DIR", str(local_model_root))
    monkeypatch.delenv("GROOT_MODELSCOPE_REVISION", raising=False)
    calls = []

    def fake_download(modelscope_repo: str, local_dir: Path, *, revision: str | None) -> str:
        calls.append(
            {
                "modelscope_repo": modelscope_repo,
                "local_dir": local_dir,
                "revision": revision,
            }
        )
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")
        return str(local_dir)

    monkeypatch.setattr(modelscope_resolver, "_download_from_modelscope", fake_download)

    resolved = modelscope_resolver.resolve_model_path("nvidia/Cosmos-Reason2-2B")

    expected_local_dir = local_model_root / "nvidia" / "Cosmos-Reason2-2B"
    assert resolved == str(expected_local_dir)
    assert calls == [
        {
            "modelscope_repo": "nv-community/Cosmos-Reason2-2B",
            "local_dir": expected_local_dir,
            "revision": None,
        }
    ]


def test_resolve_model_path_leaves_unknown_repo_ids_for_huggingface_by_default(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GROOT_LOCAL_MODEL_DIR", str(tmp_path / "model"))
    monkeypatch.delenv("GROOT_MODEL_SOURCE", raising=False)

    assert modelscope_resolver.resolve_model_path("some-org/custom-model") == (
        "some-org/custom-model"
    )
