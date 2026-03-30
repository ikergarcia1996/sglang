import json
import logging
import os
import subprocess
from functools import lru_cache

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_overlay_registry() -> dict:
    raw_value = os.getenv("SGLANG_DIFFUSION_MODEL_OVERLAY_REGISTRY", "").strip()
    if not raw_value:
        return {}
    if raw_value.startswith("{"):
        payload = json.loads(raw_value)
    else:
        with open(os.path.expanduser(raw_value), encoding="utf-8") as f:
            payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _is_overlay_diffusion_model(model_path: str) -> bool:
    registry = _load_overlay_registry()
    if model_path in registry:
        return True
    if not os.path.exists(model_path):
        return False
    base_name = os.path.basename(os.path.normpath(model_path))
    return any(base_name == key.rsplit("/", 1)[-1] for key in registry)


def _is_diffusers_model_dir(model_dir: str) -> bool:
    """Check if a local directory contains a valid diffusers model_index.json."""
    config_path = os.path.join(model_dir, "model_index.json")
    if not os.path.exists(config_path):
        return False

    with open(config_path) as f:
        config = json.load(f)

    return "_diffusers_version" in config


def get_is_diffusion_model(model_path: str) -> bool:
    """Detect whether model_path points to a diffusion model.

    For local directories, checks the filesystem directly.
    For HF/ModelScope model IDs, attempts to fetch only model_index.json.
    Returns False on any failure (network error, 404, offline mode, etc.)
    so that the caller falls through to the standard LLM server path.
    """
    from sglang.multimodal_gen.registry import is_known_non_diffusers_multimodal_model

    if _is_overlay_diffusion_model(model_path):
        # short-circuit, if applicable for the overlay mechanism (diffusion-only)
        return True

    if os.path.isdir(model_path):
        if _is_diffusers_model_dir(model_path):
            return True
        return is_known_non_diffusers_multimodal_model(model_path)

    if is_known_non_diffusers_multimodal_model(model_path):
        return True

    try:
        if envs.SGLANG_USE_MODELSCOPE.get():
            from modelscope import model_file_download

            file_path = model_file_download(
                model_id=model_path, file_path="model_index.json"
            )
        else:
            from huggingface_hub import hf_hub_download

            file_path = hf_hub_download(repo_id=model_path, filename="model_index.json")

        return _is_diffusers_model_dir(os.path.dirname(file_path))
    except Exception as e:
        logger.debug("Failed to auto-detect diffusion model for %s: %s", model_path, e)
        return False


def get_model_path(extra_argv):
    # Find the model_path argument
    model_path = None
    for i, arg in enumerate(extra_argv):
        if arg == "--model-path":
            if i + 1 < len(extra_argv):
                model_path = extra_argv[i + 1]
                break
        elif arg.startswith("--model-path="):
            model_path = arg.split("=", 1)[1]
            break

    if model_path is None:
        # Fallback for --help or other cases where model-path is not provided
        if any(h in extra_argv for h in ["-h", "--help"]):
            raise Exception(
                "Usage: sglang serve --model-path <model-name-or-path> [additional-arguments]\n\n"
                "This command can launch either a standard language model server or a diffusion model server.\n"
                "The server type is determined by the --model-path.\n"
            )
        else:
            raise Exception(
                "Error: --model-path is required. "
                "Please provide the path to the model."
            )
    return model_path


@lru_cache(maxsize=1)
def get_git_commit_hash() -> str:
    try:
        commit_hash = os.environ.get("SGLANG_GIT_COMMIT")
        if not commit_hash:
            commit_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .strip()
                .decode("utf-8")
            )
        _CACHED_COMMIT_HASH = commit_hash
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        _CACHED_COMMIT_HASH = "N/A"
        return "N/A"
