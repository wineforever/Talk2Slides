import logging
import os
import re
from pathlib import Path
from typing import Optional, Tuple

from huggingface_hub import snapshot_download

from app.core.config import settings

logger = logging.getLogger(__name__)

_DRIVE_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _is_local_reference(model_name: str) -> bool:
    name = str(model_name or "").strip()
    if not name:
        return False
    expanded = os.path.expanduser(name)
    if os.path.exists(expanded):
        return True
    if os.path.isabs(expanded):
        return True
    if name.startswith((".", "~")):
        return True
    return bool(_DRIVE_ABS_RE.match(name))


def _normalize_repo_id(model_name: str) -> str:
    name = str(model_name or "").strip()
    if "/" in name:
        return name
    return f"sentence-transformers/{name}"


def _model_target_dir(model_name: str) -> Path:
    model_root = Path(settings.SENTENCE_TRANSFORMER_LOCAL_MODEL_DIR).expanduser().resolve()
    repo_id = _normalize_repo_id(model_name)
    safe_name = repo_id.replace("\\", "/").strip("/").replace("/", "__")
    return model_root / safe_name


def _is_model_dir_ready(model_dir: Path) -> bool:
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    if not (model_dir / "modules.json").exists():
        return False
    return (model_dir / "config.json").exists() or (model_dir / "config_sentence_transformers.json").exists()


def resolve_local_model_path(model_name: str) -> Tuple[str, bool]:
    """Return (resolved_model_name, is_local_directory)."""
    name = str(model_name or "").strip()
    if _is_local_reference(name):
        return str(Path(name).expanduser().resolve()), True

    target_dir = _model_target_dir(name)
    if _is_model_dir_ready(target_dir):
        return str(target_dir), True
    return name, False


def bootstrap_local_model_if_needed() -> Optional[Path]:
    """Ensure model is available under SENTENCE_TRANSFORMER_LOCAL_MODEL_DIR."""
    model_name = str(settings.SENTENCE_TRANSFORMER_MODEL or "").strip()
    if not model_name:
        return None

    resolved, is_local = resolve_local_model_path(model_name)
    if is_local:
        path = Path(resolved)
        if path.exists():
            logger.info("Sentence-Transformer local model ready: %s", path)
            return path
        return None

    if not bool(settings.SENTENCE_TRANSFORMER_BOOTSTRAP_DOWNLOAD_IF_MISSING):
        return None

    target_dir = _model_target_dir(model_name)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if _is_model_dir_ready(target_dir):
        logger.info("Sentence-Transformer local model already cached: %s", target_dir)
        return target_dir

    endpoint = (str(settings.SENTENCE_TRANSFORMER_HF_ENDPOINT or "").strip() or None)
    etag_timeout = int(max(1, int(settings.SENTENCE_TRANSFORMER_HF_ETAG_TIMEOUT_SEC)))
    download_timeout = int(max(1, int(settings.SENTENCE_TRANSFORMER_HF_DOWNLOAD_TIMEOUT_SEC)))
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(etag_timeout)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(download_timeout)
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint

    repo_id = _normalize_repo_id(model_name)
    revision = str(settings.SENTENCE_TRANSFORMER_MODEL_REVISION or "").strip() or None
    logger.info("Bootstrapping Sentence-Transformer model to local dir: repo=%s, dir=%s", repo_id, target_dir)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=target_dir,
        etag_timeout=float(etag_timeout),
        endpoint=endpoint,
        local_files_only=False,
    )

    if _is_model_dir_ready(target_dir):
        logger.info("Sentence-Transformer model bootstrap completed: %s", target_dir)
        return target_dir

    raise RuntimeError(f"Model download finished but target dir is incomplete: {target_dir}")
