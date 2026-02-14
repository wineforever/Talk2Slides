from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, Response
from fastapi.responses import FileResponse
import configparser
import os
import uuid
import shutil
import logging
import time
import threading
from pathlib import Path
import re
from datetime import datetime
from typing import Optional, Dict, Any

from app.core.config import settings
from app.services.task_manager import TaskManager, TaskStatus
from app.services.ppt_service import PPTService
from app.services.srt_service import SRTService
from app.services.alignment_service import AlignmentService
from app.services.video_service import VideoService

logger = logging.getLogger(__name__)

router = APIRouter()
task_manager = TaskManager()

_INVALID_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1f]'
_INI_WRITE_LOCK = threading.Lock()
_RUNTIME_TO_ENV_KEY = {
    "similarity_threshold": "DEFAULT_SIMILARITY_THRESHOLD",
    "min_display_duration": "DEFAULT_MIN_DISPLAY_DURATION",
    "output_resolution": "DEFAULT_OUTPUT_RESOLUTION",
    "srt_merge_gap_sec": "SRT_MERGE_GAP_SEC",
    "srt_min_duration_sec": "SRT_MIN_DURATION_SEC",
    "align_max_backtrack": "ALIGN_MAX_BACKTRACK",
    "align_max_forward_jump": "ALIGN_MAX_FORWARD_JUMP",
    "align_switch_penalty": "ALIGN_SWITCH_PENALTY",
    "align_forward_jump_penalty": "ALIGN_FORWARD_JUMP_PENALTY",
    "align_switch_delay_ms": "ALIGN_SWITCH_DELAY_MS",
    "align_enforce_no_revisit": "ALIGN_ENFORCE_NO_REVISIT",
    "align_min_slide_usage_ratio": "ALIGN_MIN_SLIDE_USAGE_RATIO",
    "align_enforce_sequential": "ALIGN_ENFORCE_SEQUENTIAL",
}


def _ini_path() -> Path:
    return settings.BASE_DIR.parent / "talk2slides.ini"


def _parse_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_int(value: object, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _normalize_output_resolution(value: object) -> str:
    text = str(value or "").strip().lower()
    if re.fullmatch(r"\d{2,5}x\d{2,5}", text):
        return text
    return str(settings.DEFAULT_OUTPUT_RESOLUTION)


def _normalize_generation_params(
    similarity_threshold: object,
    min_display_duration: object,
    output_resolution: object,
    srt_merge_gap_sec: object,
    srt_min_duration_sec: object,
    align_max_backtrack: object,
    align_max_forward_jump: object,
    align_switch_penalty: object,
    align_forward_jump_penalty: object,
    align_switch_delay_ms: object,
    align_enforce_no_revisit: object,
    align_min_slide_usage_ratio: object,
    align_enforce_sequential: object,
) -> Dict[str, Any]:
    return {
        "similarity_threshold": min(max(_parse_float(similarity_threshold, settings.DEFAULT_SIMILARITY_THRESHOLD), 0.1), 0.9),
        "min_display_duration": max(_parse_float(min_display_duration, settings.DEFAULT_MIN_DISPLAY_DURATION), 1.0),
        "output_resolution": _normalize_output_resolution(output_resolution),
        "srt_merge_gap_sec": min(max(_parse_float(srt_merge_gap_sec, settings.SRT_MERGE_GAP_SEC), 0.0), 3.0),
        "srt_min_duration_sec": min(max(_parse_float(srt_min_duration_sec, settings.SRT_MIN_DURATION_SEC), 0.1), 5.0),
        "align_max_backtrack": int(min(max(_parse_int(align_max_backtrack, settings.ALIGN_MAX_BACKTRACK), 0), 5)),
        "align_max_forward_jump": int(min(max(_parse_int(align_max_forward_jump, settings.ALIGN_MAX_FORWARD_JUMP), 0), 10)),
        "align_switch_penalty": min(max(_parse_float(align_switch_penalty, settings.ALIGN_SWITCH_PENALTY), 0.0), 1.0),
        "align_forward_jump_penalty": min(max(_parse_float(align_forward_jump_penalty, settings.ALIGN_FORWARD_JUMP_PENALTY), 0.0), 1.0),
        "align_switch_delay_ms": int(min(max(_parse_int(align_switch_delay_ms, settings.ALIGN_SWITCH_DELAY_MS), 0), 2000)),
        "align_enforce_no_revisit": _parse_bool(align_enforce_no_revisit),
        "align_min_slide_usage_ratio": float(min(max(_parse_float(align_min_slide_usage_ratio, settings.ALIGN_MIN_SLIDE_USAGE_RATIO), 0.5), 1.0)),
        "align_enforce_sequential": _parse_bool(align_enforce_sequential),
    }


def _format_ini_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def _read_ini_env_values(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}

    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")
    if not cfg.has_section("env"):
        return {}

    return {str(k).upper(): str(v).strip() for k, v in cfg.items("env")}


def _upsert_ini_env_values(path: Path, env_values: Dict[str, str]) -> None:
    if not env_values:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)

    section_pattern = re.compile(r"^\s*\[([^\]]+)\]\s*$")
    env_start = None
    env_end = len(lines)

    for idx, line in enumerate(lines):
        section_match = section_pattern.match(line.strip())
        if not section_match:
            continue
        section_name = section_match.group(1).strip().lower()
        if env_start is None and section_name == "env":
            env_start = idx
            continue
        if env_start is not None:
            env_end = idx
            break

    if env_start is None:
        if lines and not lines[-1].endswith(("\n", "\r\n")):
            lines[-1] = lines[-1] + newline
        if lines and lines[-1].strip():
            lines.append(newline)
        lines.append(f"[env]{newline}")
        for key, value in env_values.items():
            lines.append(f"{key} = {value}{newline}")
        path.write_text("".join(lines), encoding="utf-8")
        return

    key_line_pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
    existing_key_to_line = {}
    for idx in range(env_start + 1, env_end):
        key_match = key_line_pattern.match(lines[idx])
        if not key_match:
            continue
        existing_key_to_line[key_match.group(1).upper()] = idx

    insert_at = env_end
    for key, value in env_values.items():
        target_idx = existing_key_to_line.get(key.upper())
        if target_idx is None:
            lines.insert(insert_at, f"{key} = {value}{newline}")
            insert_at += 1
            continue

        old_line = lines[target_idx]
        line_newline = "\r\n" if old_line.endswith("\r\n") else ("\n" if old_line.endswith("\n") else newline)
        old_key_match = key_line_pattern.match(old_line)
        display_key = old_key_match.group(1) if old_key_match else key
        lines[target_idx] = f"{display_key} = {value}{line_newline}"

    path.write_text("".join(lines), encoding="utf-8")


def _persist_generation_defaults_to_ini(params: Dict[str, Any]) -> None:
    env_values = {}
    for runtime_key, env_key in _RUNTIME_TO_ENV_KEY.items():
        if runtime_key in params:
            env_values[env_key] = _format_ini_value(params[runtime_key])

    if not env_values:
        return

    with _INI_WRITE_LOCK:
        _upsert_ini_env_values(_ini_path(), env_values)


def _load_generation_defaults() -> Dict[str, Any]:
    with _INI_WRITE_LOCK:
        env_values = _read_ini_env_values(_ini_path())

    return _normalize_generation_params(
        similarity_threshold=env_values.get("DEFAULT_SIMILARITY_THRESHOLD", settings.DEFAULT_SIMILARITY_THRESHOLD),
        min_display_duration=env_values.get("DEFAULT_MIN_DISPLAY_DURATION", settings.DEFAULT_MIN_DISPLAY_DURATION),
        output_resolution=env_values.get("DEFAULT_OUTPUT_RESOLUTION", settings.DEFAULT_OUTPUT_RESOLUTION),
        srt_merge_gap_sec=env_values.get("SRT_MERGE_GAP_SEC", settings.SRT_MERGE_GAP_SEC),
        srt_min_duration_sec=env_values.get("SRT_MIN_DURATION_SEC", settings.SRT_MIN_DURATION_SEC),
        align_max_backtrack=env_values.get("ALIGN_MAX_BACKTRACK", settings.ALIGN_MAX_BACKTRACK),
        align_max_forward_jump=env_values.get("ALIGN_MAX_FORWARD_JUMP", settings.ALIGN_MAX_FORWARD_JUMP),
        align_switch_penalty=env_values.get("ALIGN_SWITCH_PENALTY", settings.ALIGN_SWITCH_PENALTY),
        align_forward_jump_penalty=env_values.get("ALIGN_FORWARD_JUMP_PENALTY", settings.ALIGN_FORWARD_JUMP_PENALTY),
        align_switch_delay_ms=env_values.get("ALIGN_SWITCH_DELAY_MS", settings.ALIGN_SWITCH_DELAY_MS),
        align_enforce_no_revisit=env_values.get("ALIGN_ENFORCE_NO_REVISIT", settings.ALIGN_ENFORCE_NO_REVISIT),
        align_min_slide_usage_ratio=env_values.get("ALIGN_MIN_SLIDE_USAGE_RATIO", settings.ALIGN_MIN_SLIDE_USAGE_RATIO),
        align_enforce_sequential=env_values.get("ALIGN_ENFORCE_SEQUENTIAL", settings.ALIGN_ENFORCE_SEQUENTIAL),
    )


def _sanitize_filename_stem(filename: str) -> str:
    raw_name = (filename or "").strip()
    raw_name = raw_name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if "." in raw_name:
        stem = raw_name.rsplit(".", 1)[0].strip()
    else:
        stem = raw_name.strip()
    if not stem:
        stem = "presentation"
    stem = re.sub(_INVALID_FILENAME_CHARS, "_", stem)
    stem = re.sub(r"\s+", "_", stem).strip(" ._")
    return stem or "presentation"


def _next_video_version(base_stem: str) -> int:
    pattern = re.compile(rf"^{re.escape(base_stem)}_v(\d+)\.mp4$", re.IGNORECASE)
    max_version = 0
    for video_file in settings.OUTPUT_DIR.rglob("*.mp4"):
        match = pattern.match(video_file.name)
        if not match:
            continue
        version = int(match.group(1))
        if version > max_version:
            max_version = version
    return max_version + 1


def _format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _parse_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_label_source(text: str) -> str:
    compact = re.sub(r"\s+", " ", str(text or "").strip())
    compact = re.sub(r"https?://\S+", "", compact)
    compact = re.sub(r"[`*_#>\[\]{}]", " ", compact)
    compact = re.sub(r"\s{2,}", " ", compact)
    return compact.strip(" -_，,。；;：:、")


def _pick_fluent_phrase(text: str) -> str:
    normalized = _normalize_label_source(text)
    if not normalized:
        return ""

    major_parts = [part.strip() for part in re.split(r"[\r\n]+|[。！？!?；;]", normalized) if part.strip()]
    first_part = major_parts[0] if major_parts else normalized
    clauses = [part.strip() for part in re.split(r"[，,:：|/]", first_part) if part.strip()]
    phrase = clauses[0] if clauses else first_part
    phrase = re.sub(r"^[\-\*\d一二三四五六七八九十]+[\.、:：\)\]）】\s]*", "", phrase)
    phrase = phrase.strip(" -_，,。；;：:、")
    if len(phrase) < 2:
        phrase = normalized
    return phrase


def _compact_text(text: str, max_len: int = 28) -> str:
    compact = _pick_fluent_phrase(text)
    if not compact:
        return ""
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", compact))
    hard_limit = min(max_len, 14 if has_cjk else 24)
    if len(compact) <= hard_limit:
        return compact
    return compact[:hard_limit].rstrip(" -_，,。；;：:")


def _build_slide_overview(slides: list) -> list:
    overview = []
    for idx, slide in enumerate(slides or []):
        slide_index = int(slide.get("index", idx))
        title = (slide.get("title") or "").strip()
        notes = (slide.get("notes") or "").strip()
        content = (slide.get("content") or "").strip()
        full_text = (slide.get("full_text") or "").strip()
        summary = ""
        for candidate_text in [title, notes, content, full_text]:
            summary = _compact_text(candidate_text, max_len=16)
            if summary:
                break
        if not summary:
            summary = f"第{slide_index + 1}页"
        overview.append(
            {
                "slide_index": slide_index,
                "summary": summary,
            }
        )
    return overview


def _build_timeline_overview(timeline: list, slide_overview: list) -> list:
    summary_map = {
        int(item.get("slide_index", -1)): str(item.get("summary", "")).strip()
        for item in (slide_overview or [])
    }
    sorted_segments = sorted(
        [
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", seg.get("start", 0.0))),
                "slide_index": int(seg.get("slide_index", -1)),
            }
            for seg in (timeline or [])
        ],
        key=lambda x: x["start"],
    )

    merged = []
    for seg in sorted_segments:
        start = seg["start"]
        end = seg["end"]
        slide_index = seg["slide_index"]
        if end <= start:
            continue

        if (
            merged
            and merged[-1]["slide_index"] == slide_index
            and start <= merged[-1]["end"] + 1e-3
        ):
            merged[-1]["end"] = max(merged[-1]["end"], end)
            merged[-1]["duration"] = max(0.0, merged[-1]["end"] - merged[-1]["start"])
            continue

        default_label = f"第{slide_index + 1}页" if slide_index >= 0 else "未知章节"
        label = summary_map.get(slide_index) or default_label
        merged.append(
            {
                "start": start,
                "end": end,
                "duration": max(0.0, end - start),
                "slide_index": slide_index,
                "label": label,
            }
        )

    return merged


@router.options("/upload")
async def options_upload():
    return Response(status_code=200)


@router.get("/sound")
async def get_notification_sound():
    sound_path = settings.BASE_DIR.parent / "Sound.mp3"
    if not sound_path.exists():
        raise HTTPException(status_code=404, detail="Sound.mp3 not found in project root")
    return FileResponse(str(sound_path), media_type="audio/mpeg", filename="Sound.mp3")


@router.get("/defaults")
async def get_generation_defaults():
    try:
        return _load_generation_defaults()
    except Exception as exc:
        logger.warning("Failed to load defaults from ini, falling back to current settings: %s", exc)
        return _normalize_generation_params(
            similarity_threshold=settings.DEFAULT_SIMILARITY_THRESHOLD,
            min_display_duration=settings.DEFAULT_MIN_DISPLAY_DURATION,
            output_resolution=settings.DEFAULT_OUTPUT_RESOLUTION,
            srt_merge_gap_sec=settings.SRT_MERGE_GAP_SEC,
            srt_min_duration_sec=settings.SRT_MIN_DURATION_SEC,
            align_max_backtrack=settings.ALIGN_MAX_BACKTRACK,
            align_max_forward_jump=settings.ALIGN_MAX_FORWARD_JUMP,
            align_switch_penalty=settings.ALIGN_SWITCH_PENALTY,
            align_forward_jump_penalty=settings.ALIGN_FORWARD_JUMP_PENALTY,
            align_switch_delay_ms=settings.ALIGN_SWITCH_DELAY_MS,
            align_enforce_no_revisit=settings.ALIGN_ENFORCE_NO_REVISIT,
            align_min_slide_usage_ratio=settings.ALIGN_MIN_SLIDE_USAGE_RATIO,
            align_enforce_sequential=settings.ALIGN_ENFORCE_SEQUENTIAL,
        )


@router.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    pptx: UploadFile = File(...),
    mp3: UploadFile = File(...),
    srt: UploadFile = File(...),
    similarity_threshold: float = Form(settings.DEFAULT_SIMILARITY_THRESHOLD),
    min_display_duration: float = Form(settings.DEFAULT_MIN_DISPLAY_DURATION),
    output_resolution: str = Form(settings.DEFAULT_OUTPUT_RESOLUTION),
    srt_merge_gap_sec: float = Form(settings.SRT_MERGE_GAP_SEC),
    srt_min_duration_sec: float = Form(settings.SRT_MIN_DURATION_SEC),
    align_max_backtrack: int = Form(settings.ALIGN_MAX_BACKTRACK),
    align_max_forward_jump: int = Form(settings.ALIGN_MAX_FORWARD_JUMP),
    align_switch_penalty: float = Form(settings.ALIGN_SWITCH_PENALTY),
    align_forward_jump_penalty: float = Form(settings.ALIGN_FORWARD_JUMP_PENALTY),
    align_switch_delay_ms: int = Form(settings.ALIGN_SWITCH_DELAY_MS),
    align_enforce_no_revisit: bool = Form(settings.ALIGN_ENFORCE_NO_REVISIT),
    align_min_slide_usage_ratio: float = Form(settings.ALIGN_MIN_SLIDE_USAGE_RATIO),
    align_enforce_sequential: bool = Form(settings.ALIGN_ENFORCE_SEQUENTIAL),
):
    if not (pptx.filename or "").lower().endswith(".pptx"):
        raise HTTPException(status_code=400, detail="PPTX file is required")

    allowed_audio_extensions = [".mp3", ".wav", ".wave"]
    if not any((mp3.filename or "").lower().endswith(ext) for ext in allowed_audio_extensions):
        raise HTTPException(status_code=400, detail="Audio file must be MP3/WAV")

    if not (srt.filename or "").lower().endswith(".srt"):
        raise HTTPException(status_code=400, detail="SRT file is required")

    normalized_params = _normalize_generation_params(
        similarity_threshold=similarity_threshold,
        min_display_duration=min_display_duration,
        output_resolution=output_resolution,
        srt_merge_gap_sec=srt_merge_gap_sec,
        srt_min_duration_sec=srt_min_duration_sec,
        align_max_backtrack=align_max_backtrack,
        align_max_forward_jump=align_max_forward_jump,
        align_switch_penalty=align_switch_penalty,
        align_forward_jump_penalty=align_forward_jump_penalty,
        align_switch_delay_ms=align_switch_delay_ms,
        align_enforce_no_revisit=align_enforce_no_revisit,
        align_min_slide_usage_ratio=align_min_slide_usage_ratio,
        align_enforce_sequential=align_enforce_sequential,
    )
    similarity_threshold = float(normalized_params["similarity_threshold"])
    min_display_duration = float(normalized_params["min_display_duration"])
    output_resolution = str(normalized_params["output_resolution"])
    srt_merge_gap_sec = float(normalized_params["srt_merge_gap_sec"])
    srt_min_duration_sec = float(normalized_params["srt_min_duration_sec"])
    align_max_backtrack = int(normalized_params["align_max_backtrack"])
    align_max_forward_jump = int(normalized_params["align_max_forward_jump"])
    align_switch_penalty = float(normalized_params["align_switch_penalty"])
    align_forward_jump_penalty = float(normalized_params["align_forward_jump_penalty"])
    align_switch_delay_ms = int(normalized_params["align_switch_delay_ms"])
    align_enforce_no_revisit = bool(normalized_params["align_enforce_no_revisit"])
    align_min_slide_usage_ratio = float(normalized_params["align_min_slide_usage_ratio"])
    align_enforce_sequential = bool(normalized_params["align_enforce_sequential"])

    task_id = str(uuid.uuid4())
    task_dir = settings.TEMP_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    pptx_original_name = Path(pptx.filename).name
    pptx_path = task_dir / "presentation.pptx"

    audio_ext = Path(mp3.filename).suffix.lower()
    if audio_ext not in allowed_audio_extensions:
        audio_ext = ".mp3"
    mp3_path = task_dir / f"audio{audio_ext}"

    srt_path = task_dir / "subtitles.srt"

    try:
        with open(pptx_path, "wb") as f:
            f.write(await pptx.read())

        with open(mp3_path, "wb") as f:
            f.write(await mp3.read())

        with open(srt_path, "wb") as f:
            f.write(await srt.read())

        task_manager.create_task(
            task_id=task_id,
            pptx_path=str(pptx_path),
            mp3_path=str(mp3_path),
            srt_path=str(srt_path),
            similarity_threshold=similarity_threshold,
            min_display_duration=min_display_duration,
            output_resolution=output_resolution,
        )

        background_tasks.add_task(
            process_video_generation,
            task_id,
            str(pptx_path),
            str(mp3_path),
            str(srt_path),
            pptx_original_name,
            similarity_threshold,
            min_display_duration,
            output_resolution,
            srt_merge_gap_sec,
            srt_min_duration_sec,
            align_max_backtrack,
            align_max_forward_jump,
            align_switch_penalty,
            align_forward_jump_penalty,
            align_switch_delay_ms,
            align_enforce_no_revisit,
            align_min_slide_usage_ratio,
            align_enforce_sequential,
        )

        try:
            _persist_generation_defaults_to_ini(normalized_params)
        except Exception as exc:
            logger.warning("Failed to persist current params to talk2slides.ini: %s", exc)

        return {
            "task_id": task_id,
            "status": "pending",
            "message": "Upload succeeded, task started",
        }
    except Exception as exc:
        shutil.rmtree(task_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to store uploaded files: {exc}")


@router.get("/task/{task_id}/status")
async def get_task_status(task_id: str):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    elapsed_seconds: Optional[float] = None
    eta_seconds: Optional[float] = None
    eta_formatted: Optional[str] = None

    if task.created_at:
        elapsed_seconds = max(0.0, (datetime.now() - task.created_at).total_seconds())
        if task.state == TaskStatus.PROCESSING and task.progress > 0:
            remaining = max(0.0, 100.0 - float(task.progress))
            eta_seconds = elapsed_seconds * (remaining / float(task.progress))
            eta_formatted = _format_seconds(eta_seconds)

    return {
        "task_id": task_id,
        "state": task.state,
        "progress": task.progress,
        "message": task.message,
        "error": task.error,
        "elapsed_seconds": elapsed_seconds,
        "eta_seconds": eta_seconds,
        "eta_formatted": eta_formatted,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "updated_at": task.updated_at.isoformat() if task.updated_at else None,
    }


@router.get("/task/{task_id}/result")
async def get_task_result(task_id: str):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.state != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Task not completed")

    video_path = task.result.get("video_path") if task.result else None
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    filename = os.path.basename(video_path)
    return FileResponse(video_path, media_type="video/mp4", filename=filename)


@router.get("/task/{task_id}/preview")
async def get_task_preview(task_id: str):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.state != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Task not completed")

    video_path = task.result.get("video_path") if task.result else None
    filename = os.path.basename(video_path) if video_path else "generated_video.mp4"

    return {
        "task_id": task_id,
        "timeline": task.result.get("timeline", []) if task.result else [],
        "timeline_overview": task.result.get("timeline_overview", []) if task.result else [],
        "slide_overview": task.result.get("slide_overview", []) if task.result else [],
        "alignment_report": task.result.get("alignment_report", {}) if task.result else {},
        "video_url": f"/api/task/{task_id}/result",
        "filename": filename,
    }


@router.delete("/task/{task_id}")
async def delete_task(task_id: str):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    task_dir = settings.TEMP_DIR / task_id
    output_dir = settings.OUTPUT_DIR / task_id

    try:
        shutil.rmtree(task_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        task_manager.delete_task(task_id)
        return {"message": "Task deleted"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}")


async def process_video_generation(
    task_id: str,
    pptx_path: str,
    mp3_path: str,
    srt_path: str,
    pptx_original_name: str,
    similarity_threshold: float,
    min_display_duration: float,
    output_resolution: str,
    srt_merge_gap_sec: float,
    srt_min_duration_sec: float,
    align_max_backtrack: int,
    align_max_forward_jump: int,
    align_switch_penalty: float,
    align_forward_jump_penalty: float,
    align_switch_delay_ms: int,
    align_enforce_no_revisit: bool,
    align_min_slide_usage_ratio: float,
    align_enforce_sequential: bool,
):
    normalized_params = _normalize_generation_params(
        similarity_threshold=similarity_threshold,
        min_display_duration=min_display_duration,
        output_resolution=output_resolution,
        srt_merge_gap_sec=srt_merge_gap_sec,
        srt_min_duration_sec=srt_min_duration_sec,
        align_max_backtrack=align_max_backtrack,
        align_max_forward_jump=align_max_forward_jump,
        align_switch_penalty=align_switch_penalty,
        align_forward_jump_penalty=align_forward_jump_penalty,
        align_switch_delay_ms=align_switch_delay_ms,
        align_enforce_no_revisit=align_enforce_no_revisit,
        align_min_slide_usage_ratio=align_min_slide_usage_ratio,
        align_enforce_sequential=align_enforce_sequential,
    )

    similarity_threshold = float(normalized_params["similarity_threshold"])
    min_display_duration = float(normalized_params["min_display_duration"])
    output_resolution = str(normalized_params["output_resolution"])
    srt_merge_gap_sec = float(normalized_params["srt_merge_gap_sec"])
    srt_min_duration_sec = float(normalized_params["srt_min_duration_sec"])
    align_max_backtrack = int(normalized_params["align_max_backtrack"])
    align_max_forward_jump = int(normalized_params["align_max_forward_jump"])
    align_switch_penalty = float(normalized_params["align_switch_penalty"])
    align_forward_jump_penalty = float(normalized_params["align_forward_jump_penalty"])
    align_switch_delay_ms = int(normalized_params["align_switch_delay_ms"])
    align_enforce_no_revisit = bool(normalized_params["align_enforce_no_revisit"])
    align_min_slide_usage_ratio = float(normalized_params["align_min_slide_usage_ratio"])
    align_enforce_sequential = bool(normalized_params["align_enforce_sequential"])

    logger.info(
        "Video generation parameters: "
        f"similarity_threshold={similarity_threshold}, "
        f"min_display_duration={min_display_duration}, "
        f"output_resolution={output_resolution}, "
        f"srt_merge_gap_sec={srt_merge_gap_sec}, "
        f"srt_min_duration_sec={srt_min_duration_sec}, "
        f"align_max_backtrack={align_max_backtrack}, "
        f"align_max_forward_jump={align_max_forward_jump}, "
        f"align_switch_penalty={align_switch_penalty}, "
        f"align_forward_jump_penalty={align_forward_jump_penalty}, "
        f"align_switch_delay_ms={align_switch_delay_ms}, "
        f"align_enforce_no_revisit={align_enforce_no_revisit}, "
        f"align_min_slide_usage_ratio={align_min_slide_usage_ratio}, "
        f"align_enforce_sequential={align_enforce_sequential}"
    )

    processing_stage = "initialization"

    try:
        task_manager.update_task(task_id, state=TaskStatus.PROCESSING, progress=1, message="Initializing task...")

        processing_stage = "ppt_parsing"
        task_manager.update_task(task_id, progress=10, message="Parsing PPT structure...")
        ppt_service = PPTService()
        slides = ppt_service.extract_slides(pptx_path)
        if not slides:
            raise ValueError("PPT parsing returned no slides")
        task_manager.update_task(task_id, progress=18, message=f"PPT parsed, detected {len(slides)} slides")

        processing_stage = "srt_parsing"
        task_manager.update_task(task_id, progress=22, message="Parsing SRT subtitles...")
        srt_service = SRTService()
        subtitles = srt_service.parse_srt(srt_path)
        if not subtitles:
            raise ValueError("SRT parsing returned no subtitles")
        task_manager.update_task(task_id, progress=28, message=f"SRT parsed, detected {len(subtitles)} subtitle blocks")

        task_manager.update_task(task_id, progress=32, message="Preprocessing subtitles (merge/clean)...")
        segments = srt_service.preprocess_subtitles(
            subtitles=subtitles,
            merge_gap_sec=srt_merge_gap_sec,
            min_chars=settings.SRT_MIN_CHARS,
            min_duration_sec=srt_min_duration_sec,
            max_chars=settings.SRT_MAX_CHARS,
            filler_patterns=settings.SRT_FILLER_PATTERNS,
        )
        if not segments:
            raise ValueError("Subtitle preprocessing returned no segments")
        task_manager.update_task(task_id, progress=38, message=f"Subtitle preprocessing done, {len(segments)} semantic segments kept")

        processing_stage = "alignment"
        task_manager.update_task(task_id, progress=42, message="Running SRT <-> PPT alignment...")
        alignment_service = AlignmentService()
        timeline = alignment_service.align_slides_with_subtitles(
            slides=slides,
            subtitles=segments,
            similarity_threshold=similarity_threshold,
            min_display_duration=min_display_duration,
            align_max_backtrack=align_max_backtrack,
            align_max_forward_jump=align_max_forward_jump,
            align_switch_penalty=align_switch_penalty,
            align_forward_jump_penalty=align_forward_jump_penalty,
            align_switch_delay_ms=align_switch_delay_ms,
            align_enforce_no_revisit=align_enforce_no_revisit,
            align_min_slide_usage_ratio=align_min_slide_usage_ratio,
            enforce_sequential=align_enforce_sequential,
        )
        alignment_report = alignment_service.get_last_alignment_report()
        grade = alignment_report.get("summary", {}).get("grade", "N/A") if isinstance(alignment_report, dict) else "N/A"

        if not timeline:
            raise ValueError("Alignment failed: empty timeline")
        slide_overview = _build_slide_overview(slides)
        timeline_overview = _build_timeline_overview(timeline, slide_overview)
        task_manager.update_task(task_id, progress=58, message=f"Alignment done, timeline segments: {len(timeline)}, grade: {grade}")

        processing_stage = "ppt_to_images"
        task_manager.update_task(task_id, progress=62, message="Exporting PPT slides to images...")
        image_dir = settings.TEMP_DIR / task_id / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_paths = ppt_service.export_slides_to_images(
            pptx_path=pptx_path,
            output_dir=str(image_dir),
            resolution=output_resolution,
        )
        if not image_paths:
            raise ValueError("PPT image export failed: no images")
        task_manager.update_task(task_id, progress=76, message=f"Slide images exported: {len(image_paths)}")

        processing_stage = "video_synthesis"
        task_manager.update_task(task_id, progress=82, message="Preparing video synthesis...")
        video_service = VideoService()

        ppt_stem = _sanitize_filename_stem(pptx_original_name)
        version = _next_video_version(ppt_stem)
        filename = f"{ppt_stem}_v{version}.mp4"
        output_video_path = settings.OUTPUT_DIR / task_id / filename
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        task_manager.update_task(task_id, progress=88, message=f"Start synthesizing video: {filename}")

        video_progress_start = 88.0
        video_progress_end = 96.0
        progress_state = {"last_progress": -1.0, "last_ts": 0.0, "last_msg": ""}

        def _video_progress_callback(ratio: float, message: str) -> None:
            ratio = float(max(0.0, min(1.0, ratio)))
            progress_value = video_progress_start + (video_progress_end - video_progress_start) * ratio
            now = time.time()
            should_update = (
                (progress_value - progress_state["last_progress"] >= 0.2)
                or (now - progress_state["last_ts"] >= 0.8)
                or (ratio >= 1.0)
                or (message != progress_state["last_msg"])
            )
            if should_update:
                progress_state["last_progress"] = progress_value
                progress_state["last_ts"] = now
                progress_state["last_msg"] = message
                task_manager.update_task(task_id, progress=progress_value, message=message)

        video_service.create_video_from_timeline(
            image_paths=image_paths,
            timeline=timeline,
            audio_path=mp3_path,
            output_path=str(output_video_path),
            timeline_overview=timeline_overview,
            resolution=output_resolution,
            progress_callback=_video_progress_callback,
        )
        task_manager.update_task(task_id, progress=96, message="Video and audio merged, finalizing outputs...")

        task_manager.update_task(
            task_id,
            state=TaskStatus.COMPLETED,
            progress=100,
            message=f"Completed (alignment grade: {grade})",
            result={
                "video_path": str(output_video_path),
                "timeline": timeline,
                "slide_overview": slide_overview,
                "timeline_overview": timeline_overview,
                "alignment_report": alignment_report,
                "slide_count": len(slides),
                "subtitle_count": len(subtitles),
                "segment_count": len(segments),
            },
        )
    except Exception as exc:
        logger.error(f"Video generation failed at stage={processing_stage}: {exc}", exc_info=True)
        task_manager.update_task(
            task_id,
            state=TaskStatus.FAILED,
            progress=0,
            error=f"Video generation failed at stage={processing_stage}: {exc}",
            message="Processing failed",
        )
