from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, Response
from fastapi.responses import FileResponse
import configparser
import json
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
_PROCESS_LOG_LOCK = threading.Lock()
_RUNTIME_TO_ENV_KEY = {
    "similarity_threshold": "DEFAULT_SIMILARITY_THRESHOLD",
    "min_display_duration": "DEFAULT_MIN_DISPLAY_DURATION",
    "output_resolution": "DEFAULT_OUTPUT_RESOLUTION",
    "embed_progress_bar": "VIDEO_EMBED_PROGRESS_BAR",
    "burn_srt_subtitles": "VIDEO_BURN_SRT_SUBTITLES",
    "progress_max_segments": "VIDEO_PROGRESS_MAX_SEGMENTS",
    "progress_label_max_chars": "VIDEO_PROGRESS_LABEL_MAX_CHARS",
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
    "align_structure_prior_weight": "ALIGN_STRUCTURE_PRIOR_WEIGHT",
}
_PROCESS_LOG_PATH = settings.BASE_DIR.parent / "processing.log"
_STRUCTURE_PRIOR_FILENAME = "structure_prior.txt"


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
    embed_progress_bar: object,
    burn_srt_subtitles: object,
    progress_max_segments: object,
    progress_label_max_chars: object,
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
    align_structure_prior_weight: object,
) -> Dict[str, Any]:
    return {
        "similarity_threshold": min(max(_parse_float(similarity_threshold, settings.DEFAULT_SIMILARITY_THRESHOLD), 0.1), 0.9),
        "min_display_duration": max(_parse_float(min_display_duration, settings.DEFAULT_MIN_DISPLAY_DURATION), 1.0),
        "output_resolution": _normalize_output_resolution(output_resolution),
        "embed_progress_bar": _parse_bool(embed_progress_bar),
        "burn_srt_subtitles": _parse_bool(burn_srt_subtitles),
        "progress_max_segments": int(min(max(_parse_int(progress_max_segments, settings.VIDEO_PROGRESS_MAX_SEGMENTS), 2), 60)),
        "progress_label_max_chars": int(min(max(_parse_int(progress_label_max_chars, settings.VIDEO_PROGRESS_LABEL_MAX_CHARS), 3), 32)),
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
        "align_structure_prior_weight": float(min(max(_parse_float(align_structure_prior_weight, settings.ALIGN_STRUCTURE_PRIOR_WEIGHT), 0.0), 0.35)),
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
        embed_progress_bar=env_values.get("VIDEO_EMBED_PROGRESS_BAR", settings.VIDEO_EMBED_PROGRESS_BAR),
        burn_srt_subtitles=env_values.get("VIDEO_BURN_SRT_SUBTITLES", settings.VIDEO_BURN_SRT_SUBTITLES),
        progress_max_segments=env_values.get("VIDEO_PROGRESS_MAX_SEGMENTS", settings.VIDEO_PROGRESS_MAX_SEGMENTS),
        progress_label_max_chars=env_values.get("VIDEO_PROGRESS_LABEL_MAX_CHARS", settings.VIDEO_PROGRESS_LABEL_MAX_CHARS),
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
        align_structure_prior_weight=env_values.get("ALIGN_STRUCTURE_PRIOR_WEIGHT", settings.ALIGN_STRUCTURE_PRIOR_WEIGHT),
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


def _safe_preview_text(value: object, max_chars: int = 120) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _append_processing_log(task_id: str, event: str, **payload: Any) -> None:
    entry = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "task_id": task_id,
        "event": event,
    }
    if payload:
        entry.update(payload)
    try:
        with _PROCESS_LOG_LOCK:
            _PROCESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _PROCESS_LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.warning("Failed to write processing log: %s", exc)


def _task_elapsed_seconds(task: Any) -> Optional[float]:
    if not getattr(task, "created_at", None):
        return None
    end_time = datetime.now()
    task_state = getattr(task, "state", None)
    if task_state in {TaskStatus.COMPLETED, TaskStatus.FAILED} and getattr(task, "updated_at", None):
        end_time = task.updated_at
    return max(0.0, (end_time - task.created_at).total_seconds())


def _build_structure_prior_text(
    task_id: str,
    slides: list,
    structure_meta: Dict[str, Any],
    structure_weight: float,
) -> str:
    chapter_anchors = list(structure_meta.get("chapter_anchors", [])) if isinstance(structure_meta, dict) else []
    order_to_slide = dict(structure_meta.get("order_to_slide", {})) if isinstance(structure_meta, dict) else {}
    lines = [
        f"task_id: {task_id}",
        f"generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"structure_prior_weight: {float(structure_weight):.3f}",
        f"slide_count: {len(slides)}",
        "",
        "detected_structure:",
        f"cover_slide_index: {structure_meta.get('cover_idx') if isinstance(structure_meta, dict) else None}",
        f"toc_slide_index: {structure_meta.get('toc_idx') if isinstance(structure_meta, dict) else None}",
        f"ending_slide_index: {structure_meta.get('ending_idx') if isinstance(structure_meta, dict) else None}",
        "",
        "chapter_anchors:",
    ]

    if chapter_anchors:
        for idx, pair in enumerate(chapter_anchors, start=1):
            slide_idx = int(pair[0]) if isinstance(pair, (list, tuple)) and len(pair) >= 1 else -1
            order = pair[1] if isinstance(pair, (list, tuple)) and len(pair) >= 2 else None
            lines.append(
                f"{idx}. slide_index={slide_idx}, slide_no={slide_idx + 1 if slide_idx >= 0 else 'N/A'}, order={order}"
            )
    else:
        lines.append("none")

    lines.extend(["", "order_to_slide:"])
    if order_to_slide:
        for order, slide_idx in sorted(order_to_slide.items(), key=lambda item: int(item[0])):
            slide_int = int(slide_idx)
            lines.append(f"order={int(order)} -> slide_index={slide_int} (slide_no={slide_int + 1})")
    else:
        lines.append("none")

    lines.extend(["", "slides:"])
    for fallback_idx, slide in enumerate(slides or []):
        slide_idx = int(slide.get("index", fallback_idx))
        title = _safe_preview_text(slide.get("title", ""), 96) or "<empty>"
        content = _safe_preview_text(slide.get("content", ""), 120) or "<empty>"
        notes = _safe_preview_text(slide.get("notes", ""), 120) or "<empty>"
        lines.append(f"slide_index={slide_idx}, slide_no={slide_idx + 1}")
        lines.append(f"title: {title}")
        lines.append(f"content: {content}")
        lines.append(f"notes: {notes}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _write_structure_prior_text(
    task_id: str,
    task_dir: Path,
    slides: list,
    structure_meta: Dict[str, Any],
    structure_weight: float,
) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    file_path = task_dir / _STRUCTURE_PRIOR_FILENAME
    text = _build_structure_prior_text(
        task_id=task_id,
        slides=slides,
        structure_meta=structure_meta,
        structure_weight=structure_weight,
    )
    file_path.write_text(text, encoding="utf-8")
    return file_path


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
            embed_progress_bar=settings.VIDEO_EMBED_PROGRESS_BAR,
            burn_srt_subtitles=settings.VIDEO_BURN_SRT_SUBTITLES,
            progress_max_segments=settings.VIDEO_PROGRESS_MAX_SEGMENTS,
            progress_label_max_chars=settings.VIDEO_PROGRESS_LABEL_MAX_CHARS,
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
            align_structure_prior_weight=settings.ALIGN_STRUCTURE_PRIOR_WEIGHT,
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
    embed_progress_bar: bool = Form(settings.VIDEO_EMBED_PROGRESS_BAR),
    burn_srt_subtitles: bool = Form(settings.VIDEO_BURN_SRT_SUBTITLES),
    progress_max_segments: int = Form(settings.VIDEO_PROGRESS_MAX_SEGMENTS),
    progress_label_max_chars: int = Form(settings.VIDEO_PROGRESS_LABEL_MAX_CHARS),
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
    align_structure_prior_weight: float = Form(settings.ALIGN_STRUCTURE_PRIOR_WEIGHT),
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
        embed_progress_bar=embed_progress_bar,
        burn_srt_subtitles=burn_srt_subtitles,
        progress_max_segments=progress_max_segments,
        progress_label_max_chars=progress_label_max_chars,
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
        align_structure_prior_weight=align_structure_prior_weight,
    )
    similarity_threshold = float(normalized_params["similarity_threshold"])
    min_display_duration = float(normalized_params["min_display_duration"])
    output_resolution = str(normalized_params["output_resolution"])
    embed_progress_bar = bool(normalized_params["embed_progress_bar"])
    burn_srt_subtitles = bool(normalized_params["burn_srt_subtitles"])
    progress_max_segments = int(normalized_params["progress_max_segments"])
    progress_label_max_chars = int(normalized_params["progress_label_max_chars"])
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
    align_structure_prior_weight = float(normalized_params["align_structure_prior_weight"])

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
        _append_processing_log(
            task_id,
            "task_created",
            pptx_filename=pptx_original_name,
            audio_filename=Path(mp3.filename or "").name,
            srt_filename=Path(srt.filename or "").name,
            output_resolution=output_resolution,
            similarity_threshold=similarity_threshold,
            min_display_duration=min_display_duration,
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
            embed_progress_bar,
            burn_srt_subtitles,
            progress_max_segments,
            progress_label_max_chars,
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
            align_structure_prior_weight,
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
        _append_processing_log(task_id, "task_create_failed", error=str(exc))
        shutil.rmtree(task_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to store uploaded files: {exc}")


@router.get("/task/{task_id}/status")
async def get_task_status(task_id: str):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    elapsed_seconds: Optional[float] = _task_elapsed_seconds(task)
    processing_duration_seconds: Optional[float] = None
    if task.result and isinstance(task.result, dict):
        duration_candidate = task.result.get("processing_duration_seconds")
        if isinstance(duration_candidate, (int, float)):
            processing_duration_seconds = float(duration_candidate)
    if processing_duration_seconds is None:
        processing_duration_seconds = elapsed_seconds
    if task.state in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
        elapsed_seconds = processing_duration_seconds
    eta_seconds: Optional[float] = None
    eta_formatted: Optional[str] = None

    if elapsed_seconds is not None:
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
        "processing_duration_seconds": processing_duration_seconds,
        "processing_duration_formatted": _format_seconds(processing_duration_seconds) if processing_duration_seconds is not None else None,
        "eta_seconds": eta_seconds,
        "eta_formatted": eta_formatted,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "updated_at": task.updated_at.isoformat() if task.updated_at else None,
        "finished_at": task.updated_at.isoformat() if task.state in {TaskStatus.COMPLETED, TaskStatus.FAILED} and task.updated_at else None,
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
    processing_duration_seconds = None
    if task.result:
        processing_duration_seconds = task.result.get("processing_duration_seconds")
    if processing_duration_seconds is None:
        processing_duration_seconds = _task_elapsed_seconds(task)

    return {
        "task_id": task_id,
        "timeline": task.result.get("timeline", []) if task.result else [],
        "timeline_overview": task.result.get("timeline_overview", []) if task.result else [],
        "slide_overview": task.result.get("slide_overview", []) if task.result else [],
        "alignment_report": task.result.get("alignment_report", {}) if task.result else {},
        "processing_duration_seconds": processing_duration_seconds,
        "processing_duration_formatted": _format_seconds(processing_duration_seconds) if processing_duration_seconds is not None else None,
        "structure_prior_path": task.result.get("structure_prior_path") if task.result else None,
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
    embed_progress_bar: bool,
    burn_srt_subtitles: bool,
    progress_max_segments: int,
    progress_label_max_chars: int,
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
    align_structure_prior_weight: float,
):
    normalized_params = _normalize_generation_params(
        similarity_threshold=similarity_threshold,
        min_display_duration=min_display_duration,
        output_resolution=output_resolution,
        embed_progress_bar=embed_progress_bar,
        burn_srt_subtitles=burn_srt_subtitles,
        progress_max_segments=progress_max_segments,
        progress_label_max_chars=progress_label_max_chars,
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
        align_structure_prior_weight=align_structure_prior_weight,
    )

    similarity_threshold = float(normalized_params["similarity_threshold"])
    min_display_duration = float(normalized_params["min_display_duration"])
    output_resolution = str(normalized_params["output_resolution"])
    embed_progress_bar = bool(normalized_params["embed_progress_bar"])
    burn_srt_subtitles = bool(normalized_params["burn_srt_subtitles"])
    progress_max_segments = int(normalized_params["progress_max_segments"])
    progress_label_max_chars = int(normalized_params["progress_label_max_chars"])
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
    align_structure_prior_weight = float(normalized_params["align_structure_prior_weight"])

    logger.info(
        "Video generation parameters: "
        f"similarity_threshold={similarity_threshold}, "
        f"min_display_duration={min_display_duration}, "
        f"output_resolution={output_resolution}, "
        f"embed_progress_bar={embed_progress_bar}, "
        f"burn_srt_subtitles={burn_srt_subtitles}, "
        f"progress_max_segments={progress_max_segments}, "
        f"progress_label_max_chars={progress_label_max_chars}, "
        f"srt_merge_gap_sec={srt_merge_gap_sec}, "
        f"srt_min_duration_sec={srt_min_duration_sec}, "
        f"align_max_backtrack={align_max_backtrack}, "
        f"align_max_forward_jump={align_max_forward_jump}, "
        f"align_switch_penalty={align_switch_penalty}, "
        f"align_forward_jump_penalty={align_forward_jump_penalty}, "
        f"align_switch_delay_ms={align_switch_delay_ms}, "
        f"align_enforce_no_revisit={align_enforce_no_revisit}, "
        f"align_min_slide_usage_ratio={align_min_slide_usage_ratio}, "
        f"align_enforce_sequential={align_enforce_sequential}, "
        f"align_structure_prior_weight={align_structure_prior_weight}"
    )

    processing_stage = "initialization"
    task_dir = Path(pptx_path).parent
    processing_started_ts = time.perf_counter()
    structure_prior_path: Optional[Path] = None

    _append_processing_log(
        task_id,
        "processing_started",
        pptx_path=pptx_path,
        audio_path=mp3_path,
        srt_path=srt_path,
        parameters=normalized_params,
    )

    try:
        task_manager.update_task(task_id, state=TaskStatus.PROCESSING, progress=1, message="Initializing task...")
        _append_processing_log(task_id, "stage_started", stage=processing_stage)

        processing_stage = "ppt_parsing"
        task_manager.update_task(task_id, progress=10, message="Parsing PPT structure...")
        _append_processing_log(task_id, "stage_started", stage=processing_stage)
        ppt_service = PPTService()
        slides = ppt_service.extract_slides(pptx_path)
        if not slides:
            raise ValueError("PPT parsing returned no slides")
        task_manager.update_task(task_id, progress=18, message=f"PPT parsed, detected {len(slides)} slides")
        _append_processing_log(task_id, "stage_completed", stage=processing_stage, slide_count=len(slides))

        processing_stage = "srt_parsing"
        task_manager.update_task(task_id, progress=22, message="Parsing SRT subtitles...")
        _append_processing_log(task_id, "stage_started", stage=processing_stage)
        srt_service = SRTService()
        subtitles = srt_service.parse_srt(srt_path, persist_normalized=True)
        if not subtitles:
            raise ValueError("SRT parsing returned no subtitles")
        task_manager.update_task(task_id, progress=28, message=f"SRT parsed, detected {len(subtitles)} subtitle blocks")
        _append_processing_log(task_id, "stage_completed", stage=processing_stage, subtitle_count=len(subtitles))

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
        _append_processing_log(task_id, "stage_completed", stage="subtitle_preprocess", segment_count=len(segments))

        alignment_service = AlignmentService()
        structure_meta: Dict[str, Any] = {}
        try:
            structure_meta = alignment_service._analyze_slide_structure(slides)
            structure_prior_path = _write_structure_prior_text(
                task_id=task_id,
                task_dir=task_dir,
                slides=slides,
                structure_meta=structure_meta,
                structure_weight=align_structure_prior_weight,
            )
            _append_processing_log(
                task_id,
                "structure_prior_written",
                path=str(structure_prior_path),
                cover_idx=structure_meta.get("cover_idx"),
                toc_idx=structure_meta.get("toc_idx"),
                ending_idx=structure_meta.get("ending_idx"),
                chapter_anchor_count=len(structure_meta.get("chapter_anchors", [])),
            )
        except Exception as prior_exc:
            logger.warning("Failed to write structure prior text for task %s: %s", task_id, prior_exc)
            _append_processing_log(task_id, "structure_prior_write_failed", error=str(prior_exc))

        processing_stage = "alignment"
        task_manager.update_task(task_id, progress=42, message="Running SRT <-> PPT alignment...")
        _append_processing_log(task_id, "stage_started", stage=processing_stage)
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
            align_structure_prior_weight=align_structure_prior_weight,
            enforce_sequential=align_enforce_sequential,
        )
        alignment_report = alignment_service.get_last_alignment_report()
        grade = alignment_report.get("summary", {}).get("grade", "N/A") if isinstance(alignment_report, dict) else "N/A"

        if not timeline:
            raise ValueError("Alignment failed: empty timeline")
        slide_overview = _build_slide_overview(slides)
        timeline_overview = _build_timeline_overview(timeline, slide_overview)
        task_manager.update_task(task_id, progress=58, message=f"Alignment done, timeline segments: {len(timeline)}, grade: {grade}")
        _append_processing_log(task_id, "stage_completed", stage=processing_stage, timeline_segments=len(timeline), grade=grade)

        processing_stage = "ppt_to_images"
        task_manager.update_task(task_id, progress=62, message="Exporting PPT slides to images...")
        _append_processing_log(task_id, "stage_started", stage=processing_stage)
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
        _append_processing_log(task_id, "stage_completed", stage=processing_stage, image_count=len(image_paths))

        processing_stage = "video_synthesis"
        task_manager.update_task(task_id, progress=82, message="Preparing video synthesis...")
        video_service = VideoService()

        ppt_stem = _sanitize_filename_stem(pptx_original_name)
        version = _next_video_version(ppt_stem)
        filename = f"{ppt_stem}_v{version}.mp4"
        output_video_path = settings.OUTPUT_DIR / task_id / filename
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        task_manager.update_task(task_id, progress=88, message=f"Start synthesizing video: {filename}")
        _append_processing_log(task_id, "stage_started", stage=processing_stage, output_filename=filename)

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
            srt_path=srt_path,
            burn_srt_subtitles=burn_srt_subtitles,
            timeline_overview=timeline_overview,
            embed_progress_bar=embed_progress_bar,
            progress_max_segments=progress_max_segments,
            progress_label_max_chars=progress_label_max_chars,
            resolution=output_resolution,
            progress_callback=_video_progress_callback,
        )
        task_manager.update_task(task_id, progress=96, message="Video and audio merged, finalizing outputs...")

        processing_seconds = max(0.0, time.perf_counter() - processing_started_ts)
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
                "processing_duration_seconds": processing_seconds,
                "structure_prior_path": str(structure_prior_path) if structure_prior_path else None,
            },
        )
        _append_processing_log(
            task_id,
            "processing_completed",
            stage=processing_stage,
            output_video_path=str(output_video_path),
            processing_duration_seconds=round(processing_seconds, 3),
            slide_count=len(slides),
            subtitle_count=len(subtitles),
            segment_count=len(segments),
            timeline_segments=len(timeline),
        )
    except Exception as exc:
        processing_seconds = max(0.0, time.perf_counter() - processing_started_ts)
        logger.error(f"Video generation failed at stage={processing_stage}: {exc}", exc_info=True)
        _append_processing_log(
            task_id,
            "processing_failed",
            stage=processing_stage,
            error=str(exc),
            processing_duration_seconds=round(processing_seconds, 3),
        )
        task_manager.update_task(
            task_id,
            state=TaskStatus.FAILED,
            progress=0,
            error=f"Video generation failed at stage={processing_stage}: {exc}",
            message="Processing failed",
        )
