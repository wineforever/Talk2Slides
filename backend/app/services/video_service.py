import os
import re
import subprocess
import tempfile
import logging
import time
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple
import shutil

import ffmpeg

from app.core.config import settings

logger = logging.getLogger(__name__)


class VideoService:
    """Service for building final video from slide images and alignment timeline."""
    _cached_h264_encoder: Optional[str] = None

    def _build_bottom_progress_bar_spec(
        self,
        width: int,
        height: int,
        total_duration_sec: float,
    ) -> Dict[str, Any]:
        """Build style and timing spec for an in-video bottom progress bar."""
        safe_duration = float(max(0.1, total_duration_sec))
        # Structured bar: 2%-3% of frame height, pinned to bottom edge.
        bar_height = max(8, int(round(height * 0.024)))
        if bar_height % 2 != 0:
            bar_height += 1
        bar_margin = max(1, int(round(height * 0.002)))
        bar_y = max(0, height - bar_margin - bar_height)
        return {
            "duration_expr": f"{safe_duration:.6f}",
            "bar_height": int(bar_height),
            "bar_y": int(bar_y),
            # Use opaque colors to avoid expensive alpha blending in long videos.
            "track_color": "0x1A1E22",
            "progress_color": "0x00C2FF",
        }

    def _pick_h264_encoder(self) -> str:
        cached = type(self)._cached_h264_encoder
        if cached:
            return cached

        preferred = ["h264_nvenc", "h264_qsv", "h264_amf", "libx264"]
        detected = "libx264"
        try:
            result = subprocess.run(
                [settings.FFMPEG_PATH, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=15,
            )
            text = (result.stdout or "") + "\n" + (result.stderr or "")
            lower_text = text.lower()
            for encoder in preferred:
                if encoder == "libx264" or encoder in lower_text:
                    detected = encoder
                    break
        except Exception:
            detected = "libx264"

        type(self)._cached_h264_encoder = detected
        logger.info("Selected H.264 encoder: %s", detected)
        return detected

    def _build_video_encoder_args(self, encoder: str) -> List[str]:
        enc = str(encoder or "").strip().lower()
        if enc == "libx264":
            threads = max(1, min(os.cpu_count() or 4, 4))
            return [
                "-c:v",
                "libx264",
                "-threads",
                str(threads),
                "-preset",
                "ultrafast",
                "-tune",
                "stillimage",
                "-pix_fmt",
                "yuv420p",
            ]
        # Keep hardware args minimal for broad compatibility.
        return [
            "-c:v",
            enc,
            "-pix_fmt",
            "yuv420p",
        ]

    def _compact_segment_label(self, label: str, fallback: str, max_chars: int) -> str:
        compact = str(label or "").strip()
        compact = re.sub(r"[\r\n\t]+", " ", compact)
        compact = re.sub(r"\s+", " ", compact)
        compact = re.sub(r"[|/]+", " ", compact)
        compact = re.sub(r"^[\-\*\d一二三四五六七八九十]+[\.、:：\)\]）】\s]*", "", compact)
        compact = compact.strip(" -_，,。；;：:、")
        if not compact:
            compact = fallback
        chunks = [part.strip() for part in re.split(r"[。！？!?；;，,：:|]", compact) if part.strip()]
        compact = chunks[0] if chunks else compact
        parts_by_space = [part for part in compact.split(" ") if part]
        if len(parts_by_space) > 1:
            compact = parts_by_space[0]
        max_len = max(3, int(max_chars))
        if len(compact) > max_len:
            compact = compact[:max_len].rstrip(" -_，,。；;：:")
        return compact or fallback

    def _estimate_text_width_px(self, text: str, font_size: int) -> float:
        width = 0.0
        size = max(8, int(font_size))
        for ch in str(text or ""):
            if re.match(r"[\u4e00-\u9fff]", ch):
                width += size * 0.92
            elif ch.isspace():
                width += size * 0.35
            elif ch in ".,:;|!":
                width += size * 0.32
            else:
                width += size * 0.58
        return max(0.0, width)

    def _fit_segment_label_to_width(
        self,
        label: str,
        fallback: str,
        segment_width_px: int,
        font_size: int,
        max_chars: int,
    ) -> str:
        normalized = self._compact_segment_label(label, fallback, max_chars=max_chars)
        safe_max = max(3, int(max_chars))
        if len(normalized) > safe_max:
            normalized = normalized[:safe_max]

        usable_width = max(4.0, float(segment_width_px) - 4.0)
        if self._estimate_text_width_px(normalized, font_size) <= usable_width:
            return normalized

        ellipsis = "..."
        if self._estimate_text_width_px(ellipsis, font_size) > usable_width:
            return ellipsis

        core = normalized
        if len(core) > safe_max - len(ellipsis):
            core = core[: max(1, safe_max - len(ellipsis))]

        while core:
            candidate = core.rstrip(" -_，,。；;：:") + ellipsis
            if self._estimate_text_width_px(candidate, font_size) <= usable_width:
                return candidate
            core = core[:-1]
        return ellipsis

    def _escape_drawtext_text(self, text: str) -> str:
        safe = str(text or "")
        safe = safe.replace("\\", "\\\\")
        safe = safe.replace(":", "\\:")
        safe = safe.replace("'", "\\'")
        safe = safe.replace(",", "\\,")
        safe = safe.replace("%", "\\%")
        return safe

    def _pick_drawtext_font_option(self) -> str:
        if os.name == "nt":
            candidates = [
                r"C:\Windows\Fonts\msyh.ttc",
                r"C:\Windows\Fonts\simhei.ttf",
                r"C:\Windows\Fonts\simsun.ttc",
                r"C:\Windows\Fonts\arial.ttf",
            ]
        else:
            candidates = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            ]

        for path in candidates:
            p = Path(path)
            if not p.exists():
                continue
            ffmpeg_path = str(p.resolve()).replace("\\", "/")
            if len(ffmpeg_path) > 1 and ffmpeg_path[1] == ":":
                ffmpeg_path = ffmpeg_path[0] + "\\:" + ffmpeg_path[2:]
            return f"fontfile='{ffmpeg_path}'"
        return ""

    def _build_progress_segments(
        self,
        timeline: List[Dict[str, Any]],
        timeline_overview: Optional[List[Dict[str, Any]]],
        total_duration: float,
        max_segments: int,
        label_max_chars: int,
    ) -> List[Dict[str, Any]]:
        total_duration = max(0.1, float(total_duration))
        safe_max_segments = max(1, int(max_segments))
        safe_label_max_chars = max(3, int(label_max_chars))
        source = timeline_overview if timeline_overview else timeline
        raw_items: List[Dict[str, Any]] = []
        for idx, item in enumerate(source or []):
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
            if end <= start:
                continue
            label = str(item.get("label") or f"第{idx + 1}段").strip()
            raw_items.append({"start": start, "end": end, "label": label})

        raw_items.sort(key=lambda x: x["start"])
        segments: List[Dict[str, Any]] = []
        cursor = 0.0
        fallback_index = 1

        for item in raw_items:
            start = max(0.0, min(total_duration, float(item["start"])))
            end = max(0.0, min(total_duration, float(item["end"])))
            if end <= start:
                continue

            if start > cursor + 1e-3:
                segments.append(
                    {
                        "start": cursor,
                        "end": start,
                        "label": self._compact_segment_label(
                            "",
                            f"第{fallback_index}段",
                            max_chars=safe_label_max_chars,
                        ),
                    }
                )
                fallback_index += 1
                cursor = start

            start = max(start, cursor)
            if end <= start:
                continue
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "label": self._compact_segment_label(
                        item["label"],
                        f"第{fallback_index}段",
                        max_chars=safe_label_max_chars,
                    ),
                }
            )
            cursor = end
            fallback_index += 1

        if cursor < total_duration - 1e-3:
            segments.append(
                {
                    "start": cursor,
                    "end": total_duration,
                    "label": self._compact_segment_label(
                        "",
                        f"第{fallback_index}段",
                        max_chars=safe_label_max_chars,
                    ),
                }
            )

        if not segments:
            segments = [{"start": 0.0, "end": total_duration, "label": "全程"}]

        merged: List[Dict[str, Any]] = []
        for seg in segments:
            if merged and seg["label"] == merged[-1]["label"] and seg["start"] <= merged[-1]["end"] + 1e-3:
                merged[-1]["end"] = max(float(merged[-1]["end"]), float(seg["end"]))
                continue
            merged.append(dict(seg))

        while len(merged) > safe_max_segments:
            shortest_idx = min(
                range(len(merged)),
                key=lambda i: float(merged[i]["end"]) - float(merged[i]["start"]),
            )
            if len(merged) <= 1:
                break
            if shortest_idx == 0:
                merged[1]["start"] = merged[0]["start"]
                merged.pop(0)
            else:
                merged[shortest_idx - 1]["end"] = merged[shortest_idx]["end"]
                merged.pop(shortest_idx)

        return merged

    def _ensure_first_frame_is_home_slide(
        self,
        timeline: List[Dict[str, Any]],
        slide_count: int,
    ) -> List[Dict[str, Any]]:
        if not timeline or slide_count <= 0:
            return timeline

        normalized = [dict(segment) for segment in timeline]
        first_idx = int(normalized[0].get("slide_index", 0))
        if first_idx != 0:
            logger.warning("First-frame correction: force first segment to slide 0")
            normalized[0]["slide_index"] = 0
        return normalized

    def _ensure_last_tail_is_last_slide(
        self,
        timeline: List[Dict[str, Any]],
        slide_count: int,
        tail_seconds: float,
    ) -> List[Dict[str, Any]]:
        """Force the last tail_seconds of video to use the final PPT slide."""
        if not timeline or slide_count <= 0:
            return timeline

        tail_seconds = float(max(0.0, tail_seconds))
        if tail_seconds <= 0:
            return [dict(segment) for segment in timeline]

        normalized = [dict(segment) for segment in timeline]
        normalized.sort(key=lambda x: float(x.get("start", 0.0)))

        first_start = float(normalized[0].get("start", 0.0))
        final_end = float(normalized[-1].get("end", first_start))
        if final_end <= first_start:
            return normalized

        last_slide_index = int(slide_count - 1)
        tail_start = max(first_start, final_end - tail_seconds)

        rebuilt: List[Dict[str, Any]] = []
        for segment in normalized:
            seg_start = float(segment.get("start", 0.0))
            seg_end = float(segment.get("end", seg_start))
            if seg_end <= tail_start:
                rebuilt.append(dict(segment))
                continue
            if seg_start < tail_start:
                truncated = dict(segment)
                truncated["start"] = seg_start
                truncated["end"] = tail_start
                truncated["duration"] = max(0.0, tail_start - seg_start)
                if truncated["duration"] > 1e-6:
                    rebuilt.append(truncated)

        tail_duration = max(0.0, final_end - tail_start)
        if tail_duration <= 1e-6:
            normalized[-1]["slide_index"] = last_slide_index
            normalized[-1]["duration"] = max(
                0.0,
                float(normalized[-1].get("end", 0.0)) - float(normalized[-1].get("start", 0.0)),
            )
            return normalized

        if rebuilt and int(rebuilt[-1].get("slide_index", -1)) == last_slide_index:
            prev_end = float(rebuilt[-1].get("end", tail_start))
            if abs(prev_end - tail_start) < 1e-3:
                rebuilt[-1]["end"] = final_end
                rebuilt[-1]["duration"] = max(0.0, final_end - float(rebuilt[-1].get("start", tail_start)))
            else:
                rebuilt.append(
                    {
                        "start": tail_start,
                        "end": final_end,
                        "slide_index": last_slide_index,
                        "duration": tail_duration,
                    }
                )
        else:
            rebuilt.append(
                {
                    "start": tail_start,
                    "end": final_end,
                    "slide_index": last_slide_index,
                    "duration": tail_duration,
                }
            )

        cleaned: List[Dict[str, Any]] = []
        for segment in rebuilt:
            seg_start = float(segment.get("start", 0.0))
            seg_end = float(segment.get("end", seg_start))
            duration = seg_end - seg_start
            if duration <= 1e-6:
                continue
            item = dict(segment)
            item["start"] = seg_start
            item["end"] = seg_end
            item["duration"] = duration
            cleaned.append(item)

        if not cleaned:
            cleaned = [
                {
                    "start": first_start,
                    "end": final_end,
                    "slide_index": last_slide_index,
                    "duration": max(0.0, final_end - first_start),
                }
            ]

        cleaned[-1]["slide_index"] = last_slide_index
        cleaned[-1]["duration"] = max(0.0, float(cleaned[-1]["end"]) - float(cleaned[-1]["start"]))

        logger.info(
            "Last-tail correction applied: tail=%.2fs, last_slide=%d",
            tail_seconds,
            last_slide_index,
        )
        return cleaned

    def _validate_image_files(self, image_paths: List[str]) -> bool:
        for i, path in enumerate(image_paths):
            p = Path(path)
            if not p.exists() or not p.is_file():
                logger.error("Image not found or invalid: %s (index=%d)", path, i)
                return False
            try:
                with open(p, "rb") as f:
                    f.read(1)
            except Exception as exc:
                logger.error("Image unreadable: %s (index=%d), err=%s", path, i, exc)
                return False
        return True

    def _normalize_path_for_ffmpeg(self, path: str) -> str:
        return str(Path(path).resolve()).replace("\\", "/")

    def _escape_subtitles_filter_path(self, path: str) -> str:
        # Escape filtergraph metacharacters for subtitles=filename=...
        safe = self._normalize_path_for_ffmpeg(path)
        safe = safe.replace("\\", "\\\\")
        safe = safe.replace(":", "\\:")
        safe = safe.replace("'", "\\'")
        safe = safe.replace(",", "\\,")
        safe = safe.replace("[", "\\[")
        safe = safe.replace("]", "\\]")
        return safe

    def _build_subtitles_filter(self, srt_path: str) -> str:
        subtitle_file = Path(srt_path)
        if not subtitle_file.exists():
            raise FileNotFoundError(f"SRT file not found for subtitle burn-in: {srt_path}")
        escaped_path = self._escape_subtitles_filter_path(str(subtitle_file))
        return f"subtitles='{escaped_path}':charenc=UTF-8"

    def _invoke_progress(
        self,
        progress_callback: Optional[Callable[[float, str], None]],
        ratio: float,
        message: str,
    ) -> None:
        if not progress_callback:
            return
        ratio = float(max(0.0, min(1.0, ratio)))
        progress_callback(ratio, message)

    def _parse_ffmpeg_time(self, value: str) -> Optional[float]:
        try:
            parts = value.strip().split(":")
            if len(parts) != 3:
                return None
            h = float(parts[0])
            m = float(parts[1])
            s = float(parts[2])
            return h * 3600.0 + m * 60.0 + s
        except Exception:
            return None

    def _run_ffmpeg_with_progress(
        self,
        cmd: List[str],
        total_duration_sec: float,
        stage_message: str,
        progress_callback: Optional[Callable[[float, str], None]],
        progress_range: Tuple[float, float],
    ) -> None:
        start_ratio, end_ratio = progress_range
        start_ratio = float(max(0.0, min(1.0, start_ratio)))
        end_ratio = float(max(start_ratio, min(1.0, end_ratio)))

        logger.info("Running ffmpeg command: %s", " ".join(cmd))
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            # Merge stderr into stdout so ffmpeg logs are consumed continuously.
            # This avoids pipe-buffer deadlocks when stderr is verbose.
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1,
        )

        def map_ratio(local_ratio: float) -> float:
            local_ratio = float(max(0.0, min(1.0, local_ratio)))
            return start_ratio + (end_ratio - start_ratio) * local_ratio

        last_reported = -1.0
        total_duration_sec = float(max(0.0, total_duration_sec))
        combined_output: deque[str] = deque(maxlen=400)

        try:
            if process.stdout is not None:
                for raw_line in process.stdout:
                    line = raw_line.strip()
                    if not line:
                        continue
                    combined_output.append(line)

                    local_ratio = None
                    if line.startswith("out_time_ms="):
                        # ffmpeg key name is out_time_ms but value unit is microseconds.
                        raw_val = line.split("=", 1)[1].strip()
                        if raw_val:
                            try:
                                out_time_sec = float(raw_val) / 1_000_000.0
                                if total_duration_sec > 0:
                                    local_ratio = out_time_sec / total_duration_sec
                            except Exception:
                                pass
                    elif line.startswith("out_time="):
                        raw_val = line.split("=", 1)[1].strip()
                        out_time_sec = self._parse_ffmpeg_time(raw_val)
                        if out_time_sec is not None and total_duration_sec > 0:
                            local_ratio = out_time_sec / total_duration_sec
                    elif line == "progress=end":
                        local_ratio = 1.0

                    if local_ratio is None:
                        continue

                    mapped = map_ratio(local_ratio)
                    if mapped - last_reported >= 0.003 or local_ratio >= 1.0:
                        last_reported = mapped
                        self._invoke_progress(
                            progress_callback,
                            mapped,
                            f"{stage_message} {max(0.0, min(100.0, local_ratio * 100.0)):.1f}%",
                        )

            return_code = process.wait()
            if return_code != 0:
                ffmpeg_tail = "\n".join(combined_output).strip()
                if ffmpeg_tail:
                    raise RuntimeError(f"ffmpeg exited with code {return_code}\n{ffmpeg_tail}")
                raise RuntimeError(f"ffmpeg exited with code {return_code}")
        finally:
            try:
                if process.stdout:
                    process.stdout.close()
            except Exception:
                pass
            try:
                if process.stderr:
                    process.stderr.close()
            except Exception:
                pass

        self._invoke_progress(progress_callback, end_ratio, f"{stage_message} 100.0%")

    def create_video_from_timeline(
        self,
        image_paths: List[str],
        timeline: List[Dict[str, Any]],
        audio_path: str,
        output_path: str,
        srt_path: Optional[str] = None,
        burn_srt_subtitles: bool = False,
        timeline_overview: Optional[List[Dict[str, Any]]] = None,
        embed_progress_bar: bool = True,
        progress_max_segments: int = 10,
        progress_label_max_chars: int = 10,
        resolution: str = "1920x1080",
        fps: int = 30,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        if not image_paths:
            raise ValueError("image_paths must not be empty")
        if not timeline:
            raise ValueError("timeline must not be empty")
        if burn_srt_subtitles and not srt_path:
            raise ValueError("srt_path is required when burn_srt_subtitles is enabled")
        if burn_srt_subtitles and srt_path and not Path(srt_path).exists():
            raise FileNotFoundError(f"SRT file not found: {srt_path}")

        self._invoke_progress(progress_callback, 0.01, "Validating timeline and files...")

        if settings.VIDEO_FORCE_FIRST_SLIDE_FRAME:
            timeline = self._ensure_first_frame_is_home_slide(timeline=timeline, slide_count=len(image_paths))

        tail_sec = float(max(0.1, getattr(settings, "VIDEO_FORCE_LAST_SLIDE_TAIL_SEC", 5.0)))
        timeline = self._ensure_last_tail_is_last_slide(
            timeline=timeline,
            slide_count=len(image_paths),
            tail_seconds=tail_sec,
        )

        if not self._validate_image_files(image_paths):
            raise FileNotFoundError("Some slide images are missing or unreadable")

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        width, height = map(int, resolution.split("x"))

        temp_dir: Optional[Path] = None
        try:
            base_temp_dir = Path(settings.BASE_DIR) / "temp" / "video_processing"
            base_temp_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = Path(tempfile.mkdtemp(dir=base_temp_dir, prefix=f"video_{int(time.time())}_"))

            self._invoke_progress(progress_callback, 0.08, "Preparing concat list...")
            concat_file = temp_dir / "concat.txt"
            _, timeline_total_duration = self._create_concat_file(concat_file, image_paths, timeline, fps)

            target_duration = max(0.1, timeline_total_duration)
            audio_duration = 0.0
            if audio_path:
                try:
                    audio_duration = float(self.extract_audio_info(audio_path).get("duration", 0.0))
                except Exception:
                    audio_duration = 0.0

            if audio_duration > 0 and target_duration > 0:
                target_duration = min(audio_duration, target_duration)

            self._invoke_progress(progress_callback, 0.20, "Starting slide stitching...")
            temp_video = temp_dir / "temp_video.mp4"
            self._create_silent_video(
                concat_file=concat_file,
                output_path=str(temp_video),
                width=width,
                height=height,
                fps=fps,
                expected_duration=max(0.1, timeline_total_duration),
                srt_path=srt_path,
                burn_srt_subtitles=burn_srt_subtitles,
                progress_callback=progress_callback,
                progress_range=(0.20, 0.70),
            )

            if not temp_video.exists() or temp_video.stat().st_size <= 0:
                raise RuntimeError("Silent video generation failed")

            if embed_progress_bar:
                self._invoke_progress(progress_callback, 0.70, "Generating progress bar track...")
                bar_spec = self._build_bottom_progress_bar_spec(width=width, height=height, total_duration_sec=target_duration)
                progress_segments = self._build_progress_segments(
                    timeline=timeline,
                    timeline_overview=timeline_overview,
                    total_duration=target_duration,
                    max_segments=progress_max_segments,
                    label_max_chars=progress_label_max_chars,
                )
                bar_video = temp_dir / "progress_bar.mp4"
                self._create_progress_bar_video(
                    output_path=str(bar_video),
                    width=width,
                    bar_height=int(bar_spec["bar_height"]),
                    fps=fps,
                    duration=max(0.1, target_duration),
                    track_color=str(bar_spec["track_color"]),
                    progress_color=str(bar_spec["progress_color"]),
                    segments=progress_segments,
                    label_max_chars=progress_label_max_chars,
                    progress_callback=progress_callback,
                    progress_range=(0.70, 0.78),
                )

                self._invoke_progress(progress_callback, 0.78, "Compositing final video...")
                self._compose_final_video_with_progress(
                    base_video_path=str(temp_video),
                    progress_video_path=str(bar_video),
                    audio_path=audio_path if audio_path else None,
                    output_path=output_path,
                    progress_y=int(bar_spec["bar_y"]),
                    expected_duration=max(0.1, target_duration),
                    progress_callback=progress_callback,
                    progress_range=(0.78, 0.97),
                )
            else:
                self._invoke_progress(progress_callback, 0.70, "In-video progress bar disabled, exporting video...")
                if audio_path:
                    self._merge_audio_video(
                        video_path=str(temp_video),
                        audio_path=str(audio_path),
                        output_path=output_path,
                        expected_duration=max(0.1, target_duration),
                        progress_callback=progress_callback,
                        progress_range=(0.78, 0.97),
                    )
                else:
                    cmd = [
                        settings.FFMPEG_PATH,
                        "-y",
                        "-progress",
                        "pipe:1",
                        "-nostats",
                        "-i",
                        str(temp_video),
                        "-c:v",
                        "copy",
                        output_path,
                    ]
                    self._run_ffmpeg_with_progress(
                        cmd=cmd,
                        total_duration_sec=max(0.1, target_duration),
                        stage_message="Exporting final video",
                        progress_callback=progress_callback,
                        progress_range=(0.78, 0.97),
                    )

            if not output_path_obj.exists() or output_path_obj.stat().st_size <= 0:
                raise RuntimeError("Final output file was not generated")

            self._invoke_progress(progress_callback, 1.0, "Video generation completed")
            return str(output_path_obj)

        except Exception as exc:
            logger.error("Video synthesis failed: %s", exc, exc_info=True)
            raise Exception(f"Video synthesis failed: {exc}")
        finally:
            if temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass

    def _create_concat_file(
        self,
        concat_file: Path,
        image_paths: List[str],
        timeline: List[Dict[str, Any]],
        fps: int,
    ) -> Tuple[str, float]:
        lines: List[str] = []
        total_duration = 0.0

        for idx, segment in enumerate(timeline):
            slide_idx = int(segment.get("slide_index", -1))
            start_time = float(segment.get("start", 0.0))
            end_time = float(segment.get("end", start_time))
            duration = end_time - start_time

            if duration <= 0:
                raise ValueError(
                    f"Invalid timeline segment at index {idx}: start={start_time}, end={end_time}, duration={duration}"
                )
            if slide_idx < 0 or slide_idx >= len(image_paths):
                raise ValueError(f"Invalid slide index {slide_idx}, expected [0, {len(image_paths) - 1}]")

            image_path = image_paths[slide_idx]
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path} (segment={idx}, slide={slide_idx})")

            normalized_path = self._normalize_path_for_ffmpeg(image_path)
            lines.append(f"file '{normalized_path}'")
            lines.append(f"duration {duration}")
            total_duration += duration

        if timeline:
            last_slide_idx = int(timeline[-1]["slide_index"])
            last_image_path = image_paths[last_slide_idx]
            normalized_last_path = self._normalize_path_for_ffmpeg(last_image_path)
            lines.append(f"file '{normalized_last_path}'")

        content = "\n".join(lines)
        with open(concat_file, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(
            "Concat file created: %s (segments=%d, total_duration=%.2fs)",
            concat_file,
            len(timeline),
            total_duration,
        )
        return content, total_duration

    def _create_silent_video(
        self,
        concat_file: Path,
        output_path: str,
        width: int,
        height: int,
        fps: int,
        expected_duration: float,
        srt_path: Optional[str] = None,
        burn_srt_subtitles: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        progress_range: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        if not concat_file.exists():
            raise FileNotFoundError(f"Concat file not found: {concat_file}")

        vf_filters = [
            f"scale={width}:{height}:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            f"fps={int(fps)}",
        ]
        if burn_srt_subtitles and srt_path:
            vf_filters.append(self._build_subtitles_filter(srt_path))
        vf_filters.append("format=yuv420p")
        vf = ",".join(vf_filters)
        cpu_encoder_args = self._build_video_encoder_args("libx264")

        cmd = [
            settings.FFMPEG_PATH,
            "-y",
            "-progress",
            "pipe:1",
            "-nostats",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-vf",
            vf,
            *cpu_encoder_args,
            output_path,
        ]

        self._run_ffmpeg_with_progress(
            cmd=cmd,
            total_duration_sec=expected_duration,
            stage_message="Stitching slides and subtitles" if burn_srt_subtitles else "Stitching slides",
            progress_callback=progress_callback,
            progress_range=progress_range,
        )

    def _create_progress_bar_video(
        self,
        output_path: str,
        width: int,
        bar_height: int,
        fps: int,
        duration: float,
        track_color: str,
        progress_color: str,
        segments: List[Dict[str, Any]],
        label_max_chars: int,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        progress_range: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        duration = max(0.1, float(duration))
        bar_height = max(2, int(bar_height))
        if bar_height % 2 != 0:
            bar_height += 1
        width = max(2, int(width))
        safe_segments = segments or [{"start": 0.0, "end": duration, "label": "全程"}]

        progress_expr = f"max(-{width}\\,min(0\\,-{width}+{width}*t/{duration:.6f}))"
        filters: List[str] = [
            f"[0:v]setpts=PTS-STARTPTS,format=yuv420p[track]",
            f"[1:v]setpts=PTS-STARTPTS,format=yuv420p[fill]",
            f"[track][fill]overlay=x='{progress_expr}':y=0:shortest=1:repeatlast=0:format=yuv420[bar0]",
        ]

        current_stage = "bar0"
        separator_color = "0x6A7785"
        active_overlay_color = "0xFFFFFF@0.10"
        text_base_color = "0xD8DEE7"
        text_active_color = "0x101010"
        font_size = max(10, int(round(bar_height * 0.52)))
        font_option = self._pick_drawtext_font_option()
        font_prefix = (font_option + ":") if font_option else ""

        geom_segments: List[Dict[str, Any]] = []
        for idx, seg in enumerate(safe_segments):
            start = max(0.0, min(duration, float(seg.get("start", 0.0))))
            end = max(0.0, min(duration, float(seg.get("end", start))))
            if end <= start:
                continue
            x0 = int(round(width * (start / duration)))
            x1 = int(round(width * (end / duration)))
            x0 = max(0, min(width - 1, x0))
            x1 = max(x0 + 1, min(width, x1))
            geom_segments.append(
                {
                    "index": idx,
                    "start": start,
                    "end": end,
                    "x": x0,
                    "w": max(1, x1 - x0),
                    "label": str(seg.get("label") or ""),
                }
            )

        for seg in geom_segments:
            stage_name = f"bar_active_{seg['index']}"
            filters.append(
                f"[{current_stage}]drawbox=x={int(seg['x'])}:y=0:w={int(seg['w'])}:h={bar_height}:"
                f"color={active_overlay_color}:t=fill:enable='between(t,{float(seg['start']):.3f},{float(seg['end']):.3f})'"
                f"[{stage_name}]"
            )
            current_stage = stage_name

        for seg in geom_segments[:-1]:
            boundary_x = max(0, min(width - 1, int(seg["x"] + seg["w"] - 1)))
            stage_name = f"bar_sep_{seg['index']}"
            filters.append(
                f"[{current_stage}]drawbox=x={boundary_x}:y=0:w=1:h={bar_height}:color={separator_color}:t=fill[{stage_name}]"
            )
            current_stage = stage_name

        for seg in geom_segments:
            label = self._fit_segment_label_to_width(
                label=str(seg["label"]),
                fallback=f"第{int(seg['index']) + 1}段",
                segment_width_px=int(seg["w"]),
                font_size=font_size,
                max_chars=label_max_chars,
            )
            safe_label = self._escape_drawtext_text(label)
            start_t = float(seg["start"])
            end_t = float(seg["end"])
            alpha_expr = f"if(lt(t\\,{end_t:.3f})\\,0\\,1)"
            common = (
                f"{font_prefix}text='{safe_label}':fontsize={font_size}:"
                f"x={int(seg['x'])}+({int(seg['w'])}-text_w)/2:y=(h-text_h)/2"
            )
            base_stage = f"bar_txtb_{seg['index']}"
            filters.append(
                f"[{current_stage}]drawtext={common}:fontcolor={text_base_color}[{base_stage}]"
            )
            active_stage = f"bar_txta_{seg['index']}"
            filters.append(
                f"[{base_stage}]drawtext={common}:fontcolor={text_active_color}:alpha='{alpha_expr}'[{active_stage}]"
            )
            current_stage = active_stage

        filters.append(f"[{current_stage}]copy[vout]")
        filter_complex = ";".join(filters)

        cmd = [
            settings.FFMPEG_PATH,
            "-y",
            "-progress",
            "pipe:1",
            "-nostats",
            "-f",
            "lavfi",
            "-i",
            f"color=c={track_color}:s={width}x{bar_height}:r={int(fps)}:d={duration:.6f}",
            "-f",
            "lavfi",
            "-i",
            f"color=c={progress_color}:s={width}x{bar_height}:r={int(fps)}:d={duration:.6f}",
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            *self._build_video_encoder_args("libx264"),
            output_path,
        ]

        self._run_ffmpeg_with_progress(
            cmd=cmd,
            total_duration_sec=duration,
            stage_message="Rendering progress bar",
            progress_callback=progress_callback,
            progress_range=progress_range,
        )

    def _compose_final_video_with_progress(
        self,
        base_video_path: str,
        progress_video_path: str,
        audio_path: Optional[str],
        output_path: str,
        progress_y: int,
        expected_duration: float,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        progress_range: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        if not Path(base_video_path).exists():
            raise FileNotFoundError(f"Base video file not found: {base_video_path}")
        if not Path(progress_video_path).exists():
            raise FileNotFoundError(f"Progress bar video not found: {progress_video_path}")
        if audio_path and not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(str(output_dir), os.W_OK):
            raise PermissionError(f"Output directory is not writable: {output_dir}")

        filter_complex = (
            "[0:v]setpts=PTS-STARTPTS,format=yuv420p[base];"
            "[1:v]setpts=PTS-STARTPTS,format=yuv420p[bar];"
            f"[base][bar]overlay=x=0:y={int(progress_y)}:shortest=1:format=yuv420[vout]"
        )

        encoder = self._pick_h264_encoder()

        def build_cmd(video_encoder: str) -> List[str]:
            cmd = [
                settings.FFMPEG_PATH,
                "-y",
                "-progress",
                "pipe:1",
                "-nostats",
                "-i",
                base_video_path,
                "-i",
                progress_video_path,
            ]
            if audio_path:
                cmd.extend(["-i", audio_path])
            cmd.extend(
                [
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "[vout]",
                ]
            )
            if audio_path:
                cmd.extend(["-map", "2:a:0", "-c:a", "aac"])
            cmd.extend(self._build_video_encoder_args(video_encoder))
            if audio_path:
                cmd.append("-shortest")
            cmd.append(output_path)
            return cmd

        try:
            self._run_ffmpeg_with_progress(
                cmd=build_cmd(encoder),
                total_duration_sec=max(0.1, expected_duration),
                stage_message="Compositing final video",
                progress_callback=progress_callback,
                progress_range=progress_range,
            )
        except Exception as exc:
            if encoder == "libx264":
                raise
            logger.warning("Hardware encoder %s failed (%s), falling back to libx264", encoder, exc)
            type(self)._cached_h264_encoder = "libx264"
            self._run_ffmpeg_with_progress(
                cmd=build_cmd("libx264"),
                total_duration_sec=max(0.1, expected_duration),
                stage_message="Compositing final video (CPU fallback)",
                progress_callback=progress_callback,
                progress_range=progress_range,
            )

    def _analyze_ffmpeg_error(self, error_message: str, concat_file: Path) -> str:
        analysis = []
        if "Invalid data found when processing input" in error_message:
            analysis.append("1. concat format or image files may be invalid")
            analysis.append("2. image path may contain unsupported characters")
            analysis.append("3. image file may be corrupted")
            if concat_file.exists():
                analysis.append("4. verify concat file content and referenced file paths")
        elif "No such file or directory" in error_message:
            analysis.append("1. input file path does not exist")
            analysis.append("2. validate all file paths in concat and ffmpeg command")
        elif "Permission denied" in error_message:
            analysis.append("1. permission issue on input/output paths")
        else:
            analysis.append("1. unknown ffmpeg error")
            analysis.append("2. check ffmpeg installation and codec support")

        analysis.append("\nGeneral checks:")
        analysis.append("1. ffmpeg -version")
        analysis.append("2. verify output directory is writable")
        return "\n".join(analysis)

    def _merge_audio_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        expected_duration: float,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        progress_range: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(str(output_dir), os.W_OK):
            raise PermissionError(f"Output directory is not writable: {output_dir}")

        cmd = [
            settings.FFMPEG_PATH,
            "-y",
            "-progress",
            "pipe:1",
            "-nostats",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            output_path,
        ]

        self._run_ffmpeg_with_progress(
            cmd=cmd,
            total_duration_sec=max(0.1, expected_duration),
            stage_message="Merging audio",
            progress_callback=progress_callback,
            progress_range=progress_range,
        )

    def extract_audio_info(self, audio_path: str) -> Dict[str, Any]:
        try:
            probe = ffmpeg.probe(audio_path)
            audio_stream = next(
                (stream for stream in probe["streams"] if stream.get("codec_type") == "audio"),
                None,
            )
            if not audio_stream:
                raise Exception("No audio stream found")

            return {
                "duration": float(audio_stream.get("duration", 0) or 0),
                "codec": audio_stream.get("codec_name", "unknown"),
                "bit_rate": audio_stream.get("bit_rate", "unknown"),
                "sample_rate": audio_stream.get("sample_rate", "unknown"),
                "channels": audio_stream.get("channels", "unknown"),
            }
        except Exception as exc:
            raise Exception(f"Failed to extract audio info: {exc}")

    def validate_video(self, video_path: str) -> bool:
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next(
                (stream for stream in probe["streams"] if stream.get("codec_type") == "video"),
                None,
            )
            return video_stream is not None
        except Exception:
            return False

    def get_video_duration(self, video_path: str) -> float:
        try:
            probe = ffmpeg.probe(video_path)
            return float(probe["format"].get("duration", 0) or 0)
        except Exception as exc:
            raise Exception(f"Failed to get video duration: {exc}")

    def create_preview_video(
        self,
        image_paths: List[str],
        timeline: List[Dict[str, Any]],
        output_path: str,
        resolution: str = "640x360",
        fps: int = 15,
    ) -> str:
        return self.create_video_from_timeline(
            image_paths=image_paths,
            timeline=timeline,
            audio_path=None,
            output_path=output_path,
            resolution=resolution,
            fps=fps,
        )

    def convert_video_format(
        self,
        input_path: str,
        output_path: str,
        format: str = "mp4",
    ) -> str:
        try:
            (
                ffmpeg
                .input(input_path)
                .output(output_path, vcodec="libx264", acodec="aac")
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            return output_path
        except ffmpeg.Error as exc:
            error_message = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
            raise Exception(f"Video format conversion failed: {error_message}")
