import json
import re
from typing import Any, Dict, List, Optional

import pysrt


class SRTService:
    """SRT subtitle processing service."""

    _TRIM_PUNCT_PATTERN = re.compile(r"^[\s,\.\uFF0C\u3002]+|[\s,\.\uFF0C\u3002]+$")

    def normalize_subtitle_text(self, text: str) -> str:
        """Remove line breaks and trim leading/trailing comma/period (CN+EN)."""
        normalized = str(text or "")
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("\n", "")
        normalized = normalized.replace("\u3000", " ")
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = self._TRIM_PUNCT_PATTERN.sub("", normalized).strip()
        return normalized

    def parse_srt(self, srt_path: str, persist_normalized: bool = False) -> List[Dict[str, Any]]:
        """
        Parse an SRT file.

        Args:
            srt_path: SRT file path.
            persist_normalized: write normalized text back to the same SRT file.
        """
        try:
            subs = pysrt.open(srt_path, encoding="utf-8")
            subtitles: List[Dict[str, Any]] = []
            changed = False

            for sub in subs:
                cleaned_text = self.normalize_subtitle_text(sub.text)
                if sub.text != cleaned_text:
                    sub.text = cleaned_text
                    changed = True

                subtitle = {
                    "index": sub.index,
                    "start": sub.start.ordinal / 1000.0,
                    "end": sub.end.ordinal / 1000.0,
                    "duration": (sub.end.ordinal - sub.start.ordinal) / 1000.0,
                    "text": cleaned_text,
                    "raw_start": str(sub.start),
                    "raw_end": str(sub.end),
                }
                subtitles.append(subtitle)

            if persist_normalized and changed:
                subs.save(srt_path, encoding="utf-8")

            return subtitles
        except Exception as exc:
            raise Exception(f"SRT parsing failed: {exc}")

    def validate_srt(self, srt_path: str) -> bool:
        """Validate whether an SRT file can be parsed."""
        try:
            subs = pysrt.open(srt_path, encoding="utf-8")
            return len(subs) > 0
        except Exception:
            return False

    def get_total_duration(self, srt_path: str) -> float:
        """Get total subtitle duration (seconds)."""
        try:
            subs = pysrt.open(srt_path, encoding="utf-8")
            if not subs:
                return 0.0

            first_start = subs[0].start.ordinal / 1000.0
            last_end = subs[-1].end.ordinal / 1000.0
            return last_end - first_start
        except Exception as exc:
            raise Exception(f"Failed to get subtitle duration: {exc}")

    def merge_consecutive_subtitles(
        self,
        subtitles: List[Dict[str, Any]],
        max_gap: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Merge neighboring subtitle segments with small gaps."""
        if not subtitles:
            return []

        merged: List[Dict[str, Any]] = []
        current = subtitles[0].copy()

        for next_sub in subtitles[1:]:
            gap = next_sub["start"] - current["end"]
            if gap <= max_gap:
                current["end"] = next_sub["end"]
                current["duration"] = current["end"] - current["start"]
                current["text"] = (current["text"] + " " + next_sub["text"]).strip()
                current["raw_end"] = next_sub["raw_end"]
            else:
                merged.append(current)
                current = next_sub.copy()

        merged.append(current)
        return merged

    def preprocess_subtitles(
        self,
        subtitles: List[Dict[str, Any]],
        merge_gap_sec: float,
        min_chars: int,
        min_duration_sec: float,
        max_chars: int,
        filler_patterns: str,
    ) -> List[Dict[str, Any]]:
        """Merge and clean subtitle chunks for robust alignment."""
        if not subtitles:
            return []

        filler_re = re.compile(f"({filler_patterns})") if filler_patterns else None

        def _normalize_text(text: str) -> str:
            text = str(text or "").replace("\u3000", " ")
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        def _clean_text(text: str) -> str:
            cleaned = _normalize_text(text)
            if filler_re:
                cleaned = filler_re.sub("", cleaned)
                cleaned = _normalize_text(cleaned)
            return cleaned

        def _is_weak(text: str) -> bool:
            if not text:
                return True
            if len(text) < min_chars:
                return True
            core = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
            return len(core) == 0

        segments: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None

        for sub in subtitles:
            raw_text = _normalize_text(sub.get("text", ""))
            cleaned_text = _clean_text(sub.get("text", ""))
            weak = _is_weak(cleaned_text)

            if current is None:
                current = {
                    "start": sub["start"],
                    "end": sub["end"],
                    "duration": sub["end"] - sub["start"],
                    "text": cleaned_text,
                    "raw_text": raw_text,
                    "source_indices": [sub.get("index")],
                    "weak": weak,
                }
                continue

            gap = sub["start"] - current["end"]
            current_len = len(current.get("text", ""))
            current_duration = current.get("duration", 0.0)
            should_merge = (
                gap <= merge_gap_sec
                or weak
                or current.get("weak", False)
                or current_len < min_chars
                or current_duration < min_duration_sec
            )

            merged_text = (current.get("text", "") + " " + cleaned_text).strip()
            if should_merge and (len(merged_text) <= max_chars or current_len == 0):
                current["end"] = sub["end"]
                current["duration"] = current["end"] - current["start"]
                current["text"] = merged_text
                current["raw_text"] = (current.get("raw_text", "") + " " + raw_text).strip()
                current["source_indices"].append(sub.get("index"))
                current["weak"] = _is_weak(current["text"])
            else:
                if not (
                    len(current.get("text", "")) < min_chars
                    and current.get("duration", 0.0) < min_duration_sec
                ):
                    current.pop("weak", None)
                    segments.append(current)

                current = {
                    "start": sub["start"],
                    "end": sub["end"],
                    "duration": sub["end"] - sub["start"],
                    "text": cleaned_text,
                    "raw_text": raw_text,
                    "source_indices": [sub.get("index")],
                    "weak": weak,
                }

        if current is not None:
            if not (
                len(current.get("text", "")) < min_chars
                and current.get("duration", 0.0) < min_duration_sec
            ):
                current.pop("weak", None)
                segments.append(current)

        return segments

    def create_timeline(
        self,
        subtitles: List[Dict[str, Any]],
        slide_indices: List[int],
    ) -> List[Dict[str, Any]]:
        """Build a timeline mapping subtitle segments to slide indices."""
        if len(subtitles) != len(slide_indices):
            raise ValueError("Subtitle segment count does not match slide index count")

        timeline: List[Dict[str, Any]] = []
        for sub, slide_idx in zip(subtitles, slide_indices):
            timeline.append(
                {
                    "start": sub["start"],
                    "end": sub["end"],
                    "slide_index": slide_idx,
                    "subtitle_text": sub["text"][:100],
                }
            )

        return timeline

    def export_to_json(self, subtitles: List[Dict[str, Any]], output_path: str):
        """Export subtitle data to JSON."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(subtitles, f, ensure_ascii=False, indent=2)
