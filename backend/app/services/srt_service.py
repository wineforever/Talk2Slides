import pysrt
import re
from typing import List, Dict, Any
from datetime import timedelta

class SRTService:
    """SRT字幕处理服务"""
    
    def parse_srt(self, srt_path: str) -> List[Dict[str, Any]]:
        """解析SRT字幕文件
        
        Args:
            srt_path: SRT文件路径
            
        Returns:
            字幕片段列表，每个片段包含开始时间、结束时间和文本
        """
        try:
            subs = pysrt.open(srt_path, encoding="utf-8")
            subtitles = []
            
            for sub in subs:
                subtitle = {
                    "index": sub.index,
                    "start": sub.start.ordinal / 1000.0,
                    "end": sub.end.ordinal / 1000.0,
                    "duration": (sub.end.ordinal - sub.start.ordinal) / 1000.0,
                    "text": sub.text.strip(),
                    "raw_start": str(sub.start),
                    "raw_end": str(sub.end)
                }
                subtitles.append(subtitle)
            
            return subtitles
            
        except Exception as e:
            raise Exception(f"SRT解析失败: {str(e)}")
    
    def validate_srt(self, srt_path: str) -> bool:
        """验证SRT文件是否有效"""
        try:
            subs = pysrt.open(srt_path, encoding="utf-8")
            return len(subs) > 0
        except:
            return False
    
    def get_total_duration(self, srt_path: str) -> float:
        """获取字幕总时长（秒）"""
        try:
            subs = pysrt.open(srt_path, encoding="utf-8")
            if not subs:
                return 0.0
            
            first_start = subs[0].start.ordinal / 1000.0
            last_end = subs[-1].end.ordinal / 1000.0
            return last_end - first_start
            
        except Exception as e:
            raise Exception(f"获取字幕时长失败: {str(e)}")
    
    def merge_consecutive_subtitles(
        self, 
        subtitles: List[Dict[str, Any]], 
        max_gap: float = 0.5
    ) -> List[Dict[str, Any]]:
        """合并连续的字幕片段
        
        Args:
            subtitles: 字幕片段列表
            max_gap: 最大允许间隔（秒），小于此间隔的片段将被合并
            
        Returns:
            合并后的字幕片段列表
        """
        if not subtitles:
            return []
        
        merged = []
        current = subtitles[0].copy()
        
        for next_sub in subtitles[1:]:
            gap = next_sub["start"] - current["end"]
            
            if gap <= max_gap:
                # 合并字幕
                current["end"] = next_sub["end"]
                current["duration"] = current["end"] - current["start"]
                current["text"] = current["text"] + " " + next_sub["text"]
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
        filler_patterns: str
    ) -> List[Dict[str, Any]]:
        """对字幕进行合并与清洗，生成更稳健的对齐片段

        Args:
            subtitles: 原始字幕片段列表
            merge_gap_sec: 允许合并的最大时间间隔
            min_chars: 最小文本长度阈值
            min_duration_sec: 最小时长阈值
            max_chars: 合并后最大文本长度
            filler_patterns: 口头禅/填充词正则

        Returns:
            预处理后的片段列表
        """
        if not subtitles:
            return []

        filler_re = None
        if filler_patterns:
            filler_re = re.compile(f"({filler_patterns})")

        def _normalize_text(text: str) -> str:
            text = text.replace("\u3000", " ")
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        def _clean_text(text: str) -> str:
            text = _normalize_text(text)
            if filler_re:
                text = filler_re.sub("", text)
                text = _normalize_text(text)
            return text

        def _is_weak(text: str) -> bool:
            if not text:
                return True
            if len(text) < min_chars:
                return True
            core = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
            return len(core) == 0

        segments: List[Dict[str, Any]] = []
        current = None

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
                    "weak": weak
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
                if not (len(current.get("text", "")) < min_chars and current.get("duration", 0.0) < min_duration_sec):
                    current.pop("weak", None)
                    segments.append(current)

                current = {
                    "start": sub["start"],
                    "end": sub["end"],
                    "duration": sub["end"] - sub["start"],
                    "text": cleaned_text,
                    "raw_text": raw_text,
                    "source_indices": [sub.get("index")],
                    "weak": weak
                }

        if current is not None:
            if not (len(current.get("text", "")) < min_chars and current.get("duration", 0.0) < min_duration_sec):
                current.pop("weak", None)
                segments.append(current)

        return segments
    

    
    def create_timeline(
        self, 
        subtitles: List[Dict[str, Any]], 
        slide_indices: List[int]
    ) -> List[Dict[str, Any]]:
        """创建时间轴映射
        
        Args:
            subtitles: 字幕片段列表
            slide_indices: 每个字幕片段对应的幻灯片索引列表
            
        Returns:
            时间轴映射列表，每个元素包含开始时间、结束时间和幻灯片索引
        """
        if len(subtitles) != len(slide_indices):
            raise ValueError("字幕片段数量与幻灯片索引数量不匹配")
        
        timeline = []
        for sub, slide_idx in zip(subtitles, slide_indices):
            timeline.append({
                "start": sub["start"],
                "end": sub["end"],
                "slide_index": slide_idx,
                "subtitle_text": sub["text"][:100]  # 只保留前100个字符
            })
        
        return timeline
    
    def export_to_json(self, subtitles: List[Dict[str, Any]], output_path: str):
        """将字幕数据导出为JSON文件"""
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(subtitles, f, ensure_ascii=False, indent=2)
