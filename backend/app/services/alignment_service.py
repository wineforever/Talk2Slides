import numpy as np
import re
from typing import List, Dict, Any, Tuple, Optional, Set
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import threading

from app.core.config import settings

logger = logging.getLogger(__name__)

class AlignmentService:
    """璇箟瀵归綈鏈嶅姟"""

    _MODEL_CACHE: Dict[Tuple[str, Optional[str]], SentenceTransformer] = {}
    _MODEL_CACHE_LOCK = threading.Lock()
    _SIGNAL_TOKEN_RE = re.compile(r"[A-Za-z]*\d+[A-Za-z0-9]*|\d+(?:\.\d+)?%?|[A-Za-z]{2,}")
    
    def __init__(self, model_name: str = None):
        """鍒濆鍖栧榻愭湇鍔?        
        Args:
            model_name: sentence-transformers妯″瀷鍚嶇О
        """
        self.model_name = model_name or settings.SENTENCE_TRANSFORMER_MODEL
        self.model_device = (getattr(settings, "SENTENCE_TRANSFORMER_DEVICE", "") or "").strip()
        self.model = None
        self.last_alignment_report: Dict[str, Any] = {}
        # 寤惰繜鍔犺浇妯″瀷锛屽湪绗竴娆′娇鐢ㄦ椂鍔犺浇
    
    def _load_model(self):
        """鍔犺浇棰勮缁冩ā鍨?"""
        try:
            cache_key = (self.model_name, self.model_device or None)
            with self._MODEL_CACHE_LOCK:
                cached_model = self._MODEL_CACHE.get(cache_key)
                if cached_model is not None:
                    self.model = cached_model
                    logger.info(f"澶嶇敤宸茬紦瀛樻ā鍨? {self.model_name} device={self.model_device or 'default'}")
                    return

            logger.info(f"姝ｅ湪鍔犺浇妯″瀷: {self.model_name} device={self.model_device or 'default'}")
            if self.model_device:
                loaded_model = SentenceTransformer(self.model_name, device=self.model_device)
            else:
                loaded_model = SentenceTransformer(self.model_name)
            self.model = loaded_model
            with self._MODEL_CACHE_LOCK:
                self._MODEL_CACHE[cache_key] = loaded_model
            logger.info("妯″瀷鍔犺浇瀹屾垚")
        except Exception as e:
            logger.error(f"妯″瀷鍔犺浇澶辫触: {str(e)}")
            raise Exception(f"鏃犳硶鍔犺浇妯″瀷 {self.model_name}: {str(e)}")
    
    def align_slides_with_subtitles(
        self,
        slides: List[Dict[str, Any]],
        subtitles: List[Dict[str, Any]],
        similarity_threshold: float = 0.5,
        min_display_duration: float = 2.0,
        align_max_backtrack: int = None,
        align_max_forward_jump: int = None,
        align_switch_penalty: float = None,
        align_backtrack_penalty: float = None,
        align_forward_jump_penalty: float = None,
        align_switch_delay_ms: Optional[int] = None,
        align_enforce_no_revisit: Optional[bool] = None,
        align_min_slide_usage_ratio: Optional[float] = None,
        enforce_sequential: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """将幻灯片与字幕进行语义对齐并输出时间轴。"""
        self.last_alignment_report = {}

        if not slides:
            logger.error("幻灯片列表为空，PPT解析可能失败")
            return []
        if not subtitles:
            logger.error("字幕列表为空，SRT解析可能失败")
            return []

        if self.model is None:
            self._load_model()

        sequential_mode = settings.ALIGN_ENFORCE_SEQUENTIAL if enforce_sequential is None else bool(enforce_sequential)
        require_full_coverage = settings.ALIGN_REQUIRE_FULL_COVERAGE if sequential_mode else False
        keep_short_segments = settings.ALIGN_KEEP_SHORT_SEGMENTS_FOR_COVERAGE if sequential_mode else False

        max_backtrack = settings.ALIGN_MAX_BACKTRACK if align_max_backtrack is None else align_max_backtrack
        max_forward_jump = settings.ALIGN_MAX_FORWARD_JUMP if align_max_forward_jump is None else align_max_forward_jump
        switch_penalty = settings.ALIGN_SWITCH_PENALTY if align_switch_penalty is None else align_switch_penalty
        backtrack_penalty = settings.ALIGN_BACKTRACK_PENALTY if align_backtrack_penalty is None else align_backtrack_penalty
        forward_jump_penalty = settings.ALIGN_FORWARD_JUMP_PENALTY if align_forward_jump_penalty is None else align_forward_jump_penalty
        switch_delay_ms = settings.ALIGN_SWITCH_DELAY_MS if align_switch_delay_ms is None else int(align_switch_delay_ms)
        enforce_no_revisit = settings.ALIGN_ENFORCE_NO_REVISIT if align_enforce_no_revisit is None else bool(align_enforce_no_revisit)
        min_slide_usage_ratio = (
            settings.ALIGN_MIN_SLIDE_USAGE_RATIO
            if align_min_slide_usage_ratio is None
            else float(align_min_slide_usage_ratio)
        )
        switch_delay_ms = int(min(max(switch_delay_ms, 0), 2000))
        min_slide_usage_ratio = float(min(max(min_slide_usage_ratio, 0.0), 1.0))
        switch_delay_sec = switch_delay_ms / 1000.0
        pause_threshold_sec = float(max(0.0, settings.ALIGN_PAUSE_SWITCH_THRESHOLD_SEC))

        semantic_weight = float(max(0.0, settings.ALIGN_SEMANTIC_WEIGHT))
        lexical_weight = float(max(0.0, settings.ALIGN_LEXICAL_WEIGHT))
        numeric_weight = float(max(0.0, settings.ALIGN_NUMERIC_WEIGHT))
        total_weight = semantic_weight + lexical_weight + numeric_weight
        if total_weight <= 0:
            semantic_weight, lexical_weight, numeric_weight = 1.0, 0.0, 0.0
            total_weight = 1.0
        semantic_weight /= total_weight
        lexical_weight /= total_weight
        numeric_weight /= total_weight

        logger.info(
            "开始语义对齐: "
            f"slides={len(slides)}, subtitles={len(subtitles)}, threshold={similarity_threshold}, "
            f"switch_delay_ms={switch_delay_ms}, sequential={sequential_mode}, "
            f"no_revisit={enforce_no_revisit}, min_usage_ratio={min_slide_usage_ratio:.2f}"
        )

        slide_texts = self._build_slide_alignment_texts(slides)
        subtitle_texts = [
            (subtitle.get("text") or "").strip() or (subtitle.get("raw_text") or "").strip()
            for subtitle in subtitles
        ]

        slide_embeddings = self.model.encode(slide_texts, convert_to_numpy=True)
        subtitle_embeddings = self.model.encode(subtitle_texts, convert_to_numpy=True)
        semantic_similarity_matrix = cosine_similarity(subtitle_embeddings, slide_embeddings)
        lexical_similarity_matrix = self._compute_lexical_similarity_matrix(slide_texts, subtitle_texts)
        numeric_bonus_matrix = self._compute_numeric_bonus_matrix(slide_texts, subtitle_texts)

        similarity_matrix = (
            semantic_weight * semantic_similarity_matrix
            + lexical_weight * lexical_similarity_matrix
            + numeric_weight * numeric_bonus_matrix
        )

        max_similarities = similarity_matrix.max(axis=1) if similarity_matrix.size else np.array([])
        avg_similarity = float(max_similarities.mean()) if len(max_similarities) > 0 else 0.0
        matches_above_threshold = int((max_similarities >= similarity_threshold).sum()) if len(max_similarities) > 0 else 0
        p10 = float(np.percentile(max_similarities, 10)) if len(max_similarities) > 0 else 0.0
        p50 = float(np.percentile(max_similarities, 50)) if len(max_similarities) > 0 else 0.0
        p90 = float(np.percentile(max_similarities, 90)) if len(max_similarities) > 0 else 0.0

        logger.info(
            "相似度统计: "
            f"avg={avg_similarity:.3f}, P10={p10:.3f}, P50={p50:.3f}, P90={p90:.3f}, "
            f"matches={matches_above_threshold}/{len(subtitles)}"
        )

        if sequential_mode:
            path = self._align_strict_sequential(similarity_matrix=similarity_matrix)
        else:
            if enforce_no_revisit:
                path = self._align_no_revisit_with_coverage(
                    similarity_matrix=similarity_matrix,
                    min_usage_ratio=min_slide_usage_ratio
                )
                if not path:
                    logger.warning("不可回页对齐失败，降级为更宽松的不可回页策略")
                    path = self._align_no_revisit_with_coverage(
                        similarity_matrix=similarity_matrix,
                        min_usage_ratio=0.0
                    )
                if not path:
                    logger.warning("不可回页策略仍失败，回退到严格顺序策略")
                    path = self._align_strict_sequential(similarity_matrix=similarity_matrix)
            else:
                path = self._align_with_viterbi(
                    similarity_matrix=similarity_matrix,
                    max_backtrack=max_backtrack,
                    max_forward_jump=max_forward_jump,
                    switch_penalty=switch_penalty,
                    backtrack_penalty=backtrack_penalty,
                    forward_jump_penalty=forward_jump_penalty
                )

        switch_times = self._build_switch_time_map(
            path=path,
            subtitles=subtitles,
            switch_delay_sec=switch_delay_sec,
            pause_threshold_sec=pause_threshold_sec
        )

        timeline = self._apply_min_duration_constraint(
            path=path,
            subtitles=subtitles,
            min_duration=min_display_duration,
            keep_short_segments=keep_short_segments,
            switch_times=switch_times
        )

        if (not timeline) and sequential_mode and require_full_coverage:
            logger.warning("主对齐未生成有效时间轴，启用顺序全覆盖兜底策略")
            timeline = self._create_uniform_timeline_fallback(
                slides=slides,
                subtitles=subtitles,
                min_display_duration=min_display_duration,
                force_full_coverage=True
            )

        self.last_alignment_report = self._build_alignment_report(
            path=path,
            timeline=timeline,
            similarity_matrix=similarity_matrix,
            subtitles=subtitles,
            similarity_threshold=similarity_threshold,
            min_display_duration=min_display_duration,
            switch_times=switch_times,
            switch_delay_ms=switch_delay_ms,
            pause_threshold_sec=pause_threshold_sec,
            scoring_weights={
                "semantic": semantic_weight,
                "lexical": lexical_weight,
                "numeric": numeric_weight
            },
            coverage_policy={
                "enforce_no_revisit": bool(enforce_no_revisit),
                "min_slide_usage_ratio": float(min_slide_usage_ratio),
                "sequential_mode": bool(sequential_mode)
            }
        )

        if not timeline:
            logger.warning("语义对齐失败，没有生成任何时间片段")
            logger.error(f"幻灯片数: {len(slides)}, 片段数: {len(subtitles)}")
        else:
            logger.info(f"语义对齐成功，生成{len(timeline)}个时间片段")
            logger.info(
                f"质量评级: {self.last_alignment_report.get('summary', {}).get('grade', 'N/A')} "
                f"(score={self.last_alignment_report.get('summary', {}).get('health_score', 'N/A')})"
            )

        return timeline

    def _build_slide_alignment_texts(self, slides: List[Dict[str, Any]]) -> List[str]:
        """鏋勫缓鐢ㄤ簬瀵归綈鐨勫够鐏墖鏂囨湰锛堜紭鍏堜娇鐢ㄥ娉級"""
        texts = []
        notes_count = 0
        empty_count = 0

        for slide in slides:
            notes = (slide.get("notes") or "").strip()
            if notes:
                text = notes
                notes_count += 1
            else:
                title = (slide.get("title") or "").strip()
                content = (slide.get("content") or "").strip()
                parts = [p for p in [title, content] if p]
                text = " ".join(parts).strip()
                if not text:
                    text = (slide.get("full_text") or "").strip()

            if not text:
                empty_count += 1
            texts.append(text)

        if slides:
            logger.info(f"瀵归綈鏂囨湰鏋勫缓: 澶囨敞浼樺厛浣跨敤{notes_count}/{len(slides)}椤? 绌烘枃鏈〉={empty_count}")

        return texts

    def get_last_alignment_report(self) -> Dict[str, Any]:
        """返回最近一次对齐的质量报告。"""
        return dict(self.last_alignment_report or {})

    def _compute_lexical_similarity_matrix(
        self,
        slide_texts: List[str],
        subtitle_texts: List[str]
    ) -> np.ndarray:
        """计算词面相似度矩阵（TF-IDF char ngram）。"""
        rows = len(subtitle_texts)
        cols = len(slide_texts)
        if rows == 0 or cols == 0:
            return np.zeros((rows, cols), dtype=float)

        try:
            corpus = [(t or "").strip() for t in (slide_texts + subtitle_texts)]
            if not any(corpus):
                return np.zeros((rows, cols), dtype=float)

            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
            tfidf = vectorizer.fit_transform(corpus)
            slide_tfidf = tfidf[:cols]
            subtitle_tfidf = tfidf[cols:]
            return cosine_similarity(subtitle_tfidf, slide_tfidf)
        except Exception as exc:
            logger.warning(f"词面相似度计算失败，回退为0矩阵: {exc}")
            return np.zeros((rows, cols), dtype=float)

    def _extract_signal_tokens(self, text: str) -> Set[str]:
        """提取数字/英文信号token，用于命中加权。"""
        if not text:
            return set()

        tokens: Set[str] = set()
        for token in self._SIGNAL_TOKEN_RE.findall(text.upper()):
            cleaned = token.strip()
            if not cleaned:
                continue
            if len(cleaned) < 2 and not any(ch.isdigit() for ch in cleaned):
                continue
            tokens.add(cleaned)
        return tokens

    def _compute_numeric_bonus_matrix(
        self,
        slide_texts: List[str],
        subtitle_texts: List[str]
    ) -> np.ndarray:
        """计算数字/英文token命中矩阵。"""
        rows = len(subtitle_texts)
        cols = len(slide_texts)
        matrix = np.zeros((rows, cols), dtype=float)
        if rows == 0 or cols == 0:
            return matrix

        slide_token_sets = [self._extract_signal_tokens(text) for text in slide_texts]
        subtitle_token_sets = [self._extract_signal_tokens(text) for text in subtitle_texts]

        for row_idx, sub_tokens in enumerate(subtitle_token_sets):
            if not sub_tokens:
                continue
            denominator = float(max(1, len(sub_tokens)))
            for col_idx, slide_tokens in enumerate(slide_token_sets):
                if not slide_tokens:
                    continue
                overlap = len(sub_tokens.intersection(slide_tokens))
                if overlap > 0:
                    matrix[row_idx, col_idx] = overlap / denominator
        return matrix

    def _build_switch_time_map(
        self,
        path: List[Tuple[int, int, float]],
        subtitles: List[Dict[str, Any]],
        switch_delay_sec: float,
        pause_threshold_sec: float
    ) -> Dict[int, float]:
        """为路径中的每个切页点生成更贴口播的切页时间。"""
        switch_times: Dict[int, float] = {}
        if not path or len(path) < 2 or not subtitles:
            return switch_times

        epsilon = 1e-3
        for path_idx in range(1, len(path)):
            prev_seg_idx, prev_slide, _ = path[path_idx - 1]
            curr_seg_idx, curr_slide, _ = path[path_idx]
            if curr_slide == prev_slide:
                continue

            prev_sub = subtitles[prev_seg_idx]
            curr_sub = subtitles[curr_seg_idx]
            prev_end = float(prev_sub.get("end", prev_sub.get("start", 0.0)))
            curr_start = float(curr_sub.get("start", prev_end))
            curr_end = float(curr_sub.get("end", curr_start))
            gap = curr_start - prev_end

            if gap >= pause_threshold_sec and pause_threshold_sec > 0:
                boundary = prev_end + gap * 0.5
            else:
                boundary = curr_start + switch_delay_sec

            lower = max(prev_end, curr_start)
            upper = max(lower + epsilon, curr_end)
            boundary = min(max(boundary, lower), upper)
            switch_times[path_idx] = boundary

        return switch_times

    def _build_alignment_report(
        self,
        path: List[Tuple[int, int, float]],
        timeline: List[Dict[str, Any]],
        similarity_matrix: np.ndarray,
        subtitles: List[Dict[str, Any]],
        similarity_threshold: float,
        min_display_duration: float,
        switch_times: Dict[int, float],
        switch_delay_ms: int,
        pause_threshold_sec: float,
        scoring_weights: Dict[str, float],
        coverage_policy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成对齐质量报告，用于评估“变准还是变差”。"""
        report: Dict[str, Any] = {
            "summary": {
                "grade": "C",
                "health_score": 0,
                "path_segments": len(path),
                "timeline_segments": len(timeline),
                "total_slides": int(similarity_matrix.shape[1]) if similarity_matrix is not None and similarity_matrix.ndim == 2 else 0
            },
            "scoring": {
                "similarity_threshold": float(similarity_threshold),
                "weights": {
                    "semantic": float(scoring_weights.get("semantic", 0.0)),
                    "lexical": float(scoring_weights.get("lexical", 0.0)),
                    "numeric": float(scoring_weights.get("numeric", 0.0))
                }
            },
            "timing_policy": {
                "switch_delay_ms": int(switch_delay_ms),
                "pause_threshold_sec": float(pause_threshold_sec)
            },
            "coverage_policy": {
                "enforce_no_revisit": bool(coverage_policy.get("enforce_no_revisit", False)),
                "min_slide_usage_ratio": float(coverage_policy.get("min_slide_usage_ratio", 0.0)),
                "sequential_mode": bool(coverage_policy.get("sequential_mode", False))
            },
            "confidence": {},
            "boundary_quality": {},
            "transition_stats": {},
            "low_confidence_segments": [],
            "suspicious_boundaries": []
        }

        if not path or similarity_matrix is None or similarity_matrix.size == 0:
            return report

        low_conf_margin = float(max(0.0, settings.ALIGN_LOW_CONFIDENCE_MARGIN))
        boundary_cross_min = float(settings.ALIGN_BOUNDARY_CROSS_MIN)

        margins: List[float] = []
        low_conf_segments: List[Dict[str, Any]] = []
        suspicious_boundaries: List[Dict[str, Any]] = []
        boundary_scores: List[float] = []

        for seg_idx, assigned_slide, _ in path:
            row = similarity_matrix[seg_idx]
            if row.size == 0:
                continue
            best_slide = int(np.argmax(row))
            best_score = float(row[best_slide])
            if row.size > 1:
                top2 = np.partition(row, -2)[-2:]
                second_score = float(min(top2[0], top2[1]))
            else:
                second_score = 0.0
            margin = float(best_score - second_score)
            margins.append(margin)

            if 0 <= assigned_slide < row.size:
                assigned_score = float(row[assigned_slide])
            else:
                assigned_score = best_score
            if margin < low_conf_margin:
                sub = subtitles[seg_idx]
                low_conf_segments.append({
                    "segment_index": int(seg_idx),
                    "start": float(sub.get("start", 0.0)),
                    "end": float(sub.get("end", 0.0)),
                    "assigned_slide": int(assigned_slide),
                    "assigned_score": assigned_score,
                    "best_slide": best_slide,
                    "best_score": best_score,
                    "margin": margin
                })

        for path_idx in range(1, len(path)):
            prev_seg_idx, prev_slide, _ = path[path_idx - 1]
            curr_seg_idx, curr_slide, _ = path[path_idx]
            if prev_slide == curr_slide:
                continue

            prev_row = similarity_matrix[prev_seg_idx]
            curr_row = similarity_matrix[curr_seg_idx]
            cross_strength = float(
                (prev_row[prev_slide] - curr_row[prev_slide]) +
                (curr_row[curr_slide] - prev_row[curr_slide])
            )
            boundary_scores.append(cross_strength)
            if cross_strength < boundary_cross_min:
                boundary_time = float(switch_times.get(path_idx, subtitles[curr_seg_idx].get("start", 0.0)))
                suspicious_boundaries.append({
                    "path_index": int(path_idx),
                    "time": boundary_time,
                    "from_slide": int(prev_slide),
                    "to_slide": int(curr_slide),
                    "cross_strength": cross_strength
                })

        confidence_avg = float(np.mean(margins)) if margins else 0.0
        confidence_p10 = float(np.percentile(margins, 10)) if margins else 0.0
        low_conf_ratio = float(len(low_conf_segments) / len(path)) if path else 1.0

        boundary_avg = float(np.mean(boundary_scores)) if boundary_scores else 0.0
        boundary_min = float(np.min(boundary_scores)) if boundary_scores else 0.0
        boundary_bad_ratio = float(len(suspicious_boundaries) / max(1, len(boundary_scores)))

        backtrack_count = 0
        forward_jump_count = 0
        jump_distances: List[int] = []
        revisit_violations = 0
        seen_slides: Set[int] = set()
        if path:
            seen_slides.add(int(path[0][1]))
        for idx in range(1, len(path)):
            delta = int(path[idx][1] - path[idx - 1][1])
            curr_slide = int(path[idx][1])
            prev_slide = int(path[idx - 1][1])
            if curr_slide != prev_slide and curr_slide in seen_slides:
                revisit_violations += 1
            seen_slides.add(curr_slide)
            if delta < 0:
                backtrack_count += 1
            elif delta > 0:
                forward_jump_count += 1
                jump_distances.append(delta)

        timeline_durations = [float(seg.get("duration", 0.0)) for seg in timeline]
        short_segments = [d for d in timeline_durations if d < min_display_duration]
        short_ratio = float(len(short_segments) / max(1, len(timeline_durations)))

        health_score = 100
        if confidence_avg < 0.05:
            health_score -= 25
        elif confidence_avg < 0.10:
            health_score -= 15
        if low_conf_ratio > 0.30:
            health_score -= 20
        elif low_conf_ratio > 0.15:
            health_score -= 10
        if boundary_bad_ratio > 0.40:
            health_score -= 20
        elif boundary_bad_ratio > 0.20:
            health_score -= 10
        if backtrack_count > 0:
            health_score -= min(15, backtrack_count * 2)
        if revisit_violations > 0:
            health_score -= min(25, revisit_violations * 5)
        if short_ratio > 0.25:
            health_score -= 10
        health_score = int(max(0, min(100, health_score)))

        if health_score >= 85:
            grade = "A"
        elif health_score >= 70:
            grade = "B"
        else:
            grade = "C"

        report["summary"] = {
            "grade": grade,
            "health_score": health_score,
            "path_segments": len(path),
            "timeline_segments": len(timeline),
            "total_slides": int(similarity_matrix.shape[1]),
            "unique_slides_used": len(seen_slides),
            "slide_coverage_ratio": float(len(seen_slides) / max(1, similarity_matrix.shape[1]))
        }
        report["confidence"] = {
            "avg_margin": confidence_avg,
            "p10_margin": confidence_p10,
            "low_confidence_ratio": low_conf_ratio,
            "low_confidence_count": len(low_conf_segments)
        }
        report["boundary_quality"] = {
            "avg_cross_strength": boundary_avg,
            "min_cross_strength": boundary_min,
            "suspicious_ratio": boundary_bad_ratio,
            "suspicious_count": len(suspicious_boundaries)
        }
        report["transition_stats"] = {
            "switches": max(0, len(path) - 1),
            "backtracks": backtrack_count,
            "revisit_violations": revisit_violations,
            "forward_jumps": forward_jump_count,
            "avg_forward_jump": float(np.mean(jump_distances)) if jump_distances else 0.0,
            "timeline_avg_duration": float(np.mean(timeline_durations)) if timeline_durations else 0.0,
            "timeline_short_ratio": short_ratio,
            "total_slides": int(similarity_matrix.shape[1]),
            "unique_slides_used": len(seen_slides),
            "slide_coverage_ratio": float(len(seen_slides) / max(1, similarity_matrix.shape[1]))
        }
        report["low_confidence_segments"] = low_conf_segments[:50]
        report["suspicious_boundaries"] = suspicious_boundaries[:50]
        return report

    def _align_no_revisit_with_coverage(
        self,
        similarity_matrix: np.ndarray,
        min_usage_ratio: float
    ) -> List[Tuple[int, int, float]]:
        """对齐约束：不可回到已讲过页面，并覆盖尽可能多页面。"""
        if similarity_matrix is None or similarity_matrix.size == 0:
            return []

        n_segments, n_slides = similarity_matrix.shape
        if n_segments == 0 or n_slides == 0:
            return []

        ratio = float(min(max(min_usage_ratio, 0.0), 1.0))
        target_count = int(np.ceil(n_slides * ratio))
        target_count = max(1, min(target_count, n_slides, n_segments))

        peak_scores = similarity_matrix.max(axis=0)
        anchor_segments = similarity_matrix.argmax(axis=0)
        ranked_slides = np.argsort(-peak_scores).tolist()
        selected_slides = set(int(idx) for idx in ranked_slides[:target_count])
        ordered_slides = sorted(
            selected_slides,
            key=lambda idx: (int(anchor_segments[idx]), -float(peak_scores[idx]), int(idx))
        )

        if not ordered_slides:
            return []

        reduced = similarity_matrix[:, ordered_slides]
        reduced_path = self._align_strict_sequential(similarity_matrix=reduced)
        if not reduced_path:
            return []

        mapped_path: List[Tuple[int, int, float]] = []
        for seg_idx, reduced_slide_idx, _ in reduced_path:
            real_slide_idx = int(ordered_slides[int(reduced_slide_idx)])
            mapped_path.append((int(seg_idx), real_slide_idx, float(similarity_matrix[seg_idx, real_slide_idx])))

        logger.info(
            "不可回页对齐完成: "
            f"target_slides={target_count}, used_slides={len(set(s for _, s, _ in mapped_path))}/{n_slides}"
        )
        return mapped_path

    def _align_with_viterbi(
        self,
        similarity_matrix: np.ndarray,
        max_backtrack: int,
        max_forward_jump: int,
        switch_penalty: float,
        backtrack_penalty: float,
        forward_jump_penalty: float
    ) -> List[Tuple[int, int, float]]:
        """鍙楅檺鍥為€€鐨勫崟璋冨簭鍒楀榻愶紙Viterbi/DP锛?"""
        if similarity_matrix is None or similarity_matrix.size == 0:
            return []

        n_segments, n_slides = similarity_matrix.shape
        if n_segments == 0 or n_slides == 0:
            return []

        dp = np.full((n_segments, n_slides), -np.inf, dtype=float)
        prev = np.full((n_segments, n_slides), -1, dtype=int)

        # 鍒濆鍖?
        dp[0, :] = similarity_matrix[0, :]

        for seg_idx in range(1, n_segments):
            for slide_idx in range(n_slides):
                start = max(0, slide_idx - max_forward_jump)
                end = min(n_slides - 1, slide_idx + max_backtrack)
                best_score = -np.inf
                best_prev = -1

                for prev_idx in range(start, end + 1):
                    prev_score = dp[seg_idx - 1, prev_idx]
                    if prev_score == -np.inf:
                        continue

                    penalty = 0.0
                    if prev_idx != slide_idx:
                        penalty += switch_penalty
                        if slide_idx > prev_idx:
                            penalty += (slide_idx - prev_idx) * forward_jump_penalty
                        else:
                            penalty += (prev_idx - slide_idx) * backtrack_penalty

                    score = prev_score + similarity_matrix[seg_idx, slide_idx] - penalty
                    if score > best_score:
                        best_score = score
                        best_prev = prev_idx

                dp[seg_idx, slide_idx] = best_score
                prev[seg_idx, slide_idx] = best_prev

        # 鍥炴函
        last_slide = int(np.argmax(dp[-1, :]))
        path = [last_slide]
        for seg_idx in range(n_segments - 1, 0, -1):
            last_slide = int(prev[seg_idx, last_slide])
            if last_slide < 0:
                last_slide = int(np.argmax(dp[seg_idx - 1, :]))
            path.append(last_slide)
        path.reverse()

        # 缁熻瀵归綈琛屼负
        switch_count = 0
        backtrack_count = 0
        max_backtrack_actual = 0
        for i in range(1, len(path)):
            if path[i] != path[i - 1]:
                switch_count += 1
            if path[i] < path[i - 1]:
                backtrack_count += 1
                max_backtrack_actual = max(max_backtrack_actual, path[i - 1] - path[i])

        logger.info(
            f"瀵归綈鍒囬〉缁熻: 鎬荤墖娈?{n_segments}, 鍒囬〉娆℃暟={switch_count}, 鍥為€€娆℃暟={backtrack_count}, 鏈€澶у洖閫€椤垫暟={max_backtrack_actual}"
        )

        return [(int(i), int(path[i]), float(similarity_matrix[i, path[i]])) for i in range(n_segments)]

    def _align_strict_sequential(
        self,
        similarity_matrix: np.ndarray
    ) -> List[Tuple[int, int, float]]:
        """涓ユ牸椤哄簭瀵归綈锛氫粠绗竴椤靛紑濮嬶紝浠呭厑璁稿仠鐣欐垨鍓嶈繘1椤碉紝涓旀渶缁堣鐩栧埌鏈€鍚庝竴椤点€?"""
        if similarity_matrix is None or similarity_matrix.size == 0:
            return []

        n_segments, n_slides = similarity_matrix.shape
        if n_segments == 0 or n_slides == 0:
            return []

        if n_segments < n_slides:
            logger.warning(
                f"涓ユ牸椤哄簭瀵归綈涓嶅彲琛? 瀛楀箷鐗囨鏁?{n_segments})灏戜簬骞荤伅鐗囨暟({n_slides})"
            )
            return []

        dp = np.full((n_segments, n_slides), -np.inf, dtype=float)
        prev = np.full((n_segments, n_slides), -1, dtype=int)

        # 寮哄埗棣栨浠庣涓€椤靛紑濮?
        dp[0, 0] = similarity_matrix[0, 0]

        for seg_idx in range(1, n_segments):
            for slide_idx in range(n_slides):
                # 鍙揪鎬х害鏉燂細鏈€澶氭瘡娈靛墠杩?椤?
                if slide_idx > seg_idx:
                    continue

                # 缁堟€佸彲杈炬€э細鍓╀綑鐗囨蹇呴』瓒冲璧板埌鏈€鍚庝竴椤?
                remaining_segments = (n_segments - 1) - seg_idx
                remaining_slides = (n_slides - 1) - slide_idx
                if remaining_segments < remaining_slides:
                    continue

                best_score = -np.inf
                best_prev = -1

                # 1) 鍋滅暀褰撳墠椤?
                stay_score = dp[seg_idx - 1, slide_idx]
                if stay_score != -np.inf:
                    score = stay_score + similarity_matrix[seg_idx, slide_idx]
                    if score > best_score:
                        best_score = score
                        best_prev = slide_idx

                # 2) 浠呭厑璁稿墠杩涗竴椤?
                if slide_idx > 0:
                    forward_score = dp[seg_idx - 1, slide_idx - 1]
                    if forward_score != -np.inf:
                        score = forward_score + similarity_matrix[seg_idx, slide_idx]
                        if score > best_score:
                            best_score = score
                            best_prev = slide_idx - 1

                dp[seg_idx, slide_idx] = best_score
                prev[seg_idx, slide_idx] = best_prev

        # 寮哄埗鏈€鍚庝竴娈佃惤鍦ㄦ渶鍚庝竴椤碉紝纭繚鍏ㄨ鐩?
        last_slide = n_slides - 1
        if dp[n_segments - 1, last_slide] == -np.inf:
            logger.warning("涓ユ牸椤哄簭瀵归綈澶辫触: 鏃犳硶鍦ㄦ湯娈佃鐩栧埌鏈€鍚庝竴椤?")
            return []

        path = [last_slide]
        for seg_idx in range(n_segments - 1, 0, -1):
            last_slide = int(prev[seg_idx, last_slide])
            if last_slide < 0:
                logger.warning("涓ユ牸椤哄簭瀵归綈鍥炴函澶辫触")
                return []
            path.append(last_slide)
        path.reverse()

        # 鏍￠獙锛氫笉鍏佽璺抽〉锛屽繀椤诲畬鏁磋鐩?
        max_jump = 0
        for i in range(1, len(path)):
            jump = path[i] - path[i - 1]
            if jump < 0 or jump > 1:
                logger.warning(f"涓ユ牸椤哄簭璺緞寮傚父: {path[i - 1]} -> {path[i]}")
                return []
            max_jump = max(max_jump, jump)

        covered = set(path)
        if settings.ALIGN_REQUIRE_FULL_COVERAGE and len(covered) < n_slides:
            logger.warning(
                f"涓ユ牸椤哄簭璺緞鏈鐩栧叏閮ㄩ〉闈? 瑕嗙洊{len(covered)}/{n_slides}"
            )
            return []

        logger.info(
            f"涓ユ牸椤哄簭瀵归綈鎴愬姛: 鐗囨={n_segments}, 椤甸潰={n_slides}, 鏈€澶ф杩?{max_jump}, 棣栨椤?{path[0]}, 鏈椤?{path[-1]}"
        )

        return [(int(i), int(path[i]), float(similarity_matrix[i, path[i]])) for i in range(n_segments)]
    
    def _generate_timeline_with_constraints(
        self,
        similarity_matrix: np.ndarray,
        subtitles: List[Dict[str, Any]],
        slides: List[Dict[str, Any]],
        similarity_threshold: float,
        min_display_duration: float
    ) -> List[Dict[str, Any]]:
        """搴旂敤绾︽潫鐢熸垚鏃堕棿杞存槧灏?
        
        鏀硅繘绠楁硶锛氱害鏉熷眬閮ㄦ渶浼樺尮閰?
        绾︽潫鏉′欢:
        1. 鏃堕棿鍖哄煙鍒掑垎锛氬紑濮嬪尯鍩燂紙鍓?0%锛夈€佹牳蹇冨尯鍩燂紙涓棿80%锛夈€佺粨鏉熷尯鍩燂紙鍚?0%锛?
        2. 椤甸潰闄愬埗锛氭牳蹇冨尯鍩熺姝㈠尮閰嶇涓€椤靛拰鏈€鍚庝竴椤?
        3. 璺宠浆璺濈闄愬埗锛氱浉閭荤墖娈垫渶澶ц烦杞?椤?
        4. 鐩镐技搴﹀繀椤婚珮浜庨槇鍊?
        5. 姣忎釜骞荤伅鐗囪嚦灏戝睍绀烘渶灏忔椂闀?
        """
        n_subtitles = len(subtitles)
        n_slides = len(slides)
        
        logger.info(f"浣跨敤绾︽潫灞€閮ㄦ渶浼樺尮閰嶇畻娉? {n_subtitles}涓瓧骞曠墖娈? {n_slides}寮犲够鐏墖")
        logger.info(f"鐩镐技搴﹂槇鍊? {similarity_threshold}, 鏈€灏忓睍绀烘椂闀? {min_display_duration}绉?")
        
        # 绠楁硶鍙傛暟
        start_region_ratio = 0.1  # 寮€濮嬪尯鍩熷崰鎬绘椂闀跨殑姣斾緥
        end_region_ratio = 0.1    # 缁撴潫鍖哄煙鍗犳€绘椂闀跨殑姣斾緥
        max_jump_distance = 5     # 鏈€澶ц烦杞〉鏁?
        
        logger.info(f"绠楁硶鍙傛暟: 寮€濮嬪尯鍩?{start_region_ratio*100:.0f}%, 缁撴潫鍖哄煙 {end_region_ratio*100:.0f}%, 鏈€澶ц烦杞?{max_jump_distance} 椤?")
        
        # 璁＄畻鏃堕棿杈圭晫
        total_duration = subtitles[-1]["end"] if subtitles else 0
        start_boundary = total_duration * start_region_ratio
        end_boundary = total_duration * (1 - end_region_ratio)
        
        logger.info(f"鏃堕棿杈圭晫: 鎬绘椂闀?{total_duration:.1f}s, 寮€濮嬭竟鐣?{start_boundary:.1f}s, 缁撴潫杈圭晫 {end_boundary:.1f}s")
        
        # 缁熻鍖哄煙鍒嗗竷
        region_counts = {"start": 0, "core": 0, "end": 0}
        
        # 璁板綍鐩镐技搴︾粺璁′俊鎭?
        best_similarities = []
        matched_count = 0
        no_match_count = 0
        
        # 绾︽潫搴旂敤缁熻
        constraint_stats = {
            "core_region_excluded": 0,  # 鏍稿績鍖哄煙鎺掗櫎绗竴椤?鏈€鍚庝竴椤电殑娆℃暟
            "jump_distance_limited": 0, # 璺宠浆璺濈闄愬埗鐨勬鏁?
            "candidates_filtered": 0,   # 鍊欓€夎杩囨护鐨勬鏁?
        }
        
        # 涓烘瘡涓瓧骞曠墖娈甸€夋嫨鏈€浣冲够鐏墖锛堝簲鐢ㄧ害鏉燂級
        path = []
        previous_slide = -1  # 璁板綍涓婁竴涓尮閰嶇殑骞荤伅鐗?
        
        for i in range(n_subtitles):
            subtitle = subtitles[i]
            current_time = subtitle["start"]
            
            # 纭畾褰撳墠鐗囨鎵€灞炲尯鍩?
            if current_time < start_boundary:
                region = "start"
            elif current_time > end_boundary:
                region = "end"
            else:
                region = "core"
            
            region_counts[region] += 1
            
            # 鍒濆鍖栧€欓€夊够鐏墖锛堟墍鏈夊够鐏墖锛?
            candidates = list(range(n_slides))
            candidate_count_before = len(candidates)
            
            # 绾︽潫1锛氭牳蹇冨尯鍩熺姝㈠尮閰嶇涓€椤靛拰鏈€鍚庝竴椤?
            if region == "core" and n_slides > 2:
                # 鎺掗櫎绗竴椤?(绱㈠紩0) 鍜屾渶鍚庝竴椤?(绱㈠紩n_slides-1)
                filtered_candidates = [j for j in candidates if j != 0 and j != n_slides-1]
                if filtered_candidates:
                    candidates = filtered_candidates
                    constraint_stats["core_region_excluded"] += 1
                    logger.debug(f"瀛楀箷鐗囨 {i} (鏍稿績鍖哄煙): 鎺掗櫎绗竴椤靛拰鏈€鍚庝竴椤碉紝鍊欓€変粠 {candidate_count_before} 鍑忓皯鍒?{len(candidates)}")
            
            # 绾︽潫2锛氳烦杞窛绂婚檺鍒讹紙濡傛灉鏈夊墠涓€涓尮閰嶏級
            if previous_slide >= 0 and candidates:
                candidates_before_jump = len(candidates)
                filtered_candidates = [
                    j for j in candidates 
                    if abs(j - previous_slide) <= max_jump_distance
                ]
                if filtered_candidates:
                    candidates = filtered_candidates
                    constraint_stats["jump_distance_limited"] += 1
                    logger.debug(f"瀛楀箷鐗囨 {i}: 璺宠浆璺濈闄愬埗锛屽€欓€変粠 {candidates_before_jump} 鍑忓皯鍒?{len(candidates)}")
                else:
                    # 濡傛灉娌℃湁鍊欓€夋弧瓒宠烦杞窛绂婚檺鍒讹紝鏀惧闄愬埗锛堜繚鎸佸綋鍓嶅€欓€夛級
                    logger.debug(f"瀛楀箷鐗囨 {i}: 璺宠浆璺濈闄愬埗杩囦弗锛屾棤鍊欓€夋弧瓒虫潯浠讹紝淇濇寔褰撳墠 {candidates_before_jump} 涓€欓€?")
            
            if len(candidates) < candidate_count_before:
                constraint_stats["candidates_filtered"] += 1
            
            # 鍦ㄥ€欓€夊够鐏墖涓壘鍒扮浉浼煎害鏈€楂樼殑
            best_slide_idx = -1
            best_similarity = -1.0
            
            for j in candidates:
                similarity = similarity_matrix[i, j]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_slide_idx = j
            
            best_similarities.append(best_similarity)
            
            # 妫€鏌ユ槸鍚﹁揪鍒伴槇鍊?
            if best_similarity >= similarity_threshold and best_slide_idx >= 0:
                path.append((i, best_slide_idx, best_similarity))
                matched_count += 1
                previous_slide = best_slide_idx  # 鏇存柊鍓嶄竴涓尮閰?
                logger.debug(f"瀛楀箷鐗囨 {i} ({region}鍖哄煙): 鍖归厤骞荤伅鐗?{best_slide_idx}, 鐩镐技搴?{best_similarity:.3f}")
            else:
                # 娌℃湁杈惧埌闃堝€肩殑鍖归厤锛屾爣璁颁负-1
                path.append((i, -1, best_similarity))
                no_match_count += 1
                logger.debug(f"瀛楀箷鐗囨 {i} ({region}鍖哄煙): 鏃犲尮閰?(鏈€浣崇浉浼煎害 {best_similarity:.3f} < 闃堝€?{similarity_threshold})")
        
        # 缁熻淇℃伅
        if best_similarities:
            avg_similarity = sum(best_similarities) / len(best_similarities)
            max_similarity = max(best_similarities)
            min_similarity = min(best_similarities)
            
            logger.info(f"鐩镐技搴︾粺璁? 骞冲潎 {avg_similarity:.3f}, 鏈€澶?{max_similarity:.3f}, 鏈€灏?{min_similarity:.3f}")
            logger.info(f"鍖归厤缁撴灉: {matched_count}/{n_subtitles} 涓墖娈佃揪鍒伴槇鍊? {no_match_count} 涓墖娈垫湭鍖归厤")
            logger.info(f"鍖哄煙鍒嗗竷: 寮€濮?{region_counts['start']}, 鏍稿績 {region_counts['core']}, 缁撴潫 {region_counts['end']}")
            logger.info(f"绾︽潫搴旂敤: 鏍稿績鍖哄煙鎺掗櫎 {constraint_stats['core_region_excluded']} 娆? "
                       f"璺宠浆闄愬埗 {constraint_stats['jump_distance_limited']} 娆? "
                       f"鍊欓€夎繃婊?{constraint_stats['candidates_filtered']} 娆?")
        
        # 澶勭悊鏈尮閰嶇殑鐗囨锛堜娇鐢ㄦ渶杩戠殑鏈夋晥鍖归厤锛?
        processed_path = self._handle_unmatched_segments(
            path, n_slides
        )
        
        # 搴旂敤鏈€灏忓睍绀烘椂闀跨害鏉?
        timeline = self._apply_min_duration_constraint(
            processed_path, subtitles, min_display_duration
        )
        
        # 璁板綍PPT璺宠浆缁熻淇℃伅
        if processed_path and len(processed_path) > 1:
            jump_count = 0
            forward_jumps = 0
            backward_jumps = 0
            same_slide_count = 0
            
            for idx in range(1, len(processed_path)):
                prev_slide = processed_path[idx-1][1]
                curr_slide = processed_path[idx][1]
                
                if curr_slide > prev_slide:
                    forward_jumps += 1
                    jump_count += 1
                elif curr_slide < prev_slide:
                    backward_jumps += 1
                    jump_count += 1
                else:
                    same_slide_count += 1
            
            total_transitions = len(processed_path) - 1
            logger.info(f"PPT璺宠浆缁熻: 鎬诲垏鎹?{total_transitions} 娆?")
            logger.info(f"  - 鍚戝墠璺宠浆: {forward_jumps} 娆?({forward_jumps/total_transitions*100:.1f}%)")
            logger.info(f"  - 鍚戝悗璺宠浆: {backward_jumps} 娆?({backward_jumps/total_transitions*100:.1f}%)")
            logger.info(f"  - 淇濇寔鍚屼竴椤? {same_slide_count} 娆?({same_slide_count/total_transitions*100:.1f}%)")
            
            # 璁板綍璺宠浆璺濈缁熻
            if jump_count > 0:
                jump_distances = []
                for idx in range(1, len(processed_path)):
                    prev_slide = processed_path[idx-1][1]
                    curr_slide = processed_path[idx][1]
                    if curr_slide != prev_slide:
                        jump_distances.append(abs(curr_slide - prev_slide))
                
                if jump_distances:
                    avg_jump_distance = sum(jump_distances) / len(jump_distances)
                    actual_max_jump = max(jump_distances)
                    logger.info(f"璺宠浆璺濈: 骞冲潎 {avg_jump_distance:.1f} 椤? 鏈€澶?{actual_max_jump} 椤?")
                    
                    # 妫€鏌ユ槸鍚︽湁璺宠浆瓒呰繃鏈€澶ч檺鍒?
                    if actual_max_jump > max_jump_distance:
                        logger.warning(f"鏈夎烦杞秴杩囨渶澶ч檺鍒?{max_jump_distance} 椤碉紝瀹為檯鏈€澶?{actual_max_jump} 椤?")
        
        # 鎬荤粨鍖归厤搴忓垪锛堢敤浜庤皟璇曪級
        if processed_path and len(processed_path) > 0:
            # 鎻愬彇骞荤伅鐗囩储寮曞簭鍒?
            slide_sequence = [slide_idx for _, slide_idx, _ in processed_path]
            
            # 鏋勫缓绠€鍖栫殑搴忓垪琛ㄧず
            seq_str = ""
            prev_slide = -1
            count = 1
            
            for i, slide_idx in enumerate(slide_sequence):
                if slide_idx == prev_slide:
                    count += 1
                else:
                    if prev_slide != -1:
                        seq_str += f"{prev_slide}(x{count}) -> "
                    prev_slide = slide_idx
                    count = 1
            
            if prev_slide != -1:
                seq_str += f"{prev_slide}(x{count})"
            
            # 璇嗗埆寮傚父璺宠浆锛堝ぇ骞呭悜鍚庤烦杞級
            abnormal_jumps = []
            for i in range(1, len(slide_sequence)):
                prev = slide_sequence[i-1]
                curr = slide_sequence[i]
                jump = curr - prev
                if jump < -3:  # 鍚戝悗璺宠浆瓒呰繃3椤佃涓哄紓甯?
                    abnormal_jumps.append((i, prev, curr, jump))
            
            if seq_str:
                logger.info(f"鍖归厤搴忓垪鎬荤粨: {seq_str}")
            
            if abnormal_jumps:
                logger.warning(f"鍙戠幇 {len(abnormal_jumps)} 娆″紓甯稿悜鍚庤烦杞?")
                for idx, prev, curr, jump in abnormal_jumps:
                    logger.warning(f"  浣嶇疆 {idx}: {prev} -> {curr} (璺宠浆 {jump} 椤?")
            else:
                logger.info("鏈彂鐜板紓甯稿悜鍚庤烦杞?")
        
        return timeline
    
    def _handle_unmatched_segments(
        self,
        path: List[Tuple[int, int, float]],
        n_slides: int
    ) -> List[Tuple[int, int, float]]:
        """澶勭悊鏈尮閰嶇殑鐗囨锛坰lide_index = -1锛?
        
        绛栫暐锛氫娇鐢ㄦ渶杩戠殑鏈夋晥骞荤伅鐗囩储寮曪紝濡傛灉娌℃湁鍒欎娇鐢ㄧ涓€寮犲够鐏墖
        
        Args:
            path: 璺緞鍒楄〃锛屾瘡涓厓绱犱负(瀛楀箷绱㈠紩, 骞荤伅鐗囩储寮? 鐩镐技搴?
            n_slides: 骞荤伅鐗囨€绘暟
            
        Returns:
            澶勭悊鍚庣殑璺緞锛屾墍鏈塻lide_index >= 0
        """
        if not path:
            return []
        
        logger.info(f"澶勭悊鏈尮閰嶇墖娈? 鍏�{len(path)}涓墖娈?")
        
        # 鎵惧埌鎵€鏈夋湭鍖归厤鐨勭墖娈?
        unmatched_indices = [i for i, (_, slide_idx, _) in enumerate(path) if slide_idx < 0]
        logger.info(f"鍙戠幇 {len(unmatched_indices)} 涓湭鍖归厤鐗囨")
        
        if not unmatched_indices:
            return path
        
        # 鍒涘缓澶勭悊鍚庣殑璺緞鍓湰
        processed_path = list(path)
        
        # 涓烘瘡涓湭鍖归厤鐨勭墖娈垫壘鍒版浛浠ｇ殑骞荤伅鐗囩储寮?
        for idx in unmatched_indices:
            # 绛栫暐1锛氭煡鎵惧墠鍚庢渶杩戠殑鏈夋晥鍖归厤
            slide_candidate = -1
            
            # 鍚戝墠鏌ユ壘
            for i in range(idx - 1, -1, -1):
                if processed_path[i][1] >= 0:
                    slide_candidate = processed_path[i][1]
                    break
            
            # 濡傛灉鍚戝墠娌℃壘鍒帮紝鍚戝悗鏌ユ壘
            if slide_candidate < 0:
                for i in range(idx + 1, len(processed_path)):
                    if processed_path[i][1] >= 0:
                        slide_candidate = processed_path[i][1]
                        break
            
            # 濡傛灉鍓嶅悗閮芥病鎵惧埌锛屼娇鐢ㄧ涓€寮犲够鐏墖
            if slide_candidate < 0:
                slide_candidate = 0
            
            # 鏇存柊璺緞
            subtitle_idx, _, similarity = processed_path[idx]
            processed_path[idx] = (subtitle_idx, slide_candidate, similarity)
            logger.debug(f"鏈尮閰嶇墖娈?{idx}: 浣跨敤骞荤伅鐗?{slide_candidate} 浣滀负鏇夸唬")
        
        # 楠岃瘉鎵€鏈夌墖娈甸兘鏈夋湁鏁堢殑骞荤伅鐗囩储寮?
        valid_count = sum(1 for _, slide_idx, _ in processed_path if slide_idx >= 0)
        logger.info(f"澶勭悊鍚? {valid_count}/{len(processed_path)} 涓墖娈垫湁鏈夋晥鍖归厤")
        
        return processed_path
    
    def _backtrack_dp(
        self,
        dp: np.ndarray,
        prev: np.ndarray,
        subtitles: List[Dict[str, Any]],
        similarity_threshold: float
    ) -> List[Tuple[int, int, float]]:
        """鍥炴函DP鎵惧埌鏈€浼樿矾寰勶紙杩囨椂鏂规硶锛屼繚鐣欎互淇濇寔鍏煎鎬э級
        
        娉ㄦ剰锛氭鏂规硶宸蹭笉鍐嶄娇鐢紝鏂扮殑灞€閮ㄦ渶浼樺尮閰嶇畻娉曞彇浠ｄ簡DP绠楁硶
        
        Returns:
            鍒楄〃锛屾瘡涓厓绱犱负(瀛楀箷绱㈠紩, 骞荤伅鐗囩储寮? 鐩镐技搴?
        """
        logger.warning("_backtrack_dp鏂规硶宸茶繃鏃讹紝鏂扮殑灞€閮ㄦ渶浼樺尮閰嶇畻娉曞凡鍙栦唬DP绠楁硶")
        logger.warning("姝ゆ柟娉曡繑鍥炵┖鍒楄〃锛屽疄闄呭鐞嗕娇鐢ㄦ柊鐨勫尮閰嶇畻娉?")
        
        # 姝ゆ柟娉曞凡涓嶅啀浣跨敤锛岃繑鍥炵┖鍒楄〃
        # 淇濈暀姝ゆ柟娉曚互淇濇寔浠ｇ爜鍏煎鎬?
        return []
    
    def _apply_min_duration_constraint(
        self,
        path: List[Tuple[int, int, float]],
        subtitles: List[Dict[str, Any]],
        min_duration: float,
        keep_short_segments: bool = False,
        switch_times: Optional[Dict[int, float]] = None
    ) -> List[Dict[str, Any]]:
        """搴旂敤鏈€灏忓睍绀烘椂闀跨害鏉?
        
        鍚堝苟杩炵画鐨勭浉鍚屽够鐏墖锛岀‘淇濇瘡涓够鐏墖鐗囨鑷冲皯灞曠ずmin_duration绉?
        """
        if not path:
            return []
        
        # 楠岃瘉鎵€鏈夊够鐏墖绱㈠紩閮芥槸鏈夋晥鐨?
        invalid_slides = [(i, slide_idx) for i, (_, slide_idx, _) in enumerate(path) if slide_idx < 0]
        if invalid_slides:
            logger.warning(f"鍙戠幇 {len(invalid_slides)} 涓棤鏁堝够鐏墖绱㈠紩锛屼娇鐢ㄧ涓€寮犲够鐏墖浣滀负鏇夸唬")
            # 淇鏃犳晥绱㈠紩
            fixed_path = []
            for subtitle_idx, slide_idx, similarity in path:
                if slide_idx < 0:
                    slide_idx = 0  # 浣跨敤绗竴寮犲够鐏墖
                    logger.debug(f"淇鐗囨 {subtitle_idx}: 骞荤伅鐗囩储寮曚粠 -1 鏀逛负 0")
                fixed_path.append((subtitle_idx, slide_idx, similarity))
            path = fixed_path
        
        timeline = []
        current_slide = path[0][1]
        current_start = subtitles[path[0][0]]["start"]
        current_end = subtitles[path[0][0]]["end"]
        
        for i in range(1, len(path)):
            subtitle_idx, slide_idx, similarity = path[i]
            
            if slide_idx == current_slide:
                # 鐩稿悓骞荤伅鐗囷紝鎵╁睍缁撴潫鏃堕棿
                current_end = max(current_end, subtitles[subtitle_idx]["end"])
            else:
                boundary_time = subtitles[subtitle_idx]["start"]
                if switch_times:
                    boundary_time = float(switch_times.get(i, boundary_time))
                boundary_time = max(current_end, boundary_time)

                # 骞荤伅鐗囧垏鎹紝淇濆瓨褰撳墠鐗囨
                duration = boundary_time - current_start
                if duration >= min_duration:
                    timeline.append({
                        "start": current_start,
                        "end": boundary_time,
                        "slide_index": int(current_slide),
                        "duration": duration
                    })
                else:
                    # 鐗囨鏃堕暱涓嶈冻锛岃褰曚絾涓嶇珛鍗虫坊鍔狅紝绋嶅悗澶勭悊
                    timeline.append({
                        "start": current_start,
                        "end": boundary_time,
                        "slide_index": int(current_slide),
                        "duration": duration,
                        "short_segment": True  # 鏍囪涓虹煭鐗囨
                    })
                
                # 寮€濮嬫柊鐗囨
                current_slide = slide_idx
                current_start = boundary_time
                current_end = max(boundary_time, subtitles[subtitle_idx]["end"])
        
        # 娣诲姞鏈€鍚庝竴涓墖娈?
        duration = current_end - current_start
        timeline.append({
            "start": current_start,
            "end": current_end,
            "slide_index": int(current_slide),
            "duration": duration
        })
        
        # 椤哄簭寮虹害鏉熷満鏅彲閫夋嫨淇濈暀鐭墖娈碉紝涓嶅仛璺ㄩ〉鍚堝苟
        if keep_short_segments:
            merged_timeline = timeline
        else:
            merged_timeline = self._merge_short_segments(timeline, min_duration)
        
        # 楠岃瘉鏈€缁堟椂闂磋酱
        valid_segments = 0
        short_segments = 0
        for segment in merged_timeline:
            if segment.get("duration", 0) >= min_duration:
                valid_segments += 1
            else:
                short_segments += 1
        
        if short_segments > 0:
            logger.warning(f"鏃堕棿杞村寘鍚?{short_segments} 涓煭浜庢渶灏忓睍绀烘椂闀跨殑鐗囨")
        
        logger.info(f"鏃堕棿杞寸敓鎴愬畬鎴? {len(merged_timeline)} 涓墖娈? 鍏朵腑 {valid_segments} 涓弧瓒虫渶灏忔椂闀胯姹?")
        
        return merged_timeline
    
    def _merge_short_segments(
        self,
        timeline: List[Dict[str, Any]],
        min_duration: float
    ) -> List[Dict[str, Any]]:
        """鏅鸿兘鍚堝苟杩囩煭鐨勭墖娈?
        
        绛栫暐锛?
        1. 璇嗗埆鎵€鏈夌煭浜巑in_duration鐨勭墖娈?
        2. 灏濊瘯灏嗙煭鐗囨鍚堝苟鍒扮浉閭荤殑鐩稿悓骞荤伅鐗囩墖娈?
        3. 濡傛灉娌℃湁鐩稿悓骞荤伅鐗囩殑鐩搁偦鐗囨锛屽悎骞跺埌鏃堕棿涓婃渶杩戠殑鐗囨
        4. 纭繚鍚堝苟鍚庣殑鐗囨浠嶇劧婊¤冻鏃堕棿椤哄簭
        
        Args:
            timeline: 鍘熷鏃堕棿杞村垪琛?
            min_duration: 鏈€灏忓睍绀烘椂闀?
            
        Returns:
            鍚堝苟鍚庣殑鏃堕棿杞村垪琛?
        """
        if not timeline:
            return []
        
        # 璇嗗埆鐭墖娈?
        short_segments_indices = [
            i for i, segment in enumerate(timeline)
            if segment.get("duration", 0) < min_duration or segment.get("short_segment", False)
        ]
        
        if not short_segments_indices:
            return timeline
        
        logger.info(f"鍙戠幇 {len(short_segments_indices)} 涓渶瑕佸悎骞剁殑鐭墖娈?")
        
        # 鍒涘缓鏃堕棿杞村壇鏈敤浜庝慨鏀?
        merged_timeline = list(timeline)
        merged_indices = set()  # 璁板綍宸茶鍚堝苟鐨勭墖娈电储寮?
        
        # 鎸夐『搴忓鐞嗙煭鐗囨
        for idx in short_segments_indices:
            if idx in merged_indices:
                continue  # 杩欎釜鐗囨宸茬粡琚悎骞朵簡
            
            segment = merged_timeline[idx]
            
            # 鏌ユ壘鍚堝苟鐩爣锛氫紭鍏堣€冭檻鐩稿悓骞荤伅鐗囩殑鐩搁偦鐗囨
            best_target_idx = -1
            best_target_score = -1
            
            # 妫€鏌ュ墠鍚庣墖娈?
            for target_idx in [idx-1, idx+1]:
                if 0 <= target_idx < len(merged_timeline) and target_idx not in merged_indices:
                    target_segment = merged_timeline[target_idx]
                    
                    # 璁＄畻鍖归厤鍒嗘暟
                    score = 0
                    if target_segment["slide_index"] == segment["slide_index"]:
                        score += 10  # 鐩稿悓骞荤伅鐗囷紝浼樺厛鍚堝苟
                    
                    # 鏃堕棿鎺ヨ繎搴︼紙鏃堕棿宸秺灏忚秺濂斤級
                    time_gap = abs(target_segment["start"] - segment["end"])
                    if time_gap < 1.0:  # 鏃堕棿宸皬浜?绉?
                        score += 5
                    
                    if score > best_target_score:
                        best_target_score = score
                        best_target_idx = target_idx
            
            if best_target_idx != -1:
                # 鍚堝苟鍒扮洰鏍囩墖娈?
                target_segment = merged_timeline[best_target_idx]
                source_segment = segment
                
                # 鎵╁睍鐩爣鐗囨鐨勬椂闂磋寖鍥?
                new_start = min(target_segment["start"], source_segment["start"])
                new_end = max(target_segment["end"], source_segment["end"])
                
                target_segment["start"] = new_start
                target_segment["end"] = new_end
                target_segment["duration"] = new_end - new_start
                
                # 鏍囪婧愮墖娈典负宸插悎骞?
                merged_indices.add(idx)
                
                # 濡傛灉鐩爣鐗囨涔嬪墠涔熻鏍囪涓虹煭鐗囨锛屾洿鏂扮姸鎬?
                if target_segment.get("short_segment", False):
                    target_segment.pop("short_segment", None)
                
                logger.debug(f"鍚堝苟鐗囨 {idx} (骞荤伅鐗?{source_segment['slide_index']}, 鏃堕暱 {source_segment['duration']:.1f}s) "
                           f"鍒扮墖娈?{best_target_idx} (骞荤伅鐗?{target_segment['slide_index']})")
        
        # 绉婚櫎宸茶鍚堝苟鐨勭墖娈?
        result_timeline = [
            segment for i, segment in enumerate(merged_timeline)
            if i not in merged_indices
        ]
        
        # 鎸夋椂闂存帓搴?
        result_timeline.sort(key=lambda x: x["start"])
        
        # 鍐嶆妫€鏌ユ槸鍚﹁繕鏈夌煭鐗囨
        final_short_count = sum(1 for segment in result_timeline if segment.get("duration", 0) < min_duration)
        if final_short_count > 0:
            logger.warning(f"鍚堝苟鍚庝粛鏈?{final_short_count} 涓煭鐗囨鏃犳硶鍚堝苟")
        
        logger.info(f"鐗囨鍚堝苟瀹屾垚: 浠?{len(timeline)} 涓悎骞跺埌 {len(result_timeline)} 涓?")
        
        return result_timeline
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """璁＄畻涓や釜鏂囨湰鐨勮涔夌浉浼煎害"""
        embeddings = self.model.encode([text1, text2], convert_to_numpy=True)
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def find_best_slide_for_text(
        self,
        text: str,
        slides: List[Dict[str, Any]],
        threshold: float = 0.0
    ) -> Tuple[int, float]:
        """涓虹粰瀹氭枃鏈壘鍒版渶鍖归厤鐨勫够鐏墖
        
        Returns:
            (骞荤伅鐗囩储寮? 鐩镐技搴?锛屽鏋滄病鏈夊尮閰嶅垯杩斿洖(-1, 0.0)
        """
        slide_texts = [slide["full_text"] for slide in slides]
        
        text_embedding = self.model.encode([text], convert_to_numpy=True)
        slide_embeddings = self.model.encode(slide_texts, convert_to_numpy=True)
        
        similarities = cosine_similarity(text_embedding, slide_embeddings)[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score >= threshold:
            return (best_idx, best_score)
        else:
            return (-1, best_score)
    
    def _create_uniform_timeline_fallback(
        self,
        slides: List[Dict[str, Any]],
        subtitles: List[Dict[str, Any]],
        min_display_duration: float = 2.0,
        force_full_coverage: bool = False
    ) -> List[Dict[str, Any]]:
        """鍒涘缓鍧囧寑鍒嗗竷鐨勬椂闂磋酱锛堝鐢ㄧ瓥鐣ワ級
        
        褰撹涔夊榻愬け璐ユ椂锛屽皢骞荤伅鐗囧潎鍖€鍒嗗竷鍦ㄥ瓧骞曟椂闂磋寖鍥村唴
        
        Args:
            slides: 骞荤伅鐗囧垪琛?
            subtitles: 瀛楀箷鐗囨鍒楄〃
            min_display_duration: 鏈€灏忓睍绀烘椂闀匡紙绉掞級
            force_full_coverage: 鏄惁寮哄埗姣忛〉鑷冲皯鍒嗛厤涓€涓椂闂寸墖娈?            
        Returns:
            鏃堕棿杞存槧灏勫垪琛紝姣忎釜鍏冪礌鍖呭惈start, end, slide_index
        """
        if not slides or not subtitles:
            logger.warning("澶囩敤绛栫暐锛氬够鐏墖鎴栧瓧骞曞垪琛ㄤ负绌猴紝鏃犳硶鍒涘缓鏃堕棿杞?")
            return []
        
        if force_full_coverage:
            logger.info("璇箟瀵归綈澶辫触锛屼娇鐢ㄩ『搴忓叏瑕嗙洊澶囩敤绛栫暐")

            start_time = subtitles[0]["start"]
            end_time = subtitles[-1]["end"]
            total_duration = end_time - start_time

            if total_duration <= 0:
                logger.warning(f"澶囩敤绛栫暐锛氬瓧骞曟椂闀挎棤鏁堬紙{total_duration}绉掞級锛屼娇鐢ㄦ渶灏忓厹搴曟椂闀?")
                total_duration = max(0.1 * len(slides), min_display_duration)
                end_time = start_time + total_duration

            # 寮哄埗姣忛〉閮芥湁鐙珛鏃堕棿鐗囨锛岄伩鍏嶁€滆烦椤碘€濇垨鈥滄紡椤碘€?
            slide_count = max(1, len(slides))
            slide_duration = total_duration / slide_count
            slide_duration = max(slide_duration, 0.001)

            logger.debug(
                f"澶囩敤绛栫暐鍙傛暟: total_duration={total_duration:.3f}s, slides={slide_count}, "
                f"min_display_duration={min_display_duration}s, assigned_slide_duration={slide_duration:.3f}s"
            )

            timeline = []
            current_time = start_time

            for i, slide in enumerate(slides):
                if i == slide_count - 1:
                    seg_end = end_time
                else:
                    seg_end = start_time + (i + 1) * slide_duration

                if seg_end <= current_time:
                    seg_end = current_time + 0.001

                timeline.append({
                    "start": current_time,
                    "end": seg_end,
                    "slide_index": i,
                    "duration": seg_end - current_time,
                    "similarity": 0.0,
                    "strategy": "uniform_fallback",
                    "note": f"椤哄簭鍏ㄨ鐩栧鐢ㄧ瓥鐣ワ紝鍘熷绱㈠紩: {slide.get('index', i)}"
                })

                current_time = seg_end
        else:
            logger.info("璇箟瀵归綈澶辫触锛屼娇鐢ㄥ潎鍖€鍒嗗竷澶囩敤绛栫暐")

            total_duration = subtitles[-1]["end"] - subtitles[0]["start"]
            if total_duration <= 0:
                logger.warning(f"澶囩敤绛栫暐锛氬瓧骞曟椂闀挎棤鏁堬紙{total_duration}绉掞級")
                total_duration = len(slides) * min_display_duration

            slide_duration = max(min_display_duration, total_duration / max(1, len(slides)))
            if slide_duration * len(slides) > total_duration:
                slide_duration = total_duration / max(1, len(slides))
            slide_duration = max(0.1, slide_duration)

            logger.debug(
                f"澶囩敤绛栫暐鍙傛暟: total_duration={total_duration:.2f}s, slides={len(slides)}, "
                f"min_display_duration={min_display_duration}s, calculated_slide_duration={slide_duration:.2f}s"
            )

            timeline = []
            current_time = subtitles[0]["start"]

            for i, slide in enumerate(slides):
                end_time = current_time + slide_duration
                if end_time > subtitles[-1]["end"]:
                    end_time = subtitles[-1]["end"]

                if end_time - current_time < min_display_duration / 2:
                    if timeline:
                        last_segment = timeline[-1]
                        last_segment["end"] = subtitles[-1]["end"]
                        last_segment["duration"] = last_segment["end"] - last_segment["start"]
                        logger.debug(f"澶囩敤绛栫暐锛氱{i}寮犲够鐏墖鏃堕棿涓嶈冻锛屽悎骞跺埌鍓嶄竴寮?")
                    else:
                        end_time = current_time + min_display_duration
                        if end_time > subtitles[-1]["end"]:
                            end_time = subtitles[-1]["end"]
                    break

                actual_duration = end_time - current_time
                if actual_duration <= 0:
                    logger.warning(f"澶囩敤绛栫暐锛氱墖娈碟{i}鏃堕暱鏃犳晥({actual_duration:.2f}s)锛岃烦杩囨鐗囨")
                    continue

                if end_time <= current_time:
                    logger.warning(f"澶囩敤绛栫暐锛氱墖娈碟{i}缁撴潫鏃堕棿涓嶅ぇ浜庡紑濮嬫椂闂达紝璋冩暣缁撴潫鏃堕棿")
                    end_time = current_time + slide_duration

                timeline.append({
                    "start": current_time,
                    "end": end_time,
                    "slide_index": i,
                    "duration": end_time - current_time,
                    "similarity": 0.0,
                    "strategy": "uniform_fallback",
                    "note": f"澶囩敤绛栫暐鐢熸垚锛屽師濮嬬储寮? {slide.get('index', i)}"
                })
                current_time = end_time
        
        # 楠岃瘉鐢熸垚鐨勬椂闂磋酱
        validated_timeline = self._validate_timeline(timeline)
        
        if not validated_timeline:
            logger.error("澶囩敤绛栫暐鐢熸垚鐨勬椂闂磋酱鏃犳晥锛岃繑鍥炵┖鍒楄〃")
            return []
        
        logger.info(f"澶囩敤绛栫暐鐢熸垚{len(validated_timeline)}涓湁鏁堟椂闂寸墖娈碉紝姣忕墖绾�{slide_duration:.3f}绉?")
        return validated_timeline
    
    def _validate_timeline(self, timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """楠岃瘉鏃堕棿杞存湁鏁堟€э紝杩囨护鏃犳晥鐗囨
        
        Args:
            timeline: 鏃堕棿杞村垪琛?
            
        Returns:
            楠岃瘉鍚庣殑鏃堕棿杞村垪琛紝鍙寘鍚湁鏁堢墖娈?
        """
        if not timeline:
            logger.warning("鏃堕棿杞翠负绌猴紝鏃犻渶楠岃瘉")
            return []
        
        valid_timeline = []
        invalid_count = 0
        
        for i, segment in enumerate(timeline):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            duration = end_time - start_time
            
            # 楠岃瘉鏃堕棿鐗囨鏈夋晥鎬?
            is_valid = True
            issues = []
            
            if duration <= 0:
                is_valid = False
                issues.append(f"duration={duration}<=0")
            
            if end_time <= start_time:
                is_valid = False
                issues.append(f"end={end_time}<=start={start_time}")
            
            slide_index = segment.get("slide_index", -1)
            if slide_index < 0:
                is_valid = False
                issues.append(f"slide_index={slide_index}<0")
            
            if is_valid:
                valid_timeline.append(segment)
                logger.debug(f"鏃堕棿鐗囨{i}鏈夋晥: slide_index={slide_index}, duration={duration:.2f}s")
            else:
                invalid_count += 1
                logger.warning(f"鏃堕棿鐗囨{i}鏃犳晥: {', '.join(issues)}")
        
        if invalid_count > 0:
            logger.warning(f"鏃堕棿杞撮獙璇? {len(valid_timeline)}涓湁鏁堢墖娈? {invalid_count}涓棤鏁堢墖娈佃杩囨护")
        
        if not valid_timeline:
            logger.error("鏃堕棿杞撮獙璇? 鎵€鏈夌墖娈靛潎鏃犳晥")
        
        return valid_timeline

