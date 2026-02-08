import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

class AlignmentService:
    """语义对齐服务"""
    
    def __init__(self, model_name: str = None):
        """初始化对齐服务
        
        Args:
            model_name: sentence-transformers模型名称
        """
        self.model_name = model_name or settings.SENTENCE_TRANSFORMER_MODEL
        self.model = None
        # 延迟加载模型，在第一次使用时加载
    
    def _load_model(self):
        """加载预训练模型"""
        try:
            logger.info(f"正在加载模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("模型加载完成")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise Exception(f"无法加载模型 {self.model_name}: {str(e)}")
    
    def align_slides_with_subtitles(
        self,
        slides: List[Dict[str, Any]],
        subtitles: List[Dict[str, Any]],
        similarity_threshold: float = 0.5,
        min_display_duration: float = 2.0
    ) -> List[Dict[str, Any]]:
        """将幻灯片与字幕进行语义对齐
        
        Args:
            slides: 幻灯片列表，每个幻灯片包含title/content/notes字段
            subtitles: 预处理后的字幕片段列表，每个片段包含text字段
            similarity_threshold: 相似度阈值（用于诊断与统计）
            min_display_duration: 最小展示时长（秒）
            
        Returns:
            时间轴映射列表，每个元素包含start, end, slide_index
        """
        # 验证输入数据
        if not slides:
            logger.error("幻灯片列表为空，PPT解析可能失败")
            logger.error("请检查：1) PPT文件是否存在 2) PPT文件是否损坏 3) python-pptx库是否安装正确")
            return []
        
        if not subtitles:
            logger.error("字幕列表为空，SRT解析可能失败")
            logger.error("请检查：1) SRT文件是否存在 2) SRT格式是否正确 3) pysrt库是否安装正确")
            return []
        
        # 确保模型已加载
        if self.model is None:
            self._load_model()
        
        logger.info(f"开始语义对齐: {len(slides)}张幻灯片, {len(subtitles)}个字幕片段")
        logger.info(f"参数设置: similarity_threshold={similarity_threshold}, min_display_duration={min_display_duration}")
        
        # 1. 准备文本（优先使用备注作为对齐文本）
        slide_texts = self._build_slide_alignment_texts(slides)

        # 记录内容样本用于调试（只记录前3个，避免日志过大）
        if slides:
            sample_slides = [text[:100] + "..." if len(text) > 100 else text for text in slide_texts[:3]]
            logger.debug(f"前3张幻灯片内容样本: {sample_slides}")

        if subtitles:
            sample_subtitles = [sub['text'][:100] + "..." if len(sub['text']) > 100 else sub['text'] for sub in subtitles[:3]]
            logger.debug(f"前3个字幕内容样本: {sample_subtitles}")
            avg_len = sum(len(sub.get("text", "")) for sub in subtitles) / max(1, len(subtitles))
            avg_dur = sum(sub.get("duration", sub.get("end", 0) - sub.get("start", 0)) for sub in subtitles) / max(1, len(subtitles))
            logger.info(f"预处理片段统计: 片段数={len(subtitles)}, 平均长度={avg_len:.1f}, 平均时长={avg_dur:.2f}s")
        subtitle_texts = [
            (subtitle.get("text") or "").strip() or (subtitle.get("raw_text") or "").strip()
            for subtitle in subtitles
        ]
        
        # 2. 生成嵌入向量
        slide_embeddings = self.model.encode(slide_texts, convert_to_numpy=True)
        subtitle_embeddings = self.model.encode(subtitle_texts, convert_to_numpy=True)
        
        # 3. 计算相似度矩阵
        similarity_matrix = cosine_similarity(subtitle_embeddings, slide_embeddings)
        
        # 记录相似度统计信息用于调试
        max_similarities = similarity_matrix.max(axis=1) if similarity_matrix.size else np.array([])
        avg_similarity = float(max_similarities.mean()) if len(max_similarities) > 0 else 0.0
        matches_above_threshold = int((max_similarities >= similarity_threshold).sum()) if len(max_similarities) > 0 else 0
        p10 = float(np.percentile(max_similarities, 10)) if len(max_similarities) > 0 else 0.0
        p50 = float(np.percentile(max_similarities, 50)) if len(max_similarities) > 0 else 0.0
        p90 = float(np.percentile(max_similarities, 90)) if len(max_similarities) > 0 else 0.0
        
        # 更详细的统计信息
        similarity_distribution = {
            "<0.1": (max_similarities < 0.1).sum(),
            "0.1-0.3": ((max_similarities >= 0.1) & (max_similarities < 0.3)).sum(),
            "0.3-0.5": ((max_similarities >= 0.3) & (max_similarities < 0.5)).sum(),
            "0.5-0.7": ((max_similarities >= 0.5) & (max_similarities < 0.7)).sum(),
            "0.7-0.9": ((max_similarities >= 0.7) & (max_similarities < 0.9)).sum(),
            ">=0.9": (max_similarities >= 0.9).sum(),
        }
        
        logger.info(f"相似度统计: 平均最大相似度={avg_similarity:.3f}, P10={p10:.3f}, P50={p50:.3f}, P90={p90:.3f}")
        logger.info(f"高于阈值({similarity_threshold})的匹配数={matches_above_threshold}/{len(subtitles)}")
        logger.info(f"相似度分布: {similarity_distribution}")
        
        if avg_similarity < 0.1:
            logger.warning("平均相似度过低(<0.1)，PPT和字幕可能完全不相关")
            logger.warning("可能原因:")
            logger.warning("1. PPT和字幕内容确实不相关")
            logger.warning("2. PPT文本提取失败（检查PPT内容）")
            logger.warning("3. 模型不适合当前领域（考虑更换模型）")
            logger.warning("4. 文本预处理问题（特殊字符、编码等）")
        elif avg_similarity < similarity_threshold:
            logger.warning(f"平均相似度({avg_similarity:.3f})低于阈值({similarity_threshold})，建议降低阈值")
            logger.warning(f"当前阈值下，只有 {matches_above_threshold}/{len(subtitles)} 个片段能匹配")
            logger.warning("建议阈值设置:")
            logger.warning(f"- 宽松匹配: 0.1-0.3 (当前有 {similarity_distribution['0.1-0.3'] + similarity_distribution['0.3-0.5'] + similarity_distribution['0.5-0.7'] + similarity_distribution['0.7-0.9'] + similarity_distribution['>=0.9']} 个片段)")
            logger.warning(f"- 中等匹配: 0.3-0.5 (当前有 {similarity_distribution['0.3-0.5'] + similarity_distribution['0.5-0.7'] + similarity_distribution['0.7-0.9'] + similarity_distribution['>=0.9']} 个片段)")
            logger.warning(f"- 严格匹配: 0.5+ (当前有 {similarity_distribution['0.5-0.7'] + similarity_distribution['0.7-0.9'] + similarity_distribution['>=0.9']} 个片段)")
        
        if matches_above_threshold == 0:
            logger.warning(f"没有匹配项达到阈值({similarity_threshold})，对齐质量可能不稳定")
            logger.warning("调试建议:")
            logger.warning("1. 当前阈值太高，建议降低到0.1-0.3范围")
            logger.warning(f"2. 检查相似度分布: {similarity_distribution}")
            logger.warning("3. 检查PPT文本内容（前3页样本已在上方显示）")
            logger.warning("4. 检查字幕文本内容（前3段样本已在上方显示）")
            logger.warning("5. 如果内容相关但相似度低，考虑更换模型或优化文本清洗")
        
        # 4. 受限回退的单调序列对齐（Viterbi/DP）
        path = self._align_with_viterbi(
            similarity_matrix=similarity_matrix,
            max_backtrack=settings.ALIGN_MAX_BACKTRACK,
            max_forward_jump=settings.ALIGN_MAX_FORWARD_JUMP,
            switch_penalty=settings.ALIGN_SWITCH_PENALTY,
            backtrack_penalty=settings.ALIGN_BACKTRACK_PENALTY,
            forward_jump_penalty=settings.ALIGN_FORWARD_JUMP_PENALTY
        )

        # 应用最小时长约束生成时间轴
        timeline = self._apply_min_duration_constraint(
            path, subtitles, min_display_duration
        )
        
        # 如果语义对齐失败，使用备用策略
        if not timeline:
            logger.warning("语义对齐失败，没有生成任何时间片段")
            logger.error(f"幻灯片数: {len(slides)}, 片段数: {len(subtitles)}")
        else:
            logger.info(f"语义对齐成功，生成{len(timeline)}个时间片段")
        
        return timeline

    def _build_slide_alignment_texts(self, slides: List[Dict[str, Any]]) -> List[str]:
        """构建用于对齐的幻灯片文本（优先使用备注）"""
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
            logger.info(f"对齐文本构建: 备注优先使用{notes_count}/{len(slides)}页, 空文本页={empty_count}")

        return texts

    def _align_with_viterbi(
        self,
        similarity_matrix: np.ndarray,
        max_backtrack: int,
        max_forward_jump: int,
        switch_penalty: float,
        backtrack_penalty: float,
        forward_jump_penalty: float
    ) -> List[Tuple[int, int, float]]:
        """受限回退的单调序列对齐（Viterbi/DP）"""
        if similarity_matrix is None or similarity_matrix.size == 0:
            return []

        n_segments, n_slides = similarity_matrix.shape
        if n_segments == 0 or n_slides == 0:
            return []

        dp = np.full((n_segments, n_slides), -np.inf, dtype=float)
        prev = np.full((n_segments, n_slides), -1, dtype=int)

        # 初始化
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

        # 回溯
        last_slide = int(np.argmax(dp[-1, :]))
        path = [last_slide]
        for seg_idx in range(n_segments - 1, 0, -1):
            last_slide = int(prev[seg_idx, last_slide])
            if last_slide < 0:
                last_slide = int(np.argmax(dp[seg_idx - 1, :]))
            path.append(last_slide)
        path.reverse()

        # 统计对齐行为
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
            f"对齐切页统计: 总片段={n_segments}, 切页次数={switch_count}, 回退次数={backtrack_count}, 最大回退页数={max_backtrack_actual}"
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
        """应用约束生成时间轴映射
        
        改进算法：约束局部最优匹配
        约束条件:
        1. 时间区域划分：开始区域（前10%）、核心区域（中间80%）、结束区域（后10%）
        2. 页面限制：核心区域禁止匹配第一页和最后一页
        3. 跳转距离限制：相邻片段最大跳转5页
        4. 相似度必须高于阈值
        5. 每个幻灯片至少展示最小时长
        """
        n_subtitles = len(subtitles)
        n_slides = len(slides)
        
        logger.info(f"使用约束局部最优匹配算法: {n_subtitles}个字幕片段, {n_slides}张幻灯片")
        logger.info(f"相似度阈值: {similarity_threshold}, 最小展示时长: {min_display_duration}秒")
        
        # 算法参数
        start_region_ratio = 0.1  # 开始区域占总时长的比例
        end_region_ratio = 0.1    # 结束区域占总时长的比例
        max_jump_distance = 5     # 最大跳转页数
        
        logger.info(f"算法参数: 开始区域 {start_region_ratio*100:.0f}%, 结束区域 {end_region_ratio*100:.0f}%, 最大跳转 {max_jump_distance} 页")
        
        # 计算时间边界
        total_duration = subtitles[-1]["end"] if subtitles else 0
        start_boundary = total_duration * start_region_ratio
        end_boundary = total_duration * (1 - end_region_ratio)
        
        logger.info(f"时间边界: 总时长 {total_duration:.1f}s, 开始边界 {start_boundary:.1f}s, 结束边界 {end_boundary:.1f}s")
        
        # 统计区域分布
        region_counts = {"start": 0, "core": 0, "end": 0}
        
        # 记录相似度统计信息
        best_similarities = []
        matched_count = 0
        no_match_count = 0
        
        # 约束应用统计
        constraint_stats = {
            "core_region_excluded": 0,  # 核心区域排除第一页/最后一页的次数
            "jump_distance_limited": 0, # 跳转距离限制的次数
            "candidates_filtered": 0,   # 候选被过滤的次数
        }
        
        # 为每个字幕片段选择最佳幻灯片（应用约束）
        path = []
        previous_slide = -1  # 记录上一个匹配的幻灯片
        
        for i in range(n_subtitles):
            subtitle = subtitles[i]
            current_time = subtitle["start"]
            
            # 确定当前片段所属区域
            if current_time < start_boundary:
                region = "start"
            elif current_time > end_boundary:
                region = "end"
            else:
                region = "core"
            
            region_counts[region] += 1
            
            # 初始化候选幻灯片（所有幻灯片）
            candidates = list(range(n_slides))
            candidate_count_before = len(candidates)
            
            # 约束1：核心区域禁止匹配第一页和最后一页
            if region == "core" and n_slides > 2:
                # 排除第一页 (索引0) 和最后一页 (索引n_slides-1)
                filtered_candidates = [j for j in candidates if j != 0 and j != n_slides-1]
                if filtered_candidates:
                    candidates = filtered_candidates
                    constraint_stats["core_region_excluded"] += 1
                    logger.debug(f"字幕片段 {i} (核心区域): 排除第一页和最后一页，候选从 {candidate_count_before} 减少到 {len(candidates)}")
            
            # 约束2：跳转距离限制（如果有前一个匹配）
            if previous_slide >= 0 and candidates:
                candidates_before_jump = len(candidates)
                filtered_candidates = [
                    j for j in candidates 
                    if abs(j - previous_slide) <= max_jump_distance
                ]
                if filtered_candidates:
                    candidates = filtered_candidates
                    constraint_stats["jump_distance_limited"] += 1
                    logger.debug(f"字幕片段 {i}: 跳转距离限制，候选从 {candidates_before_jump} 减少到 {len(candidates)}")
                else:
                    # 如果没有候选满足跳转距离限制，放宽限制（保持当前候选）
                    logger.debug(f"字幕片段 {i}: 跳转距离限制过严，无候选满足条件，保持当前 {candidates_before_jump} 个候选")
            
            if len(candidates) < candidate_count_before:
                constraint_stats["candidates_filtered"] += 1
            
            # 在候选幻灯片中找到相似度最高的
            best_slide_idx = -1
            best_similarity = -1.0
            
            for j in candidates:
                similarity = similarity_matrix[i, j]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_slide_idx = j
            
            best_similarities.append(best_similarity)
            
            # 检查是否达到阈值
            if best_similarity >= similarity_threshold and best_slide_idx >= 0:
                path.append((i, best_slide_idx, best_similarity))
                matched_count += 1
                previous_slide = best_slide_idx  # 更新前一个匹配
                logger.debug(f"字幕片段 {i} ({region}区域): 匹配幻灯片 {best_slide_idx}, 相似度 {best_similarity:.3f}")
            else:
                # 没有达到阈值的匹配，标记为-1
                path.append((i, -1, best_similarity))
                no_match_count += 1
                logger.debug(f"字幕片段 {i} ({region}区域): 无匹配 (最佳相似度 {best_similarity:.3f} < 阈值 {similarity_threshold})")
        
        # 统计信息
        if best_similarities:
            avg_similarity = sum(best_similarities) / len(best_similarities)
            max_similarity = max(best_similarities)
            min_similarity = min(best_similarities)
            
            logger.info(f"相似度统计: 平均 {avg_similarity:.3f}, 最大 {max_similarity:.3f}, 最小 {min_similarity:.3f}")
            logger.info(f"匹配结果: {matched_count}/{n_subtitles} 个片段达到阈值, {no_match_count} 个片段未匹配")
            logger.info(f"区域分布: 开始 {region_counts['start']}, 核心 {region_counts['core']}, 结束 {region_counts['end']}")
            logger.info(f"约束应用: 核心区域排除 {constraint_stats['core_region_excluded']} 次, "
                       f"跳转限制 {constraint_stats['jump_distance_limited']} 次, "
                       f"候选过滤 {constraint_stats['candidates_filtered']} 次")
        
        # 处理未匹配的片段（使用最近的有效匹配）
        processed_path = self._handle_unmatched_segments(
            path, n_slides
        )
        
        # 应用最小展示时长约束
        timeline = self._apply_min_duration_constraint(
            processed_path, subtitles, min_display_duration
        )
        
        # 记录PPT跳转统计信息
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
            logger.info(f"PPT跳转统计: 总切换 {total_transitions} 次")
            logger.info(f"  - 向前跳转: {forward_jumps} 次 ({forward_jumps/total_transitions*100:.1f}%)")
            logger.info(f"  - 向后跳转: {backward_jumps} 次 ({backward_jumps/total_transitions*100:.1f}%)")
            logger.info(f"  - 保持同一页: {same_slide_count} 次 ({same_slide_count/total_transitions*100:.1f}%)")
            
            # 记录跳转距离统计
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
                    logger.info(f"跳转距离: 平均 {avg_jump_distance:.1f} 页, 最大 {actual_max_jump} 页")
                    
                    # 检查是否有跳转超过最大限制
                    if actual_max_jump > max_jump_distance:
                        logger.warning(f"有跳转超过最大限制 {max_jump_distance} 页，实际最大 {actual_max_jump} 页")
        
        # 总结匹配序列（用于调试）
        if processed_path and len(processed_path) > 0:
            # 提取幻灯片索引序列
            slide_sequence = [slide_idx for _, slide_idx, _ in processed_path]
            
            # 构建简化的序列表示
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
            
            # 识别异常跳转（大幅向后跳转）
            abnormal_jumps = []
            for i in range(1, len(slide_sequence)):
                prev = slide_sequence[i-1]
                curr = slide_sequence[i]
                jump = curr - prev
                if jump < -3:  # 向后跳转超过3页视为异常
                    abnormal_jumps.append((i, prev, curr, jump))
            
            if seq_str:
                logger.info(f"匹配序列总结: {seq_str}")
            
            if abnormal_jumps:
                logger.warning(f"发现 {len(abnormal_jumps)} 次异常向后跳转:")
                for idx, prev, curr, jump in abnormal_jumps:
                    logger.warning(f"  位置 {idx}: {prev} -> {curr} (跳转 {jump} 页)")
            else:
                logger.info("未发现异常向后跳转")
        
        return timeline
    
    def _handle_unmatched_segments(
        self,
        path: List[Tuple[int, int, float]],
        n_slides: int
    ) -> List[Tuple[int, int, float]]:
        """处理未匹配的片段（slide_index = -1）
        
        策略：使用最近的有效幻灯片索引，如果没有则使用第一张幻灯片
        
        Args:
            path: 路径列表，每个元素为(字幕索引, 幻灯片索引, 相似度)
            n_slides: 幻灯片总数
            
        Returns:
            处理后的路径，所有slide_index >= 0
        """
        if not path:
            return []
        
        logger.info(f"处理未匹配片段: 共{len(path)}个片段")
        
        # 找到所有未匹配的片段
        unmatched_indices = [i for i, (_, slide_idx, _) in enumerate(path) if slide_idx < 0]
        logger.info(f"发现 {len(unmatched_indices)} 个未匹配片段")
        
        if not unmatched_indices:
            return path
        
        # 创建处理后的路径副本
        processed_path = list(path)
        
        # 为每个未匹配的片段找到替代的幻灯片索引
        for idx in unmatched_indices:
            # 策略1：查找前后最近的有效匹配
            slide_candidate = -1
            
            # 向前查找
            for i in range(idx - 1, -1, -1):
                if processed_path[i][1] >= 0:
                    slide_candidate = processed_path[i][1]
                    break
            
            # 如果向前没找到，向后查找
            if slide_candidate < 0:
                for i in range(idx + 1, len(processed_path)):
                    if processed_path[i][1] >= 0:
                        slide_candidate = processed_path[i][1]
                        break
            
            # 如果前后都没找到，使用第一张幻灯片
            if slide_candidate < 0:
                slide_candidate = 0
            
            # 更新路径
            subtitle_idx, _, similarity = processed_path[idx]
            processed_path[idx] = (subtitle_idx, slide_candidate, similarity)
            logger.debug(f"未匹配片段 {idx}: 使用幻灯片 {slide_candidate} 作为替代")
        
        # 验证所有片段都有有效的幻灯片索引
        valid_count = sum(1 for _, slide_idx, _ in processed_path if slide_idx >= 0)
        logger.info(f"处理后: {valid_count}/{len(processed_path)} 个片段有有效匹配")
        
        return processed_path
    
    def _backtrack_dp(
        self,
        dp: np.ndarray,
        prev: np.ndarray,
        subtitles: List[Dict[str, Any]],
        similarity_threshold: float
    ) -> List[Tuple[int, int, float]]:
        """回溯DP找到最优路径（过时方法，保留以保持兼容性）
        
        注意：此方法已不再使用，新的局部最优匹配算法取代了DP算法
        
        Returns:
            列表，每个元素为(字幕索引, 幻灯片索引, 相似度)
        """
        logger.warning("_backtrack_dp方法已过时，新的局部最优匹配算法已取代DP算法")
        logger.warning("此方法返回空列表，实际处理使用新的匹配算法")
        
        # 此方法已不再使用，返回空列表
        # 保留此方法以保持代码兼容性
        return []
    
    def _apply_min_duration_constraint(
        self,
        path: List[Tuple[int, int, float]],
        subtitles: List[Dict[str, Any]],
        min_duration: float
    ) -> List[Dict[str, Any]]:
        """应用最小展示时长约束
        
        合并连续的相同幻灯片，确保每个幻灯片片段至少展示min_duration秒
        """
        if not path:
            return []
        
        # 验证所有幻灯片索引都是有效的
        invalid_slides = [(i, slide_idx) for i, (_, slide_idx, _) in enumerate(path) if slide_idx < 0]
        if invalid_slides:
            logger.warning(f"发现 {len(invalid_slides)} 个无效幻灯片索引，使用第一张幻灯片作为替代")
            # 修复无效索引
            fixed_path = []
            for subtitle_idx, slide_idx, similarity in path:
                if slide_idx < 0:
                    slide_idx = 0  # 使用第一张幻灯片
                    logger.debug(f"修复片段 {subtitle_idx}: 幻灯片索引从 -1 改为 0")
                fixed_path.append((subtitle_idx, slide_idx, similarity))
            path = fixed_path
        
        timeline = []
        current_slide = path[0][1]
        current_start = subtitles[path[0][0]]["start"]
        current_end = subtitles[path[0][0]]["end"]
        
        for i in range(1, len(path)):
            subtitle_idx, slide_idx, similarity = path[i]
            
            if slide_idx == current_slide:
                # 相同幻灯片，扩展结束时间
                current_end = subtitles[subtitle_idx]["end"]
            else:
                # 幻灯片切换，保存当前片段
                duration = current_end - current_start
                if duration >= min_duration:
                    timeline.append({
                        "start": current_start,
                        "end": current_end,
                        "slide_index": int(current_slide),
                        "duration": duration
                    })
                else:
                    # 片段时长不足，记录但不立即添加，稍后处理
                    timeline.append({
                        "start": current_start,
                        "end": current_end,
                        "slide_index": int(current_slide),
                        "duration": duration,
                        "short_segment": True  # 标记为短片段
                    })
                
                # 开始新片段
                current_slide = slide_idx
                current_start = subtitles[subtitle_idx]["start"]
                current_end = subtitles[subtitle_idx]["end"]
        
        # 添加最后一个片段
        duration = current_end - current_start
        timeline.append({
            "start": current_start,
            "end": current_end,
            "slide_index": int(current_slide),
            "duration": duration
        })
        
        # 智能合并过短的片段
        merged_timeline = self._merge_short_segments(timeline, min_duration)
        
        # 验证最终时间轴
        valid_segments = 0
        short_segments = 0
        for segment in merged_timeline:
            if segment.get("duration", 0) >= min_duration:
                valid_segments += 1
            else:
                short_segments += 1
        
        if short_segments > 0:
            logger.warning(f"时间轴包含 {short_segments} 个短于最小展示时长的片段")
        
        logger.info(f"时间轴生成完成: {len(merged_timeline)} 个片段, 其中 {valid_segments} 个满足最小时长要求")
        
        return merged_timeline
    
    def _merge_short_segments(
        self,
        timeline: List[Dict[str, Any]],
        min_duration: float
    ) -> List[Dict[str, Any]]:
        """智能合并过短的片段
        
        策略：
        1. 识别所有短于min_duration的片段
        2. 尝试将短片段合并到相邻的相同幻灯片片段
        3. 如果没有相同幻灯片的相邻片段，合并到时间上最近的片段
        4. 确保合并后的片段仍然满足时间顺序
        
        Args:
            timeline: 原始时间轴列表
            min_duration: 最小展示时长
            
        Returns:
            合并后的时间轴列表
        """
        if not timeline:
            return []
        
        # 识别短片段
        short_segments_indices = [
            i for i, segment in enumerate(timeline)
            if segment.get("duration", 0) < min_duration or segment.get("short_segment", False)
        ]
        
        if not short_segments_indices:
            return timeline
        
        logger.info(f"发现 {len(short_segments_indices)} 个需要合并的短片段")
        
        # 创建时间轴副本用于修改
        merged_timeline = list(timeline)
        merged_indices = set()  # 记录已被合并的片段索引
        
        # 按顺序处理短片段
        for idx in short_segments_indices:
            if idx in merged_indices:
                continue  # 这个片段已经被合并了
            
            segment = merged_timeline[idx]
            
            # 查找合并目标：优先考虑相同幻灯片的相邻片段
            best_target_idx = -1
            best_target_score = -1
            
            # 检查前后片段
            for target_idx in [idx-1, idx+1]:
                if 0 <= target_idx < len(merged_timeline) and target_idx not in merged_indices:
                    target_segment = merged_timeline[target_idx]
                    
                    # 计算匹配分数
                    score = 0
                    if target_segment["slide_index"] == segment["slide_index"]:
                        score += 10  # 相同幻灯片，优先合并
                    
                    # 时间接近度（时间差越小越好）
                    time_gap = abs(target_segment["start"] - segment["end"])
                    if time_gap < 1.0:  # 时间差小于1秒
                        score += 5
                    
                    if score > best_target_score:
                        best_target_score = score
                        best_target_idx = target_idx
            
            if best_target_idx != -1:
                # 合并到目标片段
                target_segment = merged_timeline[best_target_idx]
                source_segment = segment
                
                # 扩展目标片段的时间范围
                new_start = min(target_segment["start"], source_segment["start"])
                new_end = max(target_segment["end"], source_segment["end"])
                
                target_segment["start"] = new_start
                target_segment["end"] = new_end
                target_segment["duration"] = new_end - new_start
                
                # 标记源片段为已合并
                merged_indices.add(idx)
                
                # 如果目标片段之前也被标记为短片段，更新状态
                if target_segment.get("short_segment", False):
                    target_segment.pop("short_segment", None)
                
                logger.debug(f"合并片段 {idx} (幻灯片 {source_segment['slide_index']}, 时长 {source_segment['duration']:.1f}s) "
                           f"到片段 {best_target_idx} (幻灯片 {target_segment['slide_index']})")
        
        # 移除已被合并的片段
        result_timeline = [
            segment for i, segment in enumerate(merged_timeline)
            if i not in merged_indices
        ]
        
        # 按时间排序
        result_timeline.sort(key=lambda x: x["start"])
        
        # 再次检查是否还有短片段
        final_short_count = sum(1 for segment in result_timeline if segment.get("duration", 0) < min_duration)
        if final_short_count > 0:
            logger.warning(f"合并后仍有 {final_short_count} 个短片段无法合并")
        
        logger.info(f"片段合并完成: 从 {len(timeline)} 个合并到 {len(result_timeline)} 个")
        
        return result_timeline
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的语义相似度"""
        embeddings = self.model.encode([text1, text2], convert_to_numpy=True)
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def find_best_slide_for_text(
        self,
        text: str,
        slides: List[Dict[str, Any]],
        threshold: float = 0.0
    ) -> Tuple[int, float]:
        """为给定文本找到最匹配的幻灯片
        
        Returns:
            (幻灯片索引, 相似度)，如果没有匹配则返回(-1, 0.0)
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
        min_display_duration: float = 2.0
    ) -> List[Dict[str, Any]]:
        """创建均匀分布的时间轴（备用策略）
        
        当语义对齐失败时，将幻灯片均匀分布在字幕时间范围内
        
        Args:
            slides: 幻灯片列表
            subtitles: 字幕片段列表
            min_display_duration: 最小展示时长（秒）
            
        Returns:
            时间轴映射列表，每个元素包含start, end, slide_index
        """
        if not slides or not subtitles:
            logger.warning("备用策略：幻灯片或字幕列表为空，无法创建时间轴")
            return []
        
        logger.info("语义对齐失败，使用均匀分布备用策略")
        
        # 计算字幕总时长
        if len(subtitles) == 0:
            logger.warning("备用策略：字幕列表为空")
            return []
            
        total_duration = subtitles[-1]["end"] - subtitles[0]["start"]
        if total_duration <= 0:
            logger.warning(f"备用策略：字幕时长无效（{total_duration}秒）")
            total_duration = len(slides) * min_display_duration  # 使用默认时长
        
        # 均匀分配时间给每张幻灯片，确保至少min_display_duration秒
        slide_duration = max(min_display_duration, total_duration / max(1, len(slides)))
        
        # 如果总时长不足，调整slide_duration
        if slide_duration * len(slides) > total_duration:
            slide_duration = total_duration / max(1, len(slides))
        
        # 确保slide_duration至少为0.1秒，防止除零或无效时长
        slide_duration = max(0.1, slide_duration)
        
        logger.debug(f"备用策略参数: total_duration={total_duration:.2f}s, slides={len(slides)}, "
                    f"min_display_duration={min_display_duration}s, calculated_slide_duration={slide_duration:.2f}s")
        
        timeline = []
        current_time = subtitles[0]["start"]
        
        for i, slide in enumerate(slides):
            end_time = current_time + slide_duration
            
            # 确保不超过总时长
            if end_time > subtitles[-1]["end"]:
                end_time = subtitles[-1]["end"]
            
            # 如果剩余时间太少，调整结束时间
            if end_time - current_time < min_display_duration / 2:
                # 时间太少，合并到前一张幻灯片（如果是第一张则扩展）
                if timeline:
                    last_segment = timeline[-1]
                    last_segment["end"] = subtitles[-1]["end"]
                    last_segment["duration"] = last_segment["end"] - last_segment["start"]
                    logger.debug(f"备用策略：第{i}张幻灯片时间不足，合并到前一张")
                else:
                    # 第一张幻灯片，至少保证最小时长
                    end_time = current_time + min_display_duration
                    if end_time > subtitles[-1]["end"]:
                        end_time = subtitles[-1]["end"]
                break
            
            # 计算实际时长并验证
            actual_duration = end_time - current_time
            if actual_duration <= 0:
                logger.warning(f"备用策略：片段{i}时长无效({actual_duration:.2f}s)，跳过此片段")
                continue
                
            if end_time <= current_time:
                logger.warning(f"备用策略：片段{i}结束时间不大于开始时间，调整结束时间")
                end_time = current_time + slide_duration
            
            timeline.append({
                "start": current_time,
                "end": end_time,
                "slide_index": i,
                "duration": end_time - current_time,
                "similarity": 0.0,
                "strategy": "uniform_fallback",
                "note": f"备用策略生成，原始索引: {slide.get('index', i)}"
            })
            
            current_time = end_time
        
        # 验证生成的时间轴
        validated_timeline = self._validate_timeline(timeline)
        
        if not validated_timeline:
            logger.error("备用策略生成的时间轴无效，返回空列表")
            return []
        
        logger.info(f"备用策略生成{len(validated_timeline)}个有效时间片段，每片约{slide_duration:.1f}秒")
        return validated_timeline
    
    def _validate_timeline(self, timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证时间轴有效性，过滤无效片段
        
        Args:
            timeline: 时间轴列表
            
        Returns:
            验证后的时间轴列表，只包含有效片段
        """
        if not timeline:
            logger.warning("时间轴为空，无需验证")
            return []
        
        valid_timeline = []
        invalid_count = 0
        
        for i, segment in enumerate(timeline):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            duration = end_time - start_time
            
            # 验证时间片段有效性
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
                logger.debug(f"时间片段{i}有效: slide_index={slide_index}, duration={duration:.2f}s")
            else:
                invalid_count += 1
                logger.warning(f"时间片段{i}无效: {', '.join(issues)}")
        
        if invalid_count > 0:
            logger.warning(f"时间轴验证: {len(valid_timeline)}个有效片段, {invalid_count}个无效片段被过滤")
        
        if not valid_timeline:
            logger.error("时间轴验证: 所有片段均无效")
        
        return valid_timeline
