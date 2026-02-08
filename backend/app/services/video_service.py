import os
import subprocess
import tempfile
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import json
import ffmpeg
import shutil

from app.core.config import settings

logger = logging.getLogger(__name__)

class VideoService:
    """视频合成服务"""
    
    def _validate_image_files(self, image_paths: List[str]) -> bool:
        """验证所有图片文件是否存在且可访问
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            True如果所有文件都存在且可访问，否则False
        """
        for i, path in enumerate(image_paths):
            path_obj = Path(path)
            if not path_obj.exists():
                logger.error(f"图片文件不存在: {path} (索引: {i})")
                return False
            if not path_obj.is_file():
                logger.error(f"不是有效的文件: {path} (索引: {i})")
                return False
            # 检查文件是否可读
            try:
                with open(path_obj, 'rb') as f:
                    f.read(1)
            except Exception as e:
                logger.error(f"无法读取文件: {path} (索引: {i}), 错误: {e}")
                return False
        return True
    
    def _normalize_path_for_ffmpeg(self, path: str) -> str:
        """为FFmpeg规范化文件路径
        
        将Windows反斜杠转换为正斜杠，确保路径格式正确
        
        Args:
            path: 原始文件路径
            
        Returns:
            规范化后的路径
        """
        # 使用Path对象解析路径
        path_obj = Path(path).resolve()
        # 转换为字符串并用正斜杠替换反斜杠
        normalized = str(path_obj).replace('\\', '/')
        logger.debug(f"路径规范化: {path} -> {normalized}")
        return normalized
    
    def create_video_from_timeline(
        self,
        image_paths: List[str],
        timeline: List[Dict[str, Any]],
        audio_path: str,
        output_path: str,
        resolution: str = "1920x1080",
        fps: int = 30
    ) -> str:
        """根据时间轴创建视频
        
        Args:
            image_paths: 图片路径列表，按幻灯片顺序排列
            timeline: 时间轴映射列表，每个元素包含start, end, slide_index
            audio_path: 音频文件路径
            output_path: 输出视频路径
            resolution: 视频分辨率，格式为"宽x高"
            fps: 视频帧率
            
        Returns:
            输出视频路径
        """
        # 验证输入参数
        if not image_paths:
            raise ValueError("图片路径列表不能为空")
        if not timeline:
            # 分析时间轴为空的原因，提供详细错误信息
            error_msg = "时间轴为空，可能原因：\n"
            error_msg += f"- 语义对齐失败，{len(image_paths)}个图片可用\n"
            error_msg += "- 请检查以下可能原因：\n"
            error_msg += "  1. PPT和字幕内容不相关（语义相似度过低）\n"
            error_msg += "  2. 相似度阈值设置过高（默认0.5）\n"
            error_msg += "  3. 最小展示时长设置过长（默认2.0秒）\n"
            error_msg += "  4. PPT或SRT文件解析失败\n"
            error_msg += "\n建议解决方案：\n"
            error_msg += "1. 降低similarity_threshold参数（如0.3）\n"
            error_msg += "2. 降低min_display_duration参数（如1.0）\n"
            error_msg += "3. 检查PPT和SRT文件内容是否相关\n"
            error_msg += "4. 查看服务器日志获取详细诊断信息\n"
            raise ValueError(error_msg)
        
        # 验证图片文件是否存在且可访问
        logger.info(f"开始验证{len(image_paths)}个图片文件")
        if not self._validate_image_files(image_paths):
            raise Exception("部分图片文件不存在或无法访问，请检查PPT处理结果")
        logger.info("所有图片文件验证通过")
        
        # 记录输出路径信息用于调试
        output_path_obj = Path(output_path)
        logger.info(f"视频生成参数: 输出路径={output_path_obj.resolve()}, "
                   f"分辨率={resolution}, 帧率={fps}fps, "
                   f"时间轴片段数={len(timeline)}, 图片数={len(image_paths)}")
        
        try:
            width, height = map(int, resolution.split('x'))
            logger.info(f"视频分辨率: {width}x{height}, 帧率: {fps}fps")
            
            # 1. 准备临时目录 - 使用不含中文字符的路径
            # 优先使用项目临时目录，避免用户目录中的中文字符
            base_temp_dir = Path(settings.BASE_DIR) / "temp" / "video_processing"
            base_temp_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = Path(tempfile.mkdtemp(dir=base_temp_dir, prefix=f"video_{int(time.time())}_"))
            logger.info(f"创建临时目录: {temp_dir}")
            
            # 2. 生成FFmpeg concat文件
            concat_file = temp_dir / "concat.txt"
            logger.info(f"生成concat文件: {concat_file}")
            self._create_concat_file(concat_file, image_paths, timeline, fps)
            
            # 记录concat文件内容用于调试
            if concat_file.exists():
                with open(concat_file, 'r', encoding='utf-8') as f:
                    concat_content = f.read()
                    logger.debug(f"concat.txt内容:\n{concat_content}")
            
            # 3. 创建无声视频（图片序列）
            temp_video = temp_dir / "temp_video.mp4"
            logger.info(f"创建无声视频: {temp_video}")
            self._create_silent_video(
                concat_file=concat_file,
                output_path=str(temp_video),
                width=width,
                height=height,
                fps=fps
            )
            
            # 验证视频文件是否创建成功
            temp_video_path = Path(temp_video)
            if not temp_video_path.exists():
                raise Exception(f"无声视频文件未生成: {temp_video}")
            
            video_size = temp_video_path.stat().st_size
            logger.info(f"无声视频创建成功，文件大小: {video_size}字节，路径: {temp_video_path.resolve()}")
            
            # 验证文件大小至少为1KB（视频文件不应该太小）
            MIN_VIDEO_SIZE = 1024  # 1KB
            if video_size < MIN_VIDEO_SIZE:
                logger.warning(f"无声视频文件可能无效: 大小只有{video_size}字节，小于最小要求{MIN_VIDEO_SIZE}字节")
                # 不直接失败，可能短视频确实很小，但记录警告
            
            # 4. 合并音频
            logger.info(f"合并音频: {audio_path}")
            self._merge_audio_video(
                video_path=str(temp_video),
                audio_path=audio_path,
                output_path=output_path
            )
            
            # 验证最终输出文件
            output_path_obj = Path(output_path)
            output_dir = output_path_obj.parent
            
            # 记录输出路径详细信息
            logger.info(f"输出文件检查: 路径={output_path_obj.resolve()}, 目录={output_dir.resolve()}")
            
            # 验证输出目录权限（在_merge_audio_video中已有验证，但这里再检查一次）
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"创建输出目录: {output_dir}")
            
            if not os.access(str(output_dir), os.W_OK):
                raise PermissionError(f"输出目录不可写: {output_dir}")
            
            # 验证文件是否存在
            if not output_path_obj.exists():
                raise Exception(f"最终视频文件未生成: {output_path_obj.resolve()}")
            
            final_size = output_path_obj.stat().st_size
            logger.info(f"最终视频创建成功，文件大小: {final_size}字节，路径: {output_path_obj.resolve()}")
            
            # 验证文件大小至少为1KB
            MIN_FINAL_VIDEO_SIZE = 1024  # 1KB
            if final_size < MIN_FINAL_VIDEO_SIZE:
                logger.warning(f"最终视频文件可能无效: 大小只有{final_size}字节，小于最小要求{MIN_FINAL_VIDEO_SIZE}字节")
                # 不直接失败，但记录严重警告
                logger.error(f"警告: 生成的视频文件异常小，可能FFmpeg处理失败或输入数据有问题")
            
            # 5. 清理临时文件（可选，保留用于调试）
            # shutil.rmtree(temp_dir, ignore_errors=True)
            # logger.info(f"清理临时目录: {temp_dir}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"视频合成失败: {str(e)}", exc_info=True)
            # 在异常中包含更多上下文信息
            raise Exception(f"视频合成失败: {str(e)}\n临时目录: {temp_dir if 'temp_dir' in locals() else '未创建'}")
    
    def _create_concat_file(
        self,
        concat_file: Path,
        image_paths: List[str],
        timeline: List[Dict[str, Any]],
        fps: int
    ):
        """创建FFmpeg concat文件
        
        格式:
        file 'slide_001.png'
        duration 5.0
        file 'slide_001.png'
        duration 3.0
        ...
        """
        lines = []
        logger.info(f"创建concat文件，包含{len(timeline)}个时间片段，{len(image_paths)}个图片文件")
        
        # 时间轴统计信息
        total_duration = 0.0
        valid_segments = 0
        invalid_segments = []
        
        for i, segment in enumerate(timeline):
            slide_idx = segment["slide_index"]
            start_time = segment["start"]
            end_time = segment["end"]
            duration = end_time - start_time
            
            # 验证时间片段有效性
            if duration <= 0:
                error_msg = f"时间片段{i}时长无效或为负数: start={start_time}, end={end_time}, duration={duration}"
                logger.error(error_msg)
                invalid_segments.append({"index": i, "start": start_time, "end": end_time, "duration": duration, "slide_idx": slide_idx})
                raise ValueError(error_msg)
            
            if end_time <= start_time:
                error_msg = f"时间片段{i}结束时间不大于开始时间: start={start_time}, end={end_time}"
                logger.error(error_msg)
                invalid_segments.append({"index": i, "start": start_time, "end": end_time, "duration": duration, "slide_idx": slide_idx})
                raise ValueError(error_msg)
            
            total_duration += duration
            valid_segments += 1
            
            if slide_idx < 0 or slide_idx >= len(image_paths):
                raise ValueError(f"无效的幻灯片索引: {slide_idx}，有效范围: 0-{len(image_paths)-1}")
            
            image_path = image_paths[slide_idx]
            
            # 验证图片文件是否存在
            if not Path(image_path).exists():
                raise FileNotFoundError(f"图片文件不存在: {image_path} (片段 {i}, 幻灯片索引 {slide_idx})")
            
            # 使用规范化路径（正斜杠，避免编码问题）
            normalized_path = self._normalize_path_for_ffmpeg(image_path)
            
            # 添加文件行
            lines.append(f"file '{normalized_path}'")
            # 添加时长行
            lines.append(f"duration {duration}")
            logger.debug(f"片段 {i}: 幻灯片 {slide_idx}, 时长 {duration:.2f}秒, 文件: {normalized_path}")
        
        # 记录时间轴统计信息
        avg_duration = total_duration / max(1, valid_segments)
        logger.info(f"时间轴统计: 有效片段={valid_segments}/{len(timeline)}, 总时长={total_duration:.2f}秒, 平均时长={avg_duration:.2f}秒")
        
        if invalid_segments:
            logger.warning(f"发现{len(invalid_segments)}个无效时间片段: {invalid_segments}")
        
        # 最后需要再添加一次文件（FFmpeg concat的要求）
        if timeline:
            last_slide_idx = timeline[-1]["slide_index"]
            last_image_path = image_paths[last_slide_idx]
            normalized_last_path = self._normalize_path_for_ffmpeg(last_image_path)
            lines.append(f"file '{normalized_last_path}'")
            logger.debug(f"最后一行（FFmpeg要求）: 文件: {normalized_last_path}")
        
        # 写入文件
        content = '\n'.join(lines)
        with open(concat_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"concat文件创建成功: {concat_file}, 共{len(lines)}行")
        return content
    
    def _create_silent_video(
        self,
        concat_file: Path,
        output_path: str,
        width: int,
        height: int,
        fps: int
    ):
        """使用FFmpeg创建无声视频"""
        logger.info(f"开始创建无声视频，concat文件: {concat_file}, 输出: {output_path}")
        
        # 验证concat文件是否存在
        if not concat_file.exists():
            raise FileNotFoundError(f"concat文件不存在: {concat_file}")
        
        # 读取并记录concat文件内容
        try:
            with open(concat_file, 'r', encoding='utf-8') as f:
                concat_content = f.read()
                logger.debug(f"使用的concat文件内容:\n{concat_content}")
        except Exception as e:
            logger.warning(f"无法读取concat文件内容: {e}")
        
        try:
            # 记录FFmpeg命令详情
            logger.info(f"FFmpeg参数: 分辨率{width}x{height}, 帧率{fps}fps, 输入文件: {concat_file}")
            
            # 使用ffmpeg-python
            (
                ffmpeg
                .input(str(concat_file), format='concat', safe=0)
                .filter('scale', width, height, force_original_aspect_ratio='decrease')
                .filter('pad', width, height, '(ow-iw)/2', '(oh-ih)/2')
                .output(output_path, r=fps, vcodec='libx264', pix_fmt='yuv420p')
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            logger.info(f"无声视频创建成功: {output_path}")
            
        except ffmpeg.Error as e:
            error_message = e.stderr.decode('utf-8') if e.stderr else str(e)
            logger.error(f"FFmpeg错误详情:\n{error_message}")
            
            # 提供更详细的错误分析和建议
            error_analysis = self._analyze_ffmpeg_error(error_message, concat_file)
            raise Exception(f"FFmpeg错误:\n{error_message}\n\n分析建议:\n{error_analysis}")
    
    def _analyze_ffmpeg_error(self, error_message: str, concat_file: Path) -> str:
        """分析FFmpeg错误并提供建议
        
        Args:
            error_message: FFmpeg错误信息
            concat_file: concat文件路径
            
        Returns:
            分析建议字符串
        """
        analysis = []
        
        # 检查常见错误模式
        if "Invalid data found when processing input" in error_message:
            analysis.append("1. concat文件格式可能不正确")
            analysis.append("2. 图片文件路径可能包含特殊字符或编码问题")
            analysis.append("3. 图片文件可能损坏或格式不支持")
            
            # 检查concat文件
            if concat_file.exists():
                try:
                    with open(concat_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        analysis.append(f"4. concat文件共{len(lines)}行，首行: {lines[0][:50] if lines else '空文件'}")
                except:
                    analysis.append("4. 无法读取concat文件内容")
        
        elif "No such file or directory" in error_message:
            analysis.append("1. 图片文件不存在或路径错误")
            analysis.append("2. 请检查concat文件中的文件路径是否正确")
            
        elif "Permission denied" in error_message:
            analysis.append("1. 文件权限问题")
            analysis.append("2. 请检查输出目录是否可写")
            
        else:
            analysis.append("1. 未知FFmpeg错误")
            analysis.append("2. 请检查FFmpeg安装和版本")
            analysis.append(f"3. 错误关键词: {error_message[:100]}...")
        
        # 通用建议
        analysis.append("\n通用解决步骤:")
        analysis.append("1. 确保所有图片文件存在且可读")
        analysis.append("2. 检查文件路径是否包含中文字符或特殊符号")
        analysis.append("3. 尝试使用英文路径和文件名")
        analysis.append("4. 验证FFmpeg安装: ffmpeg -version")
        analysis.append("5. 检查临时目录权限")
        
        return '\n'.join(analysis)
    
    def _merge_audio_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str
    ):
        """合并视频和音频"""
        logger.info(f"开始合并音视频，视频: {video_path}, 音频: {audio_path}, 输出: {output_path}")
        
        # 验证输入文件是否存在
        if not Path(video_path).exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        # 验证输出目录可写
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(str(output_dir), os.W_OK):
            raise PermissionError(f"输出目录不可写: {output_dir}")
        
        try:
            # 使用ffmpeg-python合并音频
            video = ffmpeg.input(video_path)
            audio = ffmpeg.input(audio_path)
            
            logger.info(f"FFmpeg合并命令: 视频编码copy, 音频编码aac, 输出: {output_path}")
            
            (
                ffmpeg
                .output(video, audio, output_path, 
                       vcodec='copy', acodec='aac', 
                       strict='experimental', shortest=None)
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            logger.info(f"音视频合并成功: {output_path}")
            
        except ffmpeg.Error as e:
            error_message = e.stderr.decode('utf-8') if e.stderr else str(e)
            logger.error(f"音视频合并FFmpeg错误:\n{error_message}")
            
            # 分析音频合并特定错误
            error_analysis = []
            if "Invalid data found when processing input" in error_message:
                error_analysis.append("1. 音频文件可能损坏或格式不支持")
                error_analysis.append("2. 视频文件可能损坏")
                error_analysis.append("3. 尝试使用其他音频格式（如MP3、WAV）")
            elif "No such file or directory" in error_message:
                error_analysis.append("1. 输入文件路径错误")
                error_analysis.append("2. 文件权限问题")
            elif "Permission denied" in error_message:
                error_analysis.append("1. 输出目录权限问题")
                error_analysis.append("2. 尝试使用其他输出路径")
            
            if not error_analysis:
                error_analysis.append("1. 未知音视频合并错误")
                error_analysis.append("2. 请检查FFmpeg版本和支持的编码器")
            
            error_analysis.append("\n解决步骤:")
            error_analysis.append("1. 验证音频文件: ffmpeg -i [音频文件路径]")
            error_analysis.append("2. 验证视频文件: ffmpeg -i video.mp4")
            error_analysis.append("3. 检查磁盘空间")
            error_analysis.append("4. 尝试简化输出路径（不含特殊字符）")
            
            raise Exception(f"音频合并失败:\n{error_message}\n\n建议:\n{'\n'.join(error_analysis)}")
    
    def extract_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """提取音频信息"""
        try:
            probe = ffmpeg.probe(audio_path)
            audio_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'),
                None
            )
            
            if not audio_stream:
                raise Exception("未找到音频流")
            
            return {
                "duration": float(audio_stream.get('duration', 0)),
                "codec": audio_stream.get('codec_name', 'unknown'),
                "bit_rate": audio_stream.get('bit_rate', 'unknown'),
                "sample_rate": audio_stream.get('sample_rate', 'unknown'),
                "channels": audio_stream.get('channels', 'unknown')
            }
        except Exception as e:
            raise Exception(f"音频信息提取失败: {str(e)}")
    
    def validate_video(self, video_path: str) -> bool:
        """验证视频文件是否有效"""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
                None
            )
            return video_stream is not None
        except:
            return False
    
    def get_video_duration(self, video_path: str) -> float:
        """获取视频时长（秒）"""
        try:
            probe = ffmpeg.probe(video_path)
            duration = float(probe['format']['duration'])
            return duration
        except Exception as e:
            raise Exception(f"获取视频时长失败: {str(e)}")
    
    def create_preview_video(
        self,
        image_paths: List[str],
        timeline: List[Dict[str, Any]],
        output_path: str,
        resolution: str = "640x360",
        fps: int = 15
    ) -> str:
        """创建预览视频（低分辨率，用于快速预览）"""
        return self.create_video_from_timeline(
            image_paths=image_paths,
            timeline=timeline,
            audio_path=None,  # 预览视频不需要音频
            output_path=output_path,
            resolution=resolution,
            fps=fps
        )
    
    def convert_video_format(
        self,
        input_path: str,
        output_path: str,
        format: str = "mp4"
    ) -> str:
        """转换视频格式"""
        try:
            (
                ffmpeg
                .input(input_path)
                .output(output_path, vcodec='libx264', acodec='aac')
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            return output_path
        except ffmpeg.Error as e:
            error_message = e.stderr.decode('utf-8') if e.stderr else str(e)
            raise Exception(f"视频格式转换失败: {error_message}")