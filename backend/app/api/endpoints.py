from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, Request, Response
from fastapi.responses import JSONResponse, FileResponse
import os
import uuid
import shutil
import logging
from typing import Optional
import asyncio
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

from app.core.config import settings
from app.services.task_manager import TaskManager, TaskStatus
from app.services.ppt_service import PPTService
from app.services.srt_service import SRTService
from app.services.alignment_service import AlignmentService
from app.services.video_service import VideoService

router = APIRouter()
task_manager = TaskManager()

@router.options("/upload")
async def options_upload():
    """处理CORS预检请求"""
    return Response(status_code=200)

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
    align_forward_jump_penalty: float = Form(settings.ALIGN_FORWARD_JUMP_PENALTY)
):
    """上传文件并启动视频生成任务"""
    
    # 验证文件类型
    if not pptx.filename.endswith('.pptx'):
        raise HTTPException(status_code=400, detail="PPTX文件必须是.pptx格式")
    
    # 验证音频文件类型（支持MP3和WAV格式）
    allowed_audio_extensions = ['.mp3', '.wav', '.wave']
    if not any(mp3.filename.lower().endswith(ext) for ext in allowed_audio_extensions):
        raise HTTPException(status_code=400, detail="音频文件必须是.mp3或.wav格式")
    
    if not srt.filename.endswith('.srt'):
        raise HTTPException(status_code=400, detail="字幕文件必须是.srt格式")
    
    # 创建任务ID
    task_id = str(uuid.uuid4())
    
    # 创建任务目录
    task_dir = settings.TEMP_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存上传的文件
    pptx_path = task_dir / "presentation.pptx"
    
    # 根据音频文件扩展名动态生成文件名
    audio_ext = Path(mp3.filename).suffix.lower()
    if audio_ext not in allowed_audio_extensions:
        audio_ext = '.mp3'  # 默认后缀
    mp3_path = task_dir / f"audio{audio_ext}"
    
    srt_path = task_dir / "subtitles.srt"
    
    try:
        # 保存文件
        with open(pptx_path, "wb") as f:
            content = await pptx.read()
            f.write(content)
        
        with open(mp3_path, "wb") as f:
            content = await mp3.read()
            f.write(content)
        
        with open(srt_path, "wb") as f:
            content = await srt.read()
            f.write(content)
        
        # 创建任务
        task = task_manager.create_task(
            task_id=task_id,
            pptx_path=str(pptx_path),
            mp3_path=str(mp3_path),
            srt_path=str(srt_path),
            similarity_threshold=similarity_threshold,
            min_display_duration=min_display_duration,
            output_resolution=output_resolution
        )
        
        # 在后台启动处理任务
        background_tasks.add_task(
            process_video_generation,
            task_id,
            str(pptx_path),
            str(mp3_path),
            str(srt_path),
            similarity_threshold,
            min_display_duration,
            output_resolution,
            srt_merge_gap_sec,
            srt_min_duration_sec,
            align_max_backtrack,
            align_max_forward_jump,
            align_switch_penalty,
            align_forward_jump_penalty
        )
        
        return {
            "task_id": task_id,
            "status": "pending",
            "message": "文件上传成功，任务已开始处理"
        }
        
    except Exception as e:
        # 清理任务目录
        shutil.rmtree(task_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

@router.get("/task/{task_id}/status")
async def get_task_status(task_id: str):
    """获取任务状态"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return {
        "task_id": task_id,
        "state": task.state,
        "progress": task.progress,
        "message": task.message,
        "error": task.error,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "updated_at": task.updated_at.isoformat() if task.updated_at else None
    }

@router.get("/task/{task_id}/result")
async def get_task_result(task_id: str):
    """获取任务结果（视频文件）"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task.state != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="任务尚未完成")
    
    video_path = task.result.get("video_path") if task.result else None
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="视频文件不存在")
    
    # 返回文件，使用实际生成的文件名
    filename = os.path.basename(video_path)
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=filename
    )

@router.get("/task/{task_id}/preview")
async def get_task_preview(task_id: str):
    """获取任务预览信息（如时间轴映射）"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task.state != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="任务尚未完成")
    
    # 提取视频文件名
    video_path = task.result.get("video_path") if task.result else None
    filename = os.path.basename(video_path) if video_path else "generated_video.mp4"
    
    return {
        "task_id": task_id,
        "timeline": task.result.get("timeline", []) if task.result else [],
        "video_url": f"/api/task/{task_id}/result",
        "filename": filename
    }

@router.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """删除任务及关联文件"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 清理任务目录
    task_dir = settings.TEMP_DIR / task_id
    output_dir = settings.OUTPUT_DIR / task_id
    
    try:
        shutil.rmtree(task_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        task_manager.delete_task(task_id)
        return {"message": "任务删除成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

async def process_video_generation(
    task_id: str,
    pptx_path: str,
    mp3_path: str,
    srt_path: str,
    similarity_threshold: float,
    min_display_duration: float,
    output_resolution: str,
    srt_merge_gap_sec: float,
    srt_min_duration_sec: float,
    align_max_backtrack: int,
    align_max_forward_jump: int,
    align_switch_penalty: float,
    align_forward_jump_penalty: float
):
    """视频生成处理流程"""
    
    # 参数验证和规范化
    # 限制相似度阈值在合理范围（0.1-0.9）
    similarity_threshold = min(max(similarity_threshold, 0.1), 0.9)
    # 确保最小展示时长至少1秒
    min_display_duration = max(min_display_duration, 1.0)
    # SRT预处理参数约束
    srt_merge_gap_sec = min(max(srt_merge_gap_sec, 0.0), 3.0)
    srt_min_duration_sec = min(max(srt_min_duration_sec, 0.1), 5.0)
    # 对齐参数约束
    align_max_backtrack = int(min(max(align_max_backtrack, 0), 5))
    align_max_forward_jump = int(min(max(align_max_forward_jump, 0), 10))
    align_switch_penalty = min(max(align_switch_penalty, 0.0), 1.0)
    align_forward_jump_penalty = min(max(align_forward_jump_penalty, 0.0), 1.0)
    
    logger.info(
        "视频生成参数: "
        f"similarity_threshold={similarity_threshold}, "
        f"min_display_duration={min_display_duration}, "
        f"output_resolution={output_resolution}, "
        f"srt_merge_gap_sec={srt_merge_gap_sec}, "
        f"srt_min_duration_sec={srt_min_duration_sec}, "
        f"align_max_backtrack={align_max_backtrack}, "
        f"align_max_forward_jump={align_max_forward_jump}, "
        f"align_switch_penalty={align_switch_penalty}, "
        f"align_forward_jump_penalty={align_forward_jump_penalty}"
    )
    
    # 阶段标记，用于错误诊断
    processing_stage = "initialization"
    
    try:
        task_manager.update_task(task_id, state=TaskStatus.PROCESSING, progress=0, message="开始处理...")
        
        # 1. 解析PPT
        processing_stage = "ppt_parsing"
        task_manager.update_task(task_id, progress=10, message="解析PPT文件...")
        ppt_service = PPTService()
        slides = ppt_service.extract_slides(pptx_path)
        
        if not slides:
            error_msg = f"PPT解析失败或无幻灯片内容\n文件路径: {pptx_path}\n可能原因："
            error_msg += "\n1. PPT文件为空或损坏"
            error_msg += "\n2. PPT文件格式不支持（仅支持.pptx格式）"
            error_msg += "\n3. python-pptx库安装问题"
            error_msg += "\n4. 文件权限问题"
            raise ValueError(error_msg)
        
        logger.info(f"成功解析{len(slides)}张幻灯片")
        
        # 2. 解析字幕
        processing_stage = "srt_parsing"
        task_manager.update_task(task_id, progress=20, message="解析字幕文件...")
        srt_service = SRTService()
        subtitles = srt_service.parse_srt(srt_path)
        
        if not subtitles:
            error_msg = f"SRT解析失败或无字幕内容\n文件路径: {srt_path}\n可能原因："
            error_msg += "\n1. SRT文件为空或损坏"
            error_msg += "\n2. SRT文件格式不正确"
            error_msg += "\n3. pysrt库安装问题"
            error_msg += "\n4. 文件编码问题（请使用UTF-8编码）"
            raise ValueError(error_msg)
        
        logger.info(f"成功解析{len(subtitles)}个字幕片段")

        # 2.1 预处理字幕（合并/清洗）
        segments = srt_service.preprocess_subtitles(
            subtitles=subtitles,
            merge_gap_sec=srt_merge_gap_sec,
            min_chars=settings.SRT_MIN_CHARS,
            min_duration_sec=srt_min_duration_sec,
            max_chars=settings.SRT_MAX_CHARS,
            filler_patterns=settings.SRT_FILLER_PATTERNS
        )
        logger.info(
            f"SRT预处理完成: 原始片段={len(subtitles)}, 处理后片段={len(segments)}"
        )
        
        # 3. 语义对齐
        processing_stage = "alignment"
        task_manager.update_task(task_id, progress=30, message="执行语义对齐...")
        alignment_service = AlignmentService()
        timeline = alignment_service.align_slides_with_subtitles(
            slides=slides,
            subtitles=segments,
            similarity_threshold=similarity_threshold,
            min_display_duration=min_display_duration,
            align_max_backtrack=align_max_backtrack,
            align_max_forward_jump=align_max_forward_jump,
            align_switch_penalty=align_switch_penalty,
            align_forward_jump_penalty=align_forward_jump_penalty
        )
        
        if not timeline:
            error_msg = f"语义对齐失败，无法生成时间轴\n"
            error_msg += f"幻灯片数: {len(slides)}, 字幕数: {len(subtitles)}, 处理后片段数: {len(segments)}\n"
            error_msg += f"参数: similarity_threshold={similarity_threshold}, min_display_duration={min_display_duration}\n\n"
            error_msg += "可能原因：\n"
            error_msg += "1. PPT和字幕内容不相关（语义相似度过低）\n"
            error_msg += "2. 相似度阈值设置过高（建议尝试0.3）\n"
            error_msg += "3. 最小展示时长设置过长（建议尝试1.0）\n"
            error_msg += "4. 算法约束过于严格（单调不减约束）\n\n"
            error_msg += "已尝试备用策略（均匀分布），但仍失败\n"
            error_msg += "请检查输入文件内容或调整参数重试"
            raise ValueError(error_msg)
        
        logger.info(f"语义对齐成功，生成{len(timeline)}个时间片段")
        
        # 4. 导出PPT为图片
        processing_stage = "ppt_to_images"
        task_manager.update_task(task_id, progress=50, message="导出PPT图片...")
        image_dir = settings.TEMP_DIR / task_id / "images"
        image_dir.mkdir(exist_ok=True)
        image_paths = ppt_service.export_slides_to_images(
            pptx_path=pptx_path,
            output_dir=str(image_dir),
            resolution=output_resolution
        )
        
        # 5. 生成视频
        processing_stage = "video_synthesis"
        task_manager.update_task(task_id, progress=70, message="合成视频...")
        video_service = VideoService()
        
        # 生成包含日期时间的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_{task_id}_{timestamp}.mp4"
        output_video_path = settings.OUTPUT_DIR / task_id / filename
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        video_service.create_video_from_timeline(
            image_paths=image_paths,
            timeline=timeline,
            audio_path=mp3_path,
            output_path=str(output_video_path),
            resolution=output_resolution
        )
        
        # 6. 完成任务
        task_manager.update_task(
            task_id,
            state=TaskStatus.COMPLETED,
            progress=100,
            message="视频生成完成",
            result={
                "video_path": str(output_video_path),
                "timeline": timeline,
                "slide_count": len(slides),
                "subtitle_count": len(subtitles),
                "segment_count": len(segments)
            }
        )
        
    except Exception as e:
        # 记录详细错误信息
        error_detail = str(e)
        logger.error(f"视频生成失败: 阶段={processing_stage}, 错误: {error_detail}", exc_info=True)
        
        # 增强错误信息，提供处理阶段上下文
        enhanced_error = f"视频生成失败: {error_detail}\n\n"
        enhanced_error += "处理流程: PPT解析 → SRT解析 → 语义对齐 → PPT转图片 → 视频合成\n"
        enhanced_error += f"失败阶段: {processing_stage}\n\n"
        
        # 根据阶段提供针对性的错误信息和建议
        stage_descriptions = {
            "initialization": "初始化阶段",
            "ppt_parsing": "PPT解析阶段",
            "srt_parsing": "SRT字幕解析阶段", 
            "alignment": "语义对齐阶段",
            "ppt_to_images": "PPT转图片阶段",
            "video_synthesis": "视频合成阶段"
        }
        
        stage_name = stage_descriptions.get(processing_stage, f"未知阶段({processing_stage})")
        enhanced_error += f"阶段描述: {stage_name}\n\n"
        
        # 阶段特定的建议
        if processing_stage == "ppt_parsing":
            enhanced_error += "可能原因:\n"
            enhanced_error += "1. PPT文件为空或损坏\n"
            enhanced_error += "2. PPT文件格式不支持（仅支持.pptx格式）\n"
            enhanced_error += "3. python-pptx库安装问题\n"
            enhanced_error += "4. 文件权限问题\n\n"
            enhanced_error += "建议:\n"
            enhanced_error += "1. 检查PPT文件是否可以正常打开\n"
            enhanced_error += "2. 确认文件格式为.pptx\n"
            enhanced_error += "3. 重新上传文件尝试\n"
            
        elif processing_stage == "srt_parsing":
            enhanced_error += "可能原因:\n"
            enhanced_error += "1. SRT文件为空或损坏\n"
            enhanced_error += "2. SRT文件格式不正确\n"
            enhanced_error += "3. pysrt库安装问题\n"
            enhanced_error += "4. 文件编码问题（请使用UTF-8编码）\n\n"
            enhanced_error += "建议:\n"
            enhanced_error += "1. 检查SRT文件格式是否正确\n"
            enhanced_error += "2. 使用UTF-8编码保存文件\n"
            enhanced_error += "3. 重新上传文件尝试\n"
            
        elif processing_stage == "alignment":
            enhanced_error += "可能原因:\n"
            enhanced_error += "1. PPT和字幕内容不相关（语义相似度过低）\n"
            enhanced_error += "2. 相似度阈值设置过高（建议尝试0.3）\n"
            enhanced_error += "3. 最小展示时长设置过长（建议尝试1.0）\n"
            enhanced_error += "4. 算法约束过于严格（单调不减约束）\n\n"
            enhanced_error += "建议:\n"
            enhanced_error += "1. 调整similarity_threshold参数（如0.3）\n"
            enhanced_error += "2. 调整min_display_duration参数（如1.0）\n"
            enhanced_error += "3. 检查PPT和SRT文件内容是否相关\n"
            enhanced_error += "4. 系统已尝试备用策略（均匀分布）\n"
            
        elif processing_stage == "ppt_to_images":
            enhanced_error += "可能原因:\n"
            enhanced_error += "1. LibreOffice未安装或路径不正确\n"
            enhanced_error += "2. Poppler未安装（Windows需要额外安装）\n"
            enhanced_error += "3. PPT文件内容问题\n"
            enhanced_error += "4. 临时目录权限问题\n\n"
            enhanced_error += "建议:\n"
            enhanced_error += "1. 检查LibreOffice安装\n"
            enhanced_error += "2. Windows用户需要安装Poppler\n"
            enhanced_error += "3. 检查PPT文件内容是否正常\n"
            
        elif processing_stage == "video_synthesis":
            enhanced_error += "可能原因:\n"
            enhanced_error += "1. FFmpeg未安装或路径不正确\n"
            enhanced_error += "2. 图片文件损坏或格式不支持\n"
            enhanced_error += "3. 时间轴数据无效（时长<=0）\n"
            enhanced_error += "4. 输出目录权限问题\n"
            enhanced_error += "5. 临时目录空间不足\n\n"
            enhanced_error += "建议:\n"
            enhanced_error += "1. 检查FFmpeg安装: 在命令行运行 'ffmpeg -version'\n"
            enhanced_error += "2. 检查PPT文件内容是否正常\n"
            enhanced_error += "3. 检查服务器磁盘空间\n"
            enhanced_error += "4. 查看详细日志获取具体错误信息\n"
            enhanced_error += "5. 系统已添加时间轴验证，无效时长会被检测\n"
            
        else:
            enhanced_error += "未知阶段，请查看详细日志\n"
            enhanced_error += "建议: 检查服务器日志获取更多信息\n"
            enhanced_error += f"原始错误: {error_detail}"
        
        # 更新任务状态为失败
        task_manager.update_task(
            task_id,
            state=TaskStatus.FAILED,
            progress=0,
            error=enhanced_error,
            message="处理失败"
        )
        raise Exception(enhanced_error) from e
