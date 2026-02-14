import os
import platform
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """应用配置"""
    
    # 基础配置
    APP_NAME: str = "Video Generation API"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # 路径配置
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    TEMP_DIR: Path = BASE_DIR / "temp"
    OUTPUT_DIR: Path = BASE_DIR.parent / "output"
    UPLOAD_DIR: Path = TEMP_DIR / "uploads"
    
    # 模型配置
    SENTENCE_TRANSFORMER_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # SRT预处理参数
    SRT_MERGE_GAP_SEC: float = 0.6
    SRT_MIN_CHARS: int = 6
    SRT_MIN_DURATION_SEC: float = 0.8
    SRT_MAX_CHARS: int = 120
    SRT_FILLER_PATTERNS: str = "嗯|啊|呃|就是|然后|其实|这个|那个|你知道吗|对|好|嗯嗯"

    # 对齐约束参数
    ALIGN_MAX_BACKTRACK: int = 2
    ALIGN_MAX_FORWARD_JUMP: int = 5
    ALIGN_SWITCH_PENALTY: float = 0.08
    ALIGN_BACKTRACK_PENALTY: float = 0.15
    ALIGN_FORWARD_JUMP_PENALTY: float = 0.03
    ALIGN_SWITCH_DELAY_MS: int = 250
    ALIGN_PAUSE_SWITCH_THRESHOLD_SEC: float = 1.2
    ALIGN_SEMANTIC_WEIGHT: float = 0.70
    ALIGN_LEXICAL_WEIGHT: float = 0.22
    ALIGN_NUMERIC_WEIGHT: float = 0.08
    ALIGN_LOW_CONFIDENCE_MARGIN: float = 0.08
    ALIGN_BOUNDARY_CROSS_MIN: float = 0.02
    ALIGN_ENFORCE_NO_REVISIT: bool = True
    ALIGN_MIN_SLIDE_USAGE_RATIO: float = 0.70
    ALIGN_ENFORCE_SEQUENTIAL: bool = False
    ALIGN_REQUIRE_FULL_COVERAGE: bool = False
    ALIGN_KEEP_SHORT_SEGMENTS_FOR_COVERAGE: bool = False
    
    # 处理参数默认值
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.5
    DEFAULT_MIN_DISPLAY_DURATION: float = 2.0  # 秒
    DEFAULT_OUTPUT_RESOLUTION: str = "1920x1080"
    PPT_EXPORT_DPI: int = 300
    VIDEO_FORCE_FIRST_SLIDE_FRAME: bool = True
    VIDEO_FORCE_LAST_SLIDE_TAIL_SEC: float = 5.0
    
    # 外部工具路径
    FFMPEG_PATH: str = Field(
        default="ffmpeg",
        description="FFmpeg可执行文件路径"
    )
    LIBREOFFICE_PATH: str = Field(
        default="soffice.exe" if platform.system() == "Windows" else "soffice",
        description="LibreOffice可执行文件路径，Windows需要.exe扩展名"
    )
    
    # 任务配置
    MAX_TASK_AGE_HOURS: int = 24  # 任务最大保留时间（小时）
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
