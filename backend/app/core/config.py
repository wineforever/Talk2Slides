import platform
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Base settings
    APP_NAME: str = "Video Generation API"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    TEMP_DIR: Path = BASE_DIR / "temp"
    OUTPUT_DIR: Path = BASE_DIR.parent / "output"
    UPLOAD_DIR: Path = TEMP_DIR / "uploads"

    # Model
    SENTENCE_TRANSFORMER_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
    SENTENCE_TRANSFORMER_DEVICE: str = ""
    SENTENCE_TRANSFORMER_CACHE_DIR: str = ""
    SENTENCE_TRANSFORMER_LOCAL_FILES_ONLY: bool = False
    SENTENCE_TRANSFORMER_ALLOW_REMOTE_DOWNLOAD: bool = True
    SENTENCE_TRANSFORMER_FALLBACK_TO_LEXICAL: bool = True
    SENTENCE_TRANSFORMER_LOCAL_MODEL_DIR: Path = BASE_DIR / "app" / "models"
    SENTENCE_TRANSFORMER_BOOTSTRAP_ON_STARTUP: bool = True
    SENTENCE_TRANSFORMER_BOOTSTRAP_DOWNLOAD_IF_MISSING: bool = True
    SENTENCE_TRANSFORMER_MODEL_REVISION: str = "main"
    SENTENCE_TRANSFORMER_HF_ENDPOINT: str = ""
    SENTENCE_TRANSFORMER_HF_ETAG_TIMEOUT_SEC: int = 6
    SENTENCE_TRANSFORMER_HF_DOWNLOAD_TIMEOUT_SEC: int = 10
    SENTENCE_TRANSFORMER_ENDPOINT_CHECK_TIMEOUT_SEC: float = 2.5
    SENTENCE_TRANSFORMER_SKIP_REMOTE_WHEN_ENDPOINT_UNREACHABLE: bool = True

    # SRT preprocess
    SRT_MERGE_GAP_SEC: float = 0.6
    SRT_MIN_CHARS: int = 6
    SRT_MIN_DURATION_SEC: float = 0.8
    SRT_MAX_CHARS: int = 120
    SRT_FILLER_PATTERNS: str = "嗯|啊|呃|就是|然后|其实|这个|那个|你知道吗|对|好|嗯嗯"

    # Alignment constraints
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
    ALIGN_STRUCTURE_PRIOR_WEIGHT: float = 0.16
    ALIGN_LOW_CONFIDENCE_MARGIN: float = 0.08
    ALIGN_BOUNDARY_CROSS_MIN: float = 0.02
    ALIGN_ENFORCE_NO_REVISIT: bool = True
    ALIGN_MIN_SLIDE_USAGE_RATIO: float = 0.70
    ALIGN_ENFORCE_SEQUENTIAL: bool = False
    ALIGN_REQUIRE_FULL_COVERAGE: bool = False
    ALIGN_KEEP_SHORT_SEGMENTS_FOR_COVERAGE: bool = False

    # Runtime defaults
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.5
    DEFAULT_MIN_DISPLAY_DURATION: float = 2.0
    DEFAULT_OUTPUT_RESOLUTION: str = "1920x1080"
    PPT_EXPORT_DPI: int = 300
    VIDEO_FORCE_FIRST_SLIDE_FRAME: bool = True
    VIDEO_FORCE_LAST_SLIDE_TAIL_SEC: float = 5.0
    VIDEO_EMBED_PROGRESS_BAR: bool = True
    VIDEO_BURN_SRT_SUBTITLES: bool = True
    VIDEO_PROGRESS_MAX_SEGMENTS: int = 10
    VIDEO_PROGRESS_LABEL_MAX_CHARS: int = 10

    # External tools
    FFMPEG_PATH: str = Field(
        default="ffmpeg",
        description="Path to ffmpeg executable",
    )
    LIBREOFFICE_PATH: str = Field(
        default="soffice.exe" if platform.system() == "Windows" else "soffice",
        description="Path to LibreOffice executable",
    )

    # Task lifecycle
    MAX_TASK_AGE_HOURS: int = 24

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
