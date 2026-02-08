import os
import subprocess
import tempfile
import platform
from pathlib import Path
from typing import List, Dict, Any
import json
import logging

from pptx import Presentation
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
import shutil

from app.core.config import settings

logger = logging.getLogger(__name__)

class PPTService:
    """PPT处理服务"""
    
    def _get_libreoffice_path(self) -> str:
        """获取LibreOffice可执行文件路径
        
        如果配置的路径是相对路径且不可用，尝试在常见位置查找。
        
        Returns:
            LibreOffice可执行文件完整路径
        """
        configured_path = settings.LIBREOFFICE_PATH
        
        # 如果路径已包含目录分隔符，直接返回
        if os.path.sep in configured_path or ('/' in configured_path and os.path.sep == '/'):
            return configured_path
        
        # 检查配置的路径是否可直接执行（在PATH中）
        try:
            # 使用shutil.which查找可执行文件
            which_path = shutil.which(configured_path)
            if which_path:
                return which_path
        except:
            pass
        
        # 如果在Windows上，尝试常见安装位置
        if platform.system() == "Windows":
            common_paths = [
                r"C:\Program Files\LibreOffice\program\soffice.exe",
                r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
                os.path.expandvars(r"%ProgramFiles%\LibreOffice\program\soffice.exe"),
                os.path.expandvars(r"%ProgramW6432%\LibreOffice\program\soffice.exe"),
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    return path
        
        # 如果都没找到，返回配置的路径（让后续错误处理）
        return configured_path
    
    def _check_poppler_installed(self) -> bool:
        """检查poppler是否安装并可用
        
        Returns:
            True如果poppler已安装并可用，否则False
        """
        try:
            # 尝试运行pdftoppm命令检查poppler是否可用
            result = subprocess.run(
                ["pdftoppm", "-v"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0 or "pdftoppm" in result.stderr or "pdftoppm" in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def extract_slides(self, pptx_path: str) -> List[Dict[str, Any]]:
        """从PPTX文件中提取幻灯片文本内容
        
        Args:
            pptx_path: PPTX文件路径
            
        Returns:
            幻灯片列表，每个幻灯片包含标题、正文、备注和索引
        """
        try:
            presentation = Presentation(pptx_path)
            slides = []
            
            for i, slide in enumerate(presentation.slides):
                slide_data = {
                    "index": i,
                    "title": "",
                    "content": "",
                    "notes": "",
                    "full_text": ""
                }
                
                # 提取标题（通常来自标题占位符）
                title_shapes = []
                content_shapes = []
                
                for shape in slide.shapes:
                    if not shape.has_text_frame:
                        continue
                    
                    text = shape.text.strip()
                    if not text:
                        continue
                    
                    # 简单启发式：判断是否为标题
                    # 1. 形状名称包含"title"
                    # 2. 文本较短且字体较大（这里简化处理）
                    # 3. 使用占位符类型
                    if shape.is_placeholder:
                        ph = shape.placeholder
                        if ph.type == 1:  # 标题
                            title_shapes.append(text)
                        else:
                            content_shapes.append(text)
                    else:
                        # 非占位符形状，根据文本长度判断
                        if len(text.split()) <= 10 and len(text) < 100:
                            title_shapes.append(text)
                        else:
                            content_shapes.append(text)
                
                # 提取备注框内容
                notes_text = []
                if slide.has_notes_slide:
                    notes_slide = slide.notes_slide
                    for shape in notes_slide.shapes:
                        if shape.has_text_frame:
                            text = shape.text.strip()
                            if text:
                                notes_text.append(text)
                
                # 合并标题、内容和备注
                slide_data["title"] = " ".join(title_shapes)
                slide_data["content"] = " ".join(content_shapes)
                slide_data["notes"] = " ".join(notes_text)
                
                # 构建完整文本，备注内容可能很重要，所以包含在内
                full_text_parts = []
                if slide_data["title"]:
                    full_text_parts.append(slide_data["title"])
                if slide_data["content"]:
                    full_text_parts.append(slide_data["content"])
                if slide_data["notes"]:
                    full_text_parts.append(slide_data["notes"])
                
                slide_data["full_text"] = " ".join(full_text_parts).strip()
                
                # 记录提取的备注长度（用于调试）
                if slide_data["notes"]:
                    logger.debug(f"幻灯片 {i}: 提取备注长度 {len(slide_data['notes'])} 字符")
                else:
                    logger.debug(f"幻灯片 {i}: 无备注内容")
                
                slides.append(slide_data)
            
            # 统计备注提取情况
            slides_with_notes = sum(1 for slide in slides if slide["notes"])
            logger.info(f"PPT解析完成: 共 {len(slides)} 张幻灯片，其中 {slides_with_notes} 张包含备注")
            
            return slides
            
        except Exception as e:
            raise Exception(f"PPT解析失败: {str(e)}")
    
    def export_slides_to_images(
        self, 
        pptx_path: str, 
        output_dir: str, 
        resolution: str = "1920x1080"
    ) -> List[str]:
        """将PPTX幻灯片导出为图片
        
        Args:
            pptx_path: PPTX文件路径
            output_dir: 输出目录
            resolution: 输出分辨率，格式为"宽x高"
            
        Returns:
            图片路径列表，按幻灯片顺序排列
        """
        # 在try块之前获取LibreOffice路径，以便在异常处理中使用
        libreoffice_path = self._get_libreoffice_path()
        
        # 检查poppler是否安装
        if not self._check_poppler_installed():
            if platform.system() == "Windows":
                raise Exception(
                    "检测到Poppler未安装。pdf2image需要poppler-utils来解析PDF文件。\n\n"
                    "请执行以下步骤安装Poppler：\n"
                    "1. 下载Poppler for Windows：https://github.com/oschwartz10612/poppler-windows/releases/\n"
                    "2. 解压下载的ZIP文件到目录，如：C:\\Program Files\\poppler\n"
                    "3. 将bin目录（如：C:\\Program Files\\poppler\\Library\\bin）添加到系统PATH\n"
                    "4. 重启终端使PATH生效\n\n"
                    "或者使用以下方法：\n"
                    "- 使用conda安装：conda install -c conda-forge poppler\n"
                    "- 使用chocolatey安装：choco install poppler\n\n"
                    "安装完成后请重启应用。"
                )
            else:
                raise Exception(
                    "检测到Poppler未安装。pdf2image需要poppler-utils来解析PDF文件。\n\n"
                    "请执行以下步骤安装Poppler：\n"
                    "1. Ubuntu/Debian：sudo apt-get install poppler-utils\n"
                    "2. CentOS/RHEL/Fedora：sudo yum install poppler-utils\n"
                    "3. macOS：brew install poppler\n\n"
                    "安装后请确保poppler工具在PATH中可用，然后重启应用。"
                )
        
        try:
            # 确保输出目录存在
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 创建临时目录用于PDF转换
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_pdf = Path(temp_dir) / "presentation.pdf"
                
                # 使用LibreOffice将PPTX转换为PDF
                # soffice --headless --convert-to pdf --outdir <dir> <file>
                libreoffice_cmd = [
                    libreoffice_path,
                    "--headless",
                    "--convert-to", "pdf",
                    "--outdir", temp_dir,
                    pptx_path
                ]
                
                result = subprocess.run(
                    libreoffice_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5分钟超时
                )
                
                if result.returncode != 0:
                    raise Exception(f"PDF转换失败: {result.stderr}")
                
                if not temp_pdf.exists():
                    # 尝试查找生成的PDF文件
                    pdf_files = list(Path(temp_dir).glob("*.pdf"))
                    if not pdf_files:
                        raise Exception("PDF文件未生成")
                    temp_pdf = pdf_files[0]
                
                # 使用pdf2image将PDF转换为图片
                width, height = map(int, resolution.split('x'))
                
                images = convert_from_path(
                    str(temp_pdf),
                    dpi=300,  # 高DPI以获得清晰图片
                    size=(width, height),
                    output_folder=str(output_path),
                    fmt="png",
                    paths_only=True
                )
                
                # 确保图片按顺序命名
                image_paths = []
                for i, image_path in enumerate(sorted(images, key=lambda x: x)):
                    # 重命名为 slide_001.png, slide_002.png 等
                    new_name = output_path / f"slide_{i+1:03d}.png"
                    if image_path != new_name:
                        shutil.move(image_path, new_name)
                    image_paths.append(str(new_name))
                
                return image_paths
                
        except subprocess.TimeoutExpired:
            raise Exception(
                "PDF转换超时。可能原因：\n"
                "1. LibreOffice未正确安装或启动失败\n"
                "2. PPTX文件过大或损坏\n"
                "3. 系统资源不足\n"
                "请检查LibreOffice安装并尝试简化PPTX文件"
            )
        except FileNotFoundError:
            if platform.system() == "Windows":
                raise Exception(
                    f"未找到LibreOffice可执行文件。\n"
                    f"尝试的路径：{libreoffice_path}\n\n"
                    "请执行以下步骤：\n"
                    "1. 下载安装LibreOffice：https://www.libreoffice.org/download/download/\n"
                    "2. 安装时选择'添加到PATH'选项\n"
                    "3. 或者设置LIBREOFFICE_PATH环境变量指向soffice.exe完整路径\n"
                    f"当前配置路径：{settings.LIBREOFFICE_PATH}\n"
                    f"常见安装路径：C:\\Program Files\\LibreOffice\\program\\soffice.exe"
                )
            else:
                raise Exception(
                    f"未找到LibreOffice可执行文件。\n"
                    f"尝试的路径：{libreoffice_path}\n\n"
                    "请执行以下步骤：\n"
                    "1. Linux安装：sudo apt-get install libreoffice 或 sudo yum install libreoffice\n"
                    "2. macOS安装：brew install libreoffice\n"
                    "3. 或者设置LIBREOFFICE_PATH环境变量指向soffice完整路径\n"
                    f"当前配置路径：{settings.LIBREOFFICE_PATH}"
                )
        except PDFInfoNotInstalledError:
            if platform.system() == "Windows":
                raise Exception(
                    "未找到Poppler，pdf2image需要poppler-utils来解析PDF文件。\n\n"
                    "请执行以下步骤安装Poppler：\n"
                    "1. 下载Poppler for Windows：https://github.com/oschwartz10612/poppler-windows/releases/\n"
                    "2. 解压下载的ZIP文件到目录，如：C:\\Program Files\\poppler\n"
                    "3. 将bin目录（如：C:\\Program Files\\poppler\\Library\\bin）添加到系统PATH\n"
                    "4. 重启终端使PATH生效\n\n"
                    "或者使用以下方法：\n"
                    "- 使用conda安装：conda install -c conda-forge poppler\n"
                    "- 使用chocolatey安装：choco install poppler"
                )
            else:
                raise Exception(
                    "未找到Poppler，pdf2image需要poppler-utils来解析PDF文件。\n\n"
                    "请执行以下步骤安装Poppler：\n"
                    "1. Ubuntu/Debian：sudo apt-get install poppler-utils\n"
                    "2. CentOS/RHEL/Fedora：sudo yum install poppler-utils\n"
                    "3. macOS：brew install poppler\n\n"
                    "安装后请确保poppler工具在PATH中可用"
                )
        except Exception as e:
            raise Exception(f"PPT导出失败: {str(e)}")
    
    def get_slide_count(self, pptx_path: str) -> int:
        """获取幻灯片数量"""
        try:
            presentation = Presentation(pptx_path)
            return len(presentation.slides)
        except Exception as e:
            raise Exception(f"获取幻灯片数量失败: {str(e)}")
    
    def validate_pptx(self, pptx_path: str) -> bool:
        """验证PPTX文件是否有效"""
        try:
            presentation = Presentation(pptx_path)
            return len(presentation.slides) > 0
        except:
            return False