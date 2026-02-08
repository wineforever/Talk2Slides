# Talk2Slides - 基于语义对齐的自动视频生成系统

🚀 **智能将PPTX、音频（MP3/WAV）和SRT文件转换为同步视频**

Talk2Slides是一个智能视频生成工具，能够将PPTX演示文稿、音频文件（支持MP3/WAV格式）和SRT字幕文件自动合成为一个视频。系统使用语义对齐技术，让视频中的幻灯片根据语音内容智能切换，实现完美的音画同步。

## ✨ 核心特性

- **🎯 语义智能对齐**：使用`sentence-transformers`模型计算PPT页面文本与字幕文本的语义相似度，实现内容驱动的幻灯片切换
- **⚡ 智能时间轴生成**：局部最优匹配算法支持PPT自由跳转（向前/向后），失败时自动启用备用策略（均匀分布），确保总能生成有效时间轴
- **🔧 智能错误处理**：严格的时间轴验证（end>start, duration>0），防止生成无效视频文件
- **🌐 现代化Web界面**：响应式设计，支持文件拖拽上传、实时进度显示、结果预览和下载
- **⚙️ 灵活参数配置**：可调整相似度阈值、最小展示时长、输出分辨率等参数
- **📁 智能文件管理**：输出文件包含时间戳和任务ID，便于管理和查找
- **🔊 用户体验优化**：处理完成音效提示，防止重复通知，清晰的错误信息
- **🎵 多格式音频支持**：支持MP3和WAV音频格式，满足不同场景需求


## 🏗️ 系统架构

### 技术栈
- **后端**: Python + FastAPI + SQLAlchemy
- **前端**: Vue 3 + Element Plus (CDN版本)
- **核心算法**: sentence-transformers + 局部最优匹配（支持PPT自由跳转）
- **视频处理**: FFmpeg + pdf2image

### 主要模块
1. **文件上传模块**: 接收并验证PPTX、音频（MP3/WAV）、SRT文件
2. **PPT解析模块**: 提取幻灯片标题、正文和备注框文本内容，导出为图片序列
3. **字幕解析模块**: 解析SRT文件时间戳和文本
4. **语义对齐模块**: 计算文本相似度，生成时间轴映射
5. **视频合成模块**: 根据时间轴将图片序列与音频合成为视频
6. **任务管理模块**: 管理异步任务状态和进度

## ⚙️ 安装部署

### 🛠️ 环境要求与验证

在开始安装前，请确保满足以下系统要求，并运行验证命令确认环境正确配置：

| 组件 | 要求 | 验证命令 | 预期输出 |
|------|------|----------|----------|
| **Python** | 3.9+ | `python --version` 或 `python3 --version` | Python 3.9.x 或更高 |
| **FFmpeg** | 必须安装并添加到PATH | `ffmpeg -version` | 显示FFmpeg版本信息 |
| **LibreOffice** | 完整版（非查看器） | `soffice --version` | 显示LibreOffice版本信息 |
| **系统内存** | 至少4GB RAM | - | 处理大文件时建议8GB+ |
| **磁盘空间** | 至少2GB可用空间 | - | 临时文件和处理需要空间 |

**⚠️ 重要提示**：
- **Windows用户**：确保FFmpeg的`bin`目录已添加到PATH环境变量
- **LibreOffice**：必须安装完整版，Windows上可能需要以服务模式运行
- **虚拟环境**：强烈建议使用Python虚拟环境，避免包冲突

### 1. 安装系统依赖

#### Windows
```powershell
# 安装FFmpeg (通过Chocolatey)
choco install ffmpeg

# 安装LibreOffice
choco install libreoffice
```

#### macOS
```bash
# 安装FFmpeg
brew install ffmpeg

# 安装LibreOffice
brew install --cask libreoffice
```

#### Linux (Ubuntu/Debian)
```bash
# 安装FFmpeg
sudo apt update
sudo apt install ffmpeg

# 安装LibreOffice
sudo apt install libreoffice
```

### 2. 安装Python依赖（使用虚拟环境）

**🎯 为什么使用虚拟环境？**
- 避免Python包版本冲突
- 保持项目依赖隔离
- 便于部署和环境复制

```bash
# 1. 进入后端目录
cd backend

# 2. 创建Python虚拟环境
python -m venv venv

# Install dependencies
pip install -r requirements.txt

# (Optional) Install PyTorch manually, Example (CPU only):
pip install torch torchvision torchaudio

# 3. 激活虚拟环境
# Windows (PowerShell)
venv\Scripts\activate
# Windows (CMD)
venv\Scripts\activate.bat
# Linux/macOS
source venv/bin/activate

# 4. 验证虚拟环境激活
# 命令提示符应该显示 (venv) 前缀
# 可以通过以下命令验证：
python -c "import sys; print('Python路径:', sys.executable)"

# 5. 安装项目依赖
pip install -r requirements.txt

# 6. 验证关键包安装
python -c "import fastapi; import sentence_transformers; print('✓ 核心包导入成功')"
```

**💡 提示**：每次打开新的终端窗口时，都需要重新激活虚拟环境。

### 3. 配置环境变量（可选）

大多数情况下，系统无需额外配置即可运行。只有在以下情况下需要创建 `.env` 文件：

1. **FFmpeg/LibreOffice不在PATH中**：需要指定可执行文件完整路径
2. **自定义模型**：想使用其他`sentence-transformers`模型
3. **自定义端口/主机**：需要修改默认的8000端口或0.0.0.0主机

**创建配置文件的步骤**：
```bash
# 在backend目录下创建.env文件
cd backend
nano .env  # 或使用其他文本编辑器
```

**`.env` 文件示例**：
```env
# 应用基础配置
DEBUG=true                    # 调试模式，生产环境设置为false
HOST=0.0.0.0                  # 监听所有网络接口
PORT=8000                     # 服务端口

# 模型配置
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2  # 语义相似度计算模型

# 外部工具路径（仅在不在PATH中时需要）
# Windows示例：
# FFMPEG_PATH=C:\Program Files\ffmpeg\bin\ffmpeg.exe
# LIBREOFFICE_PATH=C:\Program Files\LibreOffice\program\soffice.exe
# Linux/macOS示例：
# FFMPEG_PATH=/usr/bin/ffmpeg
# LIBREOFFICE_PATH=/usr/bin/libreoffice

# 高级配置（一般不需要修改）
# DEFAULT_SIMILARITY_THRESHOLD=0.5
# DEFAULT_MIN_DISPLAY_DURATION=2.0
# DEFAULT_OUTPUT_RESOLUTION=1920x1080
```

**🔍 路径查找方法**：
```bash
# 查找FFmpeg路径
which ffmpeg        # Linux/macOS
where ffmpeg        # Windows

# 查找LibreOffice路径
which soffice       # Linux/macOS  
where soffice       # Windows
```

### 4. 启动应用

#### 方法一：直接运行（开发环境）
```bash
# 1. 确保在backend目录且虚拟环境已激活
cd backend
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# 2. 启动FastAPI应用
python -m app.main
# 或使用uvicorn直接启动
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 方法二：使用uvicorn热重载（推荐开发使用）
```bash
cd backend
venv\Scripts\activate  # 激活虚拟环境
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**启动参数说明**：
- `--reload`：代码变更时自动重启（开发模式）
- `--host 0.0.0.0`：监听所有网络接口
- `--port 8000`：服务端口，可修改为其他端口

#### 方法三：生产环境启动
```bash
# 使用gunicorn作为WSGI服务器（Linux/macOS）
cd backend
source venv/bin/activate
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8000
```

**启动成功标志**：
```
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

✅ **访问Web界面**：打开浏览器访问 `http://localhost:8000`

🔄 **前端文件服务**：FastAPI会自动提供`frontend/index.html`文件，无需额外配置Web服务器。

## 🔌 API接口

### 文件上传
```
POST /api/upload
Content-Type: multipart/form-data

参数:
- pptx: PPTX文件
- mp3: 音频文件（支持MP3/WAV格式）
- srt: SRT字幕文件
- similarity_threshold: 相似度阈值 (0-1, 默认0.5)
- min_display_duration: 最小展示时长 (秒, 默认2.0)
- output_resolution: 输出分辨率 (默认1920x1080)
```

### 任务状态查询
```
GET /api/task/{task_id}/status
```

### 获取结果视频
```
GET /api/task/{task_id}/result
```

## 📖 使用示例

### 1. 访问Web界面
打开浏览器访问 `http://localhost:8000`

### 2. 上传文件
- 上传PPTX演示文稿文件
- 上传音频文件（支持MP3/WAV格式）
- 上传SRT字幕文件

### 3. 配置参数与调优指南

#### 🤖 算法改进说明
**改进内容**：用局部最优匹配算法替换原有的动态规划算法，支持PPT自由跳转

**主要优势**：
1. **🚀 PPT自由跳转**：支持向前和向后跳转，当音频回到之前的话题时，PPT也能跳回对应页面
2. **🎯 更好的内容匹配**：每个字幕片段独立选择最佳幻灯片，不受历史选择约束
3. **📊 详细统计信息**：提供相似度分布、PPT跳转模式、匹配成功率等详细分析
4. **🔍 智能调试建议**：根据相似度分布推荐合适的阈值范围

**算法对比**：
| 特性 | 旧算法（动态规划） | 新算法（局部最优匹配） |
|------|-------------------|----------------------|
| PPT跳转 | 只能向前，不能向后 | 支持向前和向后跳转 |
| 匹配策略 | 全局最优，受历史约束 | 局部最优，独立选择 |
| 调试信息 | 简单的成功/失败 | 详细的统计和分布信息 |
| 适用场景 | 顺序讲解的演讲 | 自由讨论、话题回退的演讲 |

**使用建议**：
- 对于**自由讨论**或**话题回退**的演讲，新算法能更好地匹配内容
- 系统会自动记录PPT跳转统计，帮助了解内容组织模式
- 如果相似度过低，系统会提供具体的阈值调整建议

#### 🎯 相似度阈值 (0.0-1.0)
**默认值**: 0.5  
**作用**: 控制字幕与幻灯片文本的匹配严格程度

**调优建议**：
- **内容高度相关**（如学术演讲、产品发布会）：0.4-0.6
- **内容一般相关**（如教学视频、培训材料）：0.3-0.5
- **内容弱相关**（如自由演讲、即兴分享）：0.2-0.4

**调试经验**：
- 默认值0.5可能过高，导致匹配失败
- 建议从0.3开始尝试，逐步提高
- 如果局部最优匹配算法失败，系统会自动切换到备用策略（均匀分布）

#### ⏱️ 最小展示时长 (秒)
**默认值**: 2.0秒  
**作用**: 确保每个幻灯片至少显示指定时间，避免页面闪烁

**调优建议**：
- **快速演示**：1.0-1.5秒
- **标准演示**：1.5-2.5秒  
- **详细讲解**：2.0-3.0秒

**注意事项**：
- 时长过短可能导致页面切换过快
- 时长过长可能导致与音频不同步
- 系统会强制每个幻灯片至少展示此时长

#### 🖥️ 输出分辨率
**默认值**: 1920x1080 (Full HD)  
**可用选项**：
- **1920x1080**：全高清，适合大多数场景
- **1280x720**：高清，文件较小，处理更快
- **1024x576**：标清，快速测试使用

**选择建议**：
- 原始PPT比例为16:9时选择1920x1080
- 需要快速处理时选择1280x720
- 测试和预览时选择1024x576

#### 💡 高级调优技巧
1. **首次运行**：使用默认参数，观察结果
2. **调整顺序**：先调整相似度阈值，再调整最小展示时长
3. **内容分析**：如果PPT和字幕内容差异大，降低相似度阈值
4. **备用策略**：当局部最优匹配算法失败时，系统会自动使用均匀分布策略
5. **日志分析**：查看控制台日志了解每个阶段的处理情况

### 4. 开始处理
点击"开始生成视频"按钮，系统将：
1. 解析PPT内容和字幕
2. 进行语义对齐计算
3. 导出幻灯片为图片
4. 合成最终视频

### 5. 查看结果
处理完成后可以：
- 在线预览生成的视频
- 下载MP4文件（文件名格式：`video_{任务ID}_YYYYMMDD_HHMMSS.mp4`）

## 🔧 调试经验与常见问题

基于实际开发调试经验，以下是您可能会遇到的问题及解决方案：

### 🎯 问题1：视频生成成功但无输出文件
**现象**：系统提示"视频生成成功"，但在输出目录找不到文件或文件大小为0字节。

**根本原因**：
1. **无效时间轴数据**：时间片段包含 `end <= start` 或 `duration <= 0` 的无效数据
2. **FFmpeg处理失败**：FFmpeg命令执行失败但错误信息被错误处理逻辑吞掉
3. **文件验证不充分**：只检查文件是否存在，不检查文件大小有效性
4. **输出目录权限问题**：应用没有写入权限

**解决方案**：
- ✅ **系统已修复**：添加严格的时间轴验证（`end > start` 且 `duration > 0`）
- ✅ **增强错误处理**：添加详细的阶段标记（ppt_parsing → srt_parsing → alignment → ppt_to_images → video_synthesis）
- ✅ **文件大小验证**：视频文件至少1KB，防止0字节文件
- ✅ **目录权限检查**：启动时验证输出目录可写权限

**验证方法**：
```bash
# 检查输出目录
ls -la Talk2Slides/output/

# 查看应用日志中的阶段信息
# 日志会显示类似："视频生成失败: 阶段=video_synthesis, 错误: ..."
```

### 📊 问题2：语义对齐失败，无法生成时间轴
**现象**：局部最优匹配算法失败，返回空时间轴或处理卡在"语义对齐"阶段。

**根本原因**：
1. **内容完全不相关**：PPT文本和字幕内容语义相似度过低（< 0.3）
2. **参数设置不当**：相似度阈值过高（默认0.5）或最小展示时长过长
3. **相似度过低**：大多数字幕片段没有达到相似度阈值
4. **备用策略失效**：备用策略生成的时间轴包含无效片段

**解决方案**：
- ✅ **双策略保障**：主算法（局部最优匹配）失败时自动切换到备用策略（均匀分布）
- ✅ **算法改进**：支持PPT自由跳转（向前/向后），不再受单调不减约束限制
- ✅ **参数调优建议**：
  - 相似度阈值：尝试 **0.2-0.4**（默认0.5可能过高，从0.3开始尝试）
  - 最小展示时长：尝试 **1.0-2.0秒**（默认2.0秒）
- ✅ **详细调试信息**：系统会记录相似度分布、匹配统计、PPT跳转模式
- ✅ **时间轴验证**：备用策略生成的时间轴会经过有效性过滤

**调试技巧**：
```python
# 查看对齐服务的详细调试信息
# 日志会显示：
# 1. 相似度统计：平均相似度、相似度分布（<0.1, 0.1-0.3, 0.3-0.5等）
# 2. 匹配结果：X/Y个片段达到阈值，X个片段未匹配
# 3. PPT跳转统计：向前跳转X次，向后跳转Y次，保持同一页Z次
# 4. 算法建议：根据相似度分布推荐合适的阈值范围
```

**新功能亮点**：
- 🚀 **PPT自由跳转**：支持向前和向后跳转，当音频回到之前的话题时，PPT也能跳回对应页面
- 📊 **详细统计信息**：提供相似度分布、PPT跳转模式等详细分析
- 🔍 **智能建议**：根据相似度分布推荐合适的阈值设置
- ⚡ **改进的算法**：局部最优匹配取代动态规划，更好的内容匹配效果

### 🎯 问题3：PPT跳转乱套，不符合实际演讲逻辑
**现象**：PPT频繁乱跳，一会跳转到第一张幻灯片，一会又跳到另一张，不符合实际演讲顺序。

**根本原因**：
1. **算法约束不足**：旧的局部最优匹配算法太自由，缺乏约束
2. **缺少页面限制**：中间内容可能匹配到第一页或最后一页
3. **跳转控制缺失**：没有限制跳转距离，导致大幅跳跃

**解决方案**：
- ✅ **改进算法**：使用约束局部最优匹配算法，添加智能约束
- ✅ **时间区域划分**：将演讲分为开始区域（前10%）、核心区域（中间80%）、结束区域（后10%）
- ✅ **页面限制**：核心区域禁止匹配第一页和最后一页（除非开始或结束）
- ✅ **跳转距离限制**：相邻片段最大跳转5页，防止大幅跳跃
- ✅ **详细调试信息**：提供相似度分布、PPT跳转模式、约束应用统计

**新算法特点**：
1. **🚀 智能约束**：时间区域划分和页面限制确保PPT跳转符合演讲逻辑
2. **📊 稳定映射**：跳转距离限制防止PPT乱跳，提供稳定的时间轴映射
3. **🎯 话题回退**：支持向后跳转（话题回退），但受跳转距离限制
4. **🔍 详细分析**：提供匹配序列、异常跳转检测等调试信息

**使用建议**：
- 查看日志中的"匹配序列总结"了解PPT跳转模式
- 检查"异常向后跳转"警告识别可能的匹配问题
- 根据相似度分布调整阈值参数

### 🗂️ 问题4：文件路径和权限问题
**现象**：文件处理失败，出现权限错误、路径不存在或编码问题。

**根本原因**：
1. **输出目录位置**：原输出目录在 `backend/output/`，不易查找
2. **Windows路径编码**：包含中文或特殊字符的路径导致问题
3. **临时目录空间不足**：处理大文件时磁盘空间不足
4. **LibreOffice服务模式**：Windows上LibreOffice需要以服务模式运行

**解决方案**：
- ✅ **优化输出路径**：输出目录改为项目根目录 `Talk2Slides/output/`
- ✅ **路径编码处理**：所有路径使用UTF-8编码和字符串表示
- ✅ **详细日志记录**：记录完整的绝对路径便于调试
- ✅ **空间检查**：处理前检查临时目录可用空间

**验证方法**：
```bash
# 检查目录结构
tree Talk2Slides/ -L 2

# 检查磁盘空间（Windows）
wmic logicaldisk get size,freespace,caption

# 检查路径权限
icacls Talk2Slides/output/
```

### ⚙️ 问题5：依赖安装和环境配置
**现象**：安装依赖失败，运行时缺少必要组件。

**实际踩坑经验**：
1. **FFmpeg安装**：Windows用户需要将ffmpeg.exe所在目录添加到PATH环境变量
2. **LibreOffice安装**：必须安装完整版，不仅仅是查看器；Windows上可能需要配置服务模式
3. **Python包冲突**：不使用虚拟环境时，包版本冲突常见
4. **模型下载失败**：网络问题导致`sentence-transformers`模型下载超时

**解决方案**：
- ✅ **虚拟环境**：始终使用Python虚拟环境
- ✅ **依赖验证脚本**：提供环境验证脚本
- ✅ **离线模型支持**：可手动下载模型到本地
- ✅ **详细错误信息**：提供针对性的解决建议

**环境验证**：
```bash
# 运行环境验证
cd backend
python -c "import sys; print('Python:', sys.version)"
ffmpeg -version
soffice --version
```

### 🚀 问题6：性能优化和大文件处理
**现象**：处理大文件时内存不足、速度慢或超时。

**经验总结**：
1. **PPT页数限制**：建议不超过100页，每页文本不超过500字
2. **音频时长限制**：建议不超过2小时
3. **内存管理**：图片导出时及时释放内存
4. **进度反馈**：长任务需要定期更新进度状态

**优化建议**：
- ✅ **异步处理**：支持后台任务处理和进度查询
- ✅ **内存优化**：及时清理临时文件
- ✅ **进度监控**：实时显示处理阶段和进度百分比
- ✅ **超时处理**：设置合理的超时时间

## 🛠️ 开发指南

### 📁 项目结构
```
Talk2Slides/
├── backend/                  # 后端Python代码
│   ├── app/                 # 应用核心代码
│   │   ├── api/             # FastAPI端点定义
│   │   ├── core/            # 配置和常量
│   │   └── services/        # 业务逻辑服务
│   │       ├── alignment_service.py    # 语义对齐服务
│   │       ├── ppt_service.py          # PPT解析服务
│   │       ├── srt_service.py          # 字幕解析服务
│   │       ├── video_service.py        # 视频合成服务
│   │       └── task_manager.py         # 任务管理服务
│   ├── requirements.txt     # Python依赖包列表
│   └── venv/                # Python虚拟环境（创建后）
├── frontend/                # 前端Web界面
│   └── index.html          # 单页面应用（Vue 3 + Element Plus）
├── output/                  # ✅ 视频输出目录（新位置）
│   └── {task_id}/          # 按任务ID组织的输出文件
│       └── video_{task_id}_YYYYMMDD_HHMMSS.mp4  # 生成的视频文件
├── temp/                    # 临时处理文件（运行时自动创建）
│   ├── uploads/            # 上传文件临时存储
│   └── {task_id}/          # 任务临时工作目录
├── .trae/                  # Trae IDE配置文件（开发环境）
├── README.md               # 本文档
└── slidesync_minimal.py    # 早期原型脚本
```

### 添加新功能
1. 在 `app/services/` 中添加新的服务类
2. 在 `app/api/endpoints.py` 中添加API端点
3. 在 `frontend/index.html` 中更新前端界面

### 测试
```bash
# 运行导入测试
python test_imports.py

# 启动开发服务器
python -m app.main
```

## ⚡ 性能优化与最佳实践

基于实际调试和部署经验，以下优化建议可以帮助您获得更好的性能和用户体验：

### 🚀 处理性能优化
1. **文件大小限制**：
   - PPT文件：建议不超过100页，每页文本不超过500字
   - 音频文件：建议不超过2小时，MP3或WAV格式最佳
   - 输出视频：根据需求选择分辨率，测试时使用较低分辨率

2. **内存管理优化**：
   - 及时清理临时文件：任务完成后自动清理临时目录
   - 分阶段处理：PPT解析、语义对齐、视频合成分阶段进行，避免内存峰值
   - 图片缓存：同一PPT文件多次处理时可复用已导出的图片

3. **处理速度优化**：
   - 使用SSD硬盘：显著加快文件读写速度
   - 调整分辨率：测试时使用1024x576，生产时使用1280x720或1920x1080
   - 关闭调试模式：生产环境设置`DEBUG=false`减少日志输出

### 🎯 质量与效果优化
1. **语义对齐质量**：
   - 确保PPT和字幕内容相关：内容相关度越高，对齐效果越好
   - 合理设置参数：根据内容类型调整相似度阈值
   - 使用备用策略：当DP算法失败时，均匀分布策略仍能生成可用的时间轴

2. **视频输出质量**：
   - 保持PPT原始比例：16:9的PPT在1920x1080分辨率下效果最佳
   - 检查时间轴有效性：系统已添加严格验证，确保时间片段有效
   - 验证输出文件：系统会检查视频文件大小和有效性

### 🔧 系统部署优化
1. **环境配置**：
   - 使用虚拟环境：避免Python包冲突
   - 确保磁盘空间：临时目录需要足够空间（建议至少2GB）
   - 网络连接：确保可以下载`sentence-transformers`模型

2. **监控与日志**：
   - 查看处理日志：了解每个阶段的耗时和状态
   - 监控资源使用：处理大文件时关注内存和CPU使用率
   - 错误处理：系统提供详细的错误信息和解决建议

3. **扩展性考虑**：
   - 异步处理：支持长时间任务，不阻塞Web请求
   - 任务管理：支持任务状态查询和结果获取
   - 文件管理：输出文件按任务ID组织，便于管理和查找

### 💡 实用技巧
- **首次运行**：使用小文件测试，验证环境配置
- **参数调优**：从默认参数开始，根据结果逐步调整
- **日志分析**：关注处理阶段的耗时，识别性能瓶颈
- **批量处理**：可通过脚本批量处理多个任务

## 📜 许可证

本项目采用MIT许可证。

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启Pull Request

## 📞 联系方式

如有问题或建议，请通过Issue反馈。