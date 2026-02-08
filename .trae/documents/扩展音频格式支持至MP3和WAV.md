# 扩展音频格式支持至MP3和WAV

## 当前状态分析
当前系统仅支持MP3音频格式，通过以下方式限制：
1. **前端限制**：`accept=".mp3"`属性限制用户只能选择MP3文件
2. **后端验证**：严格检查文件扩展名是否为`.mp3`
3. **固定文件名**：保存文件时使用硬编码的`audio.mp3`文件名

## 技术可行性
FFmpeg支持多种音频格式，包括MP3和WAV，因此后端处理逻辑无需修改。主要需要修改验证和文件名处理逻辑。

## 修改计划

### 1. 前端修改 (frontend/index.html)
**目标**：允许用户选择MP3和WAV文件

**具体修改**：
- 第270行：修改`accept=".mp3"`为`accept=".mp3,.wav,.wave"`
- 更新上传标签文字：从"将MP3文件拖到此处"改为"将音频文件拖到此处（支持MP3/WAV）"
- 更新文件类型提示信息

### 2. 后端验证修改 (backend/app/api/endpoints.py)
**目标**：支持MP3和WAV文件验证，动态处理文件名

**具体修改**：
1. **第43行**：修改文件类型验证逻辑
   ```python
   # 修改前：
   if not mp3.filename.endswith('.mp3'):
       raise HTTPException(status_code=400, detail="音频文件必须是.mp3格式")
   
   # 修改后：
   allowed_audio_extensions = ['.mp3', '.wav', '.wave']
   if not any(mp3.filename.lower().endswith(ext) for ext in allowed_audio_extensions):
       raise HTTPException(status_code=400, detail="音频文件必须是.mp3或.wav格式")
   ```

2. **第57行**：动态生成音频文件名
   ```python
   # 修改前：
   mp3_path = task_dir / "audio.mp3"
   
   # 修改后：
   audio_ext = Path(mp3.filename).suffix.lower()
   if audio_ext not in ['.mp3', '.wav', '.wave']:
       audio_ext = '.mp3'  # 默认后缀
   mp3_path = task_dir / f"audio{audio_ext}"
   ```

3. **第90行**：更新`process_video_generation`函数调用，确保音频路径正确传递

4. **更新错误消息**：将错误消息从"音频文件必须是.mp3格式"更新为"音频文件必须是.mp3或.wav格式"

### 3. 音频服务兼容性检查 (backend/app/services/video_service.py)
**目标**：确保音频服务能正确处理WAV格式

**具体检查**：
1. `extract_audio_info`方法：FFmpeg probe应该能处理WAV格式
2. `_merge_audio_video`方法：FFmpeg应该能正确合并WAV音频
3. 第442行错误建议：更新建议信息，包含WAV格式

**可能的修改**：
```python
# 第442行：更新错误建议信息
if "Invalid data found when processing input" in error_message:
    error_analysis.append("1. 音频文件可能损坏或格式不支持")
    error_analysis.append("2. 视频文件可能损坏") 
    error_analysis.append("3. 尝试使用其他音频格式（如MP3、WAV）")
```

### 4. 文件名处理优化
**目标**：确保整个处理流程使用正确的音频文件扩展名

**需要考虑的点**：
1. 任务管理器中的`mp3_path`字段名可能需要重命名为`audio_path`（可选，为了保持一致性）
2. 确保所有引用音频路径的地方都能处理不同的扩展名
3. 日志记录中更新"MP3"为"音频"以反映格式通用性

### 5. 文档更新 (README.md)
**目标**：更新文档反映新的音频格式支持

**具体更新**：
1. **特性描述**：更新核心特性，说明支持MP3和WAV格式
2. **使用示例**：更新文件格式说明
3. **故障排除**：更新相关错误信息
4. **参数说明**：澄清音频格式要求

### 6. 测试计划
**手动测试用例**：
1. 上传MP3文件（确保现有功能正常）
2. 上传WAV文件（测试新功能）
3. 尝试上传不支持的文件格式（如.mp4音频，确保正确拒绝）
4. 测试带有不同大小写扩展名的文件（如.WAV、.Wave）

## 实施步骤

### 阶段1：前端修改 (5分钟)
1. 修改`accept`属性
2. 更新UI文字提示
3. 测试前端文件选择功能

### 阶段2：后端验证修改 (10分钟)
1. 修改文件类型验证逻辑
2. 实现动态文件名生成
3. 更新错误消息

### 阶段3：音频服务检查 (5分钟)
1. 检查video_service.py的兼容性
2. 更新错误建议信息

### 阶段4：全面测试 (10分钟)
1. 测试MP3文件上传和处理
2. 测试WAV文件上传和处理
3. 测试错误情况处理
4. 验证输出视频质量

### 阶段5：文档更新 (5分钟)
1. 更新README.md
2. 验证文档准确性

## 预期结果
1. 用户可以在前端选择MP3或WAV格式的音频文件
2. 后端正确验证和处理两种格式
3. 视频生成流程保持不变，支持两种音频格式
4. 错误信息清晰指导用户使用正确的格式
5. 文档准确反映支持的音频格式

## 风险评估与缓解

### 风险1：WAV文件处理性能
- **风险**：WAV文件通常比MP3文件大，可能导致处理时间增加或内存使用增加
- **缓解**：系统使用FFmpeg进行音频处理，FFmpeg能高效处理各种音频格式。添加文件大小检查，提示用户大文件可能处理较慢

### 风险2：WAV格式变体兼容性
- **风险**：WAV文件有多种编码格式，某些格式可能不被FFmpeg支持
- **缓解**：依赖FFmpeg的广泛格式支持。在错误信息中提供明确指导

### 风险3：前端浏览器兼容性
- **风险**：不同浏览器对文件选择器的`accept`属性支持可能不同
- **缓解**：保持后端验证作为主要防线，前端限制只是用户体验优化

## 扩展性考虑
未来如果需要支持更多音频格式（如.ogg、.m4a、.flac），只需：
1. 在前端`accept`属性中添加新扩展名
2. 在后端允许的扩展名列表中添加新格式
3. 确保FFmpeg支持该格式