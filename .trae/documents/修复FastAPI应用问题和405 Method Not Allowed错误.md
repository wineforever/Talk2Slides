# 修复FastAPI应用问题和405 Method Not Allowed错误

## 问题分析

从用户提供的日志中，发现两个主要问题：

1. **405 Method Not Allowed错误**：前端POST请求到`/api/upload`返回405
2. **FastAPI弃用警告**：使用已弃用的`@app.on_event`装饰器

## 根本原因

### 405错误原因分析
通过代码审查发现：
1. **路由注册顺序问题**：`app.mount("/", StaticFiles(...))`在`app.include_router(api_router, prefix="/api")`之前注册
2. **静态文件服务干扰**：静态文件服务挂载在根路径`/`，可能捕获`/api/upload`请求，但只支持GET/HEAD方法
3. **CORS预检请求**：浏览器可能发送OPTIONS预检请求，需要确保正确处理

### 弃用警告原因
使用FastAPI已弃用的`@app.on_event("startup")`和`@app.on_event("shutdown")`，应改用lifespan事件处理器。

## 修复计划

### 第一阶段：修复路由和CORS问题（立即执行）
1. **调整路由注册顺序**：将API路由注册移到静态文件服务之前
2. **显式处理OPTIONS请求**：为`/api/upload`路由添加OPTIONS方法处理器
3. **验证CORS配置**：确保CORS中间件正确配置

### 第二阶段：修复FastAPI弃用警告
1. **替换事件处理器**：将`@app.on_event`替换为lifespan上下文管理器
2. **更新启动逻辑**：确保目录创建和初始化在lifespan中执行

### 第三阶段：前端兼容性修复
1. **检查API响应格式**：验证前端期望的响应字段与后端实际返回是否一致
2. **添加错误处理**：增强前端错误处理和用户反馈

## 具体代码修改

### 1. 修改`app/main.py`
```python
# 调整路由注册顺序：先API路由，后静态文件
app.include_router(api_router, prefix="/api")

# 然后挂载静态文件
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

# 替换弃用的on_event为lifespan
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    print(f"临时目录: {settings.TEMP_DIR}")
    print(f"输出目录: {settings.OUTPUT_DIR}")
    
    yield  # 应用运行期间
    
    # 关闭时清理（可选）
    pass

# 在创建FastAPI应用时指定lifespan
app = FastAPI(
    title="基于语义对齐的自动视频生成API",
    description="通过语义对齐将PPTX、MP3和SRT合成为自动切换页面的视频",
    version="1.0.0",
    lifespan=lifespan
)
```

### 2. 修改`app/api/endpoints.py`
```python
# 为/upload路由显式添加OPTIONS处理器
@router.api_route("/upload", methods=["POST", "OPTIONS"])
async def upload_files(...):
    # 现有代码不变
    # 对于OPTIONS请求，返回空响应和CORS头部
    if request.method == "OPTIONS":
        return Response(status_code=200)
```

### 3. 验证前端响应处理
检查前端`index.html`中第479行：
```javascript
// 当前代码
resultVideoUrl.value = resultResponse.data.video_url;

// 后端返回格式正确：{"video_url": "/api/task/{task_id}/result"}
// 需要确保resultResponse.data包含video_url字段
```

## 测试验证

修复后需要验证：
1. ✅ POST `/api/upload` 返回200而不是405
2. ✅ OPTIONS `/api/upload` 正确处理CORS预检
3. ✅ 应用启动无弃用警告
4. ✅ 文件上传和视频生成流程正常工作
5. ✅ 前端能正确获取和处理视频结果

## 风险控制

1. **向后兼容**：lifespan替换不会影响现有功能
2. **路由顺序**：调整顺序可能影响其他路由，但API路由应优先于静态文件
3. **CORS安全**：生产环境应将`allow_origins=["*"]`替换为具体域名

## 时间估计
- 代码修改：30分钟
- 测试验证：15分钟
- 总计：约45分钟

---

确认此计划后，我将开始实施修复。