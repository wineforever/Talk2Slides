#!/usr/bin/env python3
"""应用启动脚本"""

import os
import sys
import uvicorn

if __name__ == "__main__":
    # 设置Python路径
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # 导入应用
    from app.main import app
    
    # 启动服务器
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )