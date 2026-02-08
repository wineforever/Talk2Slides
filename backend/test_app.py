#!/usr/bin/env python3
"""测试FastAPI应用"""

import sys
import os
import asyncio

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_app():
    """测试应用启动和基本路由"""
    try:
        from app.main import app
        from fastapi.testclient import TestClient
        
        print("正在初始化测试客户端...")
        client = TestClient(app)
        
        # 测试健康检查端点
        print("测试健康检查端点...")
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✓ 健康检查端点正常")
        
        # 测试根路径
        print("测试根路径...")
        response = client.get("/")
        assert response.status_code in [200, 404]  # 200如果有前端文件，404如果没有
        print(f"✓ 根路径返回状态码: {response.status_code}")
        
        # 测试API文档
        print("测试API文档...")
        response = client.get("/docs")
        assert response.status_code == 200
        print("✓ API文档正常")
        
        # 测试任务管理API
        print("测试任务管理API...")
        response = client.get("/api/task/nonexistent/status")
        assert response.status_code == 404
        print("✓ 任务状态端点正常")
        
        print("\n所有测试通过！")
        return True
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_app())
    sys.exit(0 if success else 1)