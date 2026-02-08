#!/usr/bin/env python3
"""测试API修复"""

import sys
import os
import requests
import json

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_api_endpoints():
    """测试API端点"""
    base_url = "http://localhost:8000"
    
    print("测试API端点...")
    
    # 测试健康检查
    try:
        response = requests.get(f"{base_url}/health")
        print(f"GET /health: 状态码={response.status_code}, 响应={response.json()}")
        assert response.status_code == 200
        print("✓ 健康检查通过")
    except Exception as e:
        print(f"✗ 健康检查失败: {e}")
        return False
    
    # 测试OPTIONS预检请求
    try:
        response = requests.options(f"{base_url}/api/upload")
        print(f"OPTIONS /api/upload: 状态码={response.status_code}")
        # OPTIONS可能返回200或204
        assert response.status_code in [200, 204]
        print("✓ OPTIONS预检请求通过")
    except Exception as e:
        print(f"✗ OPTIONS预检请求失败: {e}")
        return False
    
    # 测试POST请求（不带文件，应该返回错误但不是405）
    try:
        response = requests.post(f"{base_url}/api/upload", data={})
        print(f"POST /api/upload (无文件): 状态码={response.status_code}")
        # 应该返回400（缺少文件）而不是405
        assert response.status_code != 405, "仍然返回405 Method Not Allowed"
        print(f"✓ POST请求不再返回405 (得到{response.status_code})")
    except AssertionError as e:
        print(f"✗ {e}")
        return False
    except Exception as e:
        print(f"POST请求错误 (不是405): {e}")
        # 这可能是预期的（如400错误）
        pass
    
    # 测试API文档
    try:
        response = requests.get(f"{base_url}/docs")
        print(f"GET /docs: 状态码={response.status_code}")
        assert response.status_code == 200
        print("✓ API文档可访问")
    except Exception as e:
        print(f"✗ API文档测试失败: {e}")
        return False
    
    print("\n所有API测试通过！")
    return True

if __name__ == "__main__":
    # 检查服务器是否运行
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        print(f"服务器运行中，健康检查: {response.status_code}")
    except:
        print("警告: 服务器可能未运行在 http://localhost:8000")
        print("请先运行: python -m app.main")
        sys.exit(1)
    
    success = test_api_endpoints()
    sys.exit(0 if success else 1)