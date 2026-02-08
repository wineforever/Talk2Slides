#!/usr/bin/env python3
"""测试所有模块的导入"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试导入所有模块"""
    modules_to_test = [
        "app.main",
        "app.core.config",
        "app.services.task_manager",
        "app.services.ppt_service",
        "app.services.srt_service",
        "app.services.alignment_service",
        "app.services.video_service",
        "app.api.endpoints"
    ]
    
    print("开始测试导入...")
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name} 导入成功")
        except Exception as e:
            print(f"✗ {module_name} 导入失败: {e}")
            return False
    
    print("\n所有模块导入成功！")
    return True

if __name__ == "__main__":
    if test_imports():
        sys.exit(0)
    else:
        sys.exit(1)