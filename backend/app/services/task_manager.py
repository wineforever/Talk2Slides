import uuid
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import threading

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    """任务数据类"""
    task_id: str
    state: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0  # 0-100
    message: str = ""
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    result: Optional[Dict[str, Any]] = None
    
    # 任务参数
    pptx_path: Optional[str] = None
    mp3_path: Optional[str] = None
    srt_path: Optional[str] = None
    similarity_threshold: float = 0.5
    min_display_duration: float = 2.0
    output_resolution: str = "1920x1080"

class TaskManager:
    """任务管理器"""
    
    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.RLock()
        self._cleanup_interval = 3600  # 每小时清理一次
        self._last_cleanup = time.time()
    
    def create_task(
        self,
        task_id: str,
        pptx_path: str,
        mp3_path: str,
        srt_path: str,
        similarity_threshold: float = 0.5,
        min_display_duration: float = 2.0,
        output_resolution: str = "1920x1080"
    ) -> Task:
        """创建新任务"""
        with self._lock:
            task = Task(
                task_id=task_id,
                pptx_path=pptx_path,
                mp3_path=mp3_path,
                srt_path=srt_path,
                similarity_threshold=similarity_threshold,
                min_display_duration=min_display_duration,
                output_resolution=output_resolution
            )
            self._tasks[task_id] = task
            self._cleanup_old_tasks()
            return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        with self._lock:
            return self._tasks.get(task_id)
    
    def update_task(
        self,
        task_id: str,
        state: Optional[TaskStatus] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None
    ) -> Optional[Task]:
        """更新任务状态"""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            
            if state is not None:
                task.state = state
            if progress is not None:
                task.progress = max(0.0, min(100.0, progress))
            if message is not None:
                task.message = message
            if error is not None:
                task.error = error
            if result is not None:
                task.result = result
            
            task.updated_at = datetime.now()
            return task
    
    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                return True
            return False
    
    def list_tasks(self, state: Optional[TaskStatus] = None) -> Dict[str, Task]:
        """列出任务"""
        with self._lock:
            if state is None:
                return self._tasks.copy()
            return {
                task_id: task 
                for task_id, task in self._tasks.items() 
                if task.state == state
            }
    
    def _cleanup_old_tasks(self):
        """清理旧任务"""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(hours=24)  # 清理24小时前的任务
            
            tasks_to_delete = []
            for task_id, task in self._tasks.items():
                if task.updated_at < cutoff:
                    tasks_to_delete.append(task_id)
            
            for task_id in tasks_to_delete:
                del self._tasks[task_id]
            
            self._last_cleanup = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """获取任务统计信息"""
        with self._lock:
            total = len(self._tasks)
            stats = {
                "total_tasks": total,
                "by_state": {},
                "oldest_task": None,
                "newest_task": None
            }
            
            if total > 0:
                for state in TaskStatus:
                    stats["by_state"][state.value] = sum(
                        1 for task in self._tasks.values() 
                        if task.state == state
                    )
                
                oldest = min(self._tasks.values(), key=lambda t: t.created_at)
                newest = max(self._tasks.values(), key=lambda t: t.created_at)
                stats["oldest_task"] = oldest.created_at.isoformat()
                stats["newest_task"] = newest.created_at.isoformat()
            
            return stats