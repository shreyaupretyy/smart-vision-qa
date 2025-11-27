"""
Background task utilities for async processing.
"""
import asyncio
from typing import Callable, Any, Dict
from datetime import datetime
from backend.core.cache import set_cache, get_cache

class TaskStatus:
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class BackgroundTaskManager:
    """
    Manager for background tasks with status tracking.
    """
    
    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}
    
    def create_task(
        self,
        task_id: str,
        coro: Callable,
        *args,
        **kwargs
    ) -> str:
        """
        Create and start a background task.
        
        Args:
            task_id: Unique task identifier
            coro: Coroutine to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Task ID
        """
        # Set initial status
        self._set_task_status(task_id, TaskStatus.PENDING)
        
        # Create task
        task = asyncio.create_task(
            self._run_task(task_id, coro, *args, **kwargs)
        )
        self.tasks[task_id] = task
        
        return task_id
    
    async def _run_task(
        self,
        task_id: str,
        coro: Callable,
        *args,
        **kwargs
    ):
        """
        Run task and track status.
        
        Args:
            task_id: Task identifier
            coro: Coroutine to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        try:
            # Update status to running
            self._set_task_status(task_id, TaskStatus.RUNNING)
            
            # Execute task
            result = await coro(*args, **kwargs)
            
            # Update status to completed
            self._set_task_status(
                task_id,
                TaskStatus.COMPLETED,
                result=result
            )
            
        except Exception as e:
            # Update status to failed
            self._set_task_status(
                task_id,
                TaskStatus.FAILED,
                error=str(e)
            )
        finally:
            # Cleanup
            if task_id in self.tasks:
                del self.tasks[task_id]
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get task status.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status information
        """
        status = get_cache(f"task:{task_id}")
        if not status:
            return {
                "status": "not_found",
                "error": "Task not found"
            }
        return status
    
    def _set_task_status(
        self,
        task_id: str,
        status: str,
        **kwargs
    ):
        """
        Set task status in cache.
        
        Args:
            task_id: Task identifier
            status: Task status
            **kwargs: Additional status data
        """
        status_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat(),
            **kwargs
        }
        set_cache(f"task:{task_id}", status_data, ttl=3600)

# Global task manager instance
task_manager = BackgroundTaskManager()
