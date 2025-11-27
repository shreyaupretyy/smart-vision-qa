"""
Cache utilities using Redis.
"""
import json
from typing import Any, Optional
import redis
from backend.core.config import settings

# Redis client instance
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    decode_responses=True
)

def get_cache(key: str) -> Optional[Any]:
    """
    Get value from cache.
    
    Args:
        key: Cache key
        
    Returns:
        Cached value or None if not found
    """
    try:
        value = redis_client.get(key)
        if value:
            return json.loads(value)
        return None
    except Exception as e:
        print(f"Cache get error: {e}")
        return None

def set_cache(key: str, value: Any, ttl: int = 3600) -> bool:
    """
    Set value in cache with TTL.
    
    Args:
        key: Cache key
        value: Value to cache
        ttl: Time to live in seconds (default: 1 hour)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        redis_client.setex(key, ttl, json.dumps(value))
        return True
    except Exception as e:
        print(f"Cache set error: {e}")
        return False

def delete_cache(key: str) -> bool:
    """
    Delete value from cache.
    
    Args:
        key: Cache key
        
    Returns:
        True if successful, False otherwise
    """
    try:
        redis_client.delete(key)
        return True
    except Exception as e:
        print(f"Cache delete error: {e}")
        return False

def clear_cache_pattern(pattern: str) -> int:
    """
    Clear all cache keys matching pattern.
    
    Args:
        pattern: Redis key pattern (e.g., "video:*")
        
    Returns:
        Number of keys deleted
    """
    try:
        keys = redis_client.keys(pattern)
        if keys:
            return redis_client.delete(*keys)
        return 0
    except Exception as e:
        print(f"Cache clear error: {e}")
        return 0
