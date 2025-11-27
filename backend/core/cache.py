"""
Cache utilities using Redis.
"""
import json
from typing import Any, Optional
try:
    import redis
    from backend.core.config import get_settings
    settings = get_settings()
    
    # Redis client instance
    try:
        redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=0,
            decode_responses=True,
            socket_connect_timeout=2
        )
        redis_client.ping()
        REDIS_AVAILABLE = True
    except:
        redis_client = None
        REDIS_AVAILABLE = False
        print("Warning: Redis not available. Caching disabled.")
except ImportError:
    redis_client = None
    REDIS_AVAILABLE = False
    print("Warning: Redis module not installed. Caching disabled.")

def get_cache(key: str) -> Optional[Any]:
    """
    Get value from cache.
    
    Args:
        key: Cache key
        
    Returns:
        Cached value or None if not found
    """
    if not REDIS_AVAILABLE:
        return None
    
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
    if not REDIS_AVAILABLE:
        return False
    
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
    if not REDIS_AVAILABLE:
        return False
    
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
    if not REDIS_AVAILABLE:
        return 0
    
    try:
        keys = redis_client.keys(pattern)
        if keys:
            return redis_client.delete(*keys)
        return 0
    except Exception as e:
        print(f"Cache clear error: {e}")
        return 0
