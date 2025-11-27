from functools import lru_cache
from backend.core.config import get_settings, Settings


@lru_cache()
def get_settings_dependency() -> Settings:
    """Dependency for settings injection"""
    return get_settings()
