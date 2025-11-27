"""
Environment configuration for different deployment stages.
"""
from enum import Enum
from typing import Optional

class Environment(str, Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class EnvironmentConfig:
    """Base environment configuration."""
    
    def __init__(self, env: Environment):
        self.env = env
        self.debug = env == Environment.DEVELOPMENT
        self.testing = env == Environment.TESTING
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.env == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.env == Environment.DEVELOPMENT

class DevelopmentConfig(EnvironmentConfig):
    """Development environment configuration."""
    
    def __init__(self):
        super().__init__(Environment.DEVELOPMENT)
        self.reload = True
        self.log_level = "DEBUG"

class ProductionConfig(EnvironmentConfig):
    """Production environment configuration."""
    
    def __init__(self):
        super().__init__(Environment.PRODUCTION)
        self.reload = False
        self.log_level = "INFO"
        self.workers = 4

class TestingConfig(EnvironmentConfig):
    """Testing environment configuration."""
    
    def __init__(self):
        super().__init__(Environment.TESTING)
        self.log_level = "WARNING"

def get_environment_config(env: Optional[str] = None) -> EnvironmentConfig:
    """
    Get environment configuration based on environment name.
    
    Args:
        env: Environment name
        
    Returns:
        Environment configuration instance
    """
    if not env:
        return DevelopmentConfig()
    
    env_map = {
        Environment.DEVELOPMENT: DevelopmentConfig,
        Environment.PRODUCTION: ProductionConfig,
        Environment.TESTING: TestingConfig,
    }
    
    env_enum = Environment(env.lower())
    config_class = env_map.get(env_enum, DevelopmentConfig)
    return config_class()
