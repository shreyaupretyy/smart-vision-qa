"""
Rate limiting middleware using Redis.
"""
import time
from typing import Callable
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from backend.core.cache import redis_client

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware to prevent API abuse.
    """
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            app: FastAPI application
            calls: Number of allowed calls
            period: Time period in seconds
        """
        super().__init__(app)
        self.calls = calls
        self.period = period
    
    async def dispatch(self, request: Request, call_next: Callable):
        """
        Process request with rate limiting.
        
        Args:
            request: HTTP request
            call_next: Next middleware/route handler
            
        Returns:
            HTTP response
        """
        # Get client identifier (IP address)
        client_ip = request.client.host
        
        # Create rate limit key
        key = f"rate_limit:{client_ip}"
        
        try:
            # Get current request count
            current = redis_client.get(key)
            
            if current is None:
                # First request in period
                redis_client.setex(key, self.period, 1)
            else:
                current_count = int(current)
                
                if current_count >= self.calls:
                    # Rate limit exceeded
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded. Please try again later."
                    )
                
                # Increment counter
                redis_client.incr(key)
        
        except HTTPException:
            raise
        except Exception as e:
            # Log error but don't block request
            print(f"Rate limit error: {e}")
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        try:
            remaining = max(0, self.calls - int(redis_client.get(key) or 0))
            response.headers["X-RateLimit-Limit"] = str(self.calls)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(time.time()) + self.period)
        except:
            pass
        
        return response
