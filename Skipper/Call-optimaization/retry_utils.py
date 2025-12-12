"""
Retry and Resilience Utilities
==============================
Provides retry logic with exponential backoff for API calls and file operations.
"""

import time
import logging
from typing import Callable, TypeVar, Any, Optional
from functools import wraps
import random

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""
    pass


class RetryStrategy:
    """Configurable retry strategy with exponential backoff."""
    
    def __init__(self,
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 30.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        """
        Initialize retry strategy.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Add random jitter to delay
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            # Add random jitter (±25%)
            jitter_amount = delay * 0.25
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)


def retry_on_exception(
    exceptions: tuple = (Exception,),
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    on_retry: Optional[Callable] = None
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        exceptions: Tuple of exception types to catch and retry
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        on_retry: Optional callback function called on each retry
        
    Example:
        @retry_on_exception(exceptions=(APIError,), max_attempts=5)
        def call_api():
            return api.request()
    """
    strategy = RetryStrategy(max_attempts, base_delay, max_delay, exponential_base)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(strategy.max_attempts):
                try:
                    return func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == strategy.max_attempts - 1:
                        # Last attempt failed
                        break
                    
                    delay = strategy.get_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{strategy.max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    time.sleep(delay)
            
            # All attempts exhausted
            logger.error(f"All {strategy.max_attempts} attempts failed for {func.__name__}")
            raise RetryError(
                f"Failed after {strategy.max_attempts} attempts: {last_exception}"
            ) from last_exception
        
        return wrapper
    return decorator


def with_timeout(seconds: int):
    """
    Decorator to add timeout to function execution.
    
    Args:
        seconds: Timeout in seconds
        
    Note: Uses threading, may not work for all scenarios
    """
    import signal
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            def timeout_handler(signum, frame):
                raise TimeoutError(f"{func.__name__} exceeded timeout of {seconds}s")
            
            # Set alarm (Unix only)
            try:
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                return result
            except AttributeError:
                # Windows doesn't have SIGALRM, just run without timeout
                logger.warning(f"Timeout not supported on this platform, running without timeout")
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing, requests rejected immediately
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = 'CLOSED'
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = 'HALF_OPEN'
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset or close circuit
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                logger.info("Circuit breaker CLOSED (recovered)")
            
            self.failure_count = 0
            return result
        
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
                logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")
            
            raise


# Example usage and convenience functions

def exponential_backoff(attempt: int, base: float = 1.0, max_delay: float = 60.0) -> float:
    """
    Calculate exponential backoff delay.
    
    Args:
        attempt: Attempt number (0-indexed)
        base: Base delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        Delay in seconds
    """
    return min(base * (2 ** attempt), max_delay)


# Pre-configured retry decorators for common scenarios

retry_api_call = retry_on_exception(
    exceptions=(ConnectionError, TimeoutError),
    max_attempts=5,
    base_delay=2.0,
    max_delay=60.0
)

retry_file_operation = retry_on_exception(
    exceptions=(IOError, OSError),
    max_attempts=3,
    base_delay=0.5,
    max_delay=5.0
)


if __name__ == "__main__":
    # Test retry mechanism
    attempt_count = 0
    
    @retry_on_exception(max_attempts=3, base_delay=0.1)
    def flaky_function():
        global attempt_count
        attempt_count += 1
        print(f"Attempt {attempt_count}")
        
        if attempt_count < 3:
            raise ConnectionError("Simulated failure")
        
        return "Success!"
    
    try:
        result = flaky_function()
        print(f"✅ {result}")
    except RetryError as e:
        print(f"❌ Failed: {e}")
