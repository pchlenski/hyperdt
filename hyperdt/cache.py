"""Cache to keep track of split candidates in HyperDT"""

from functools import lru_cache, wraps
from threading import Lock


class SplitCache:
    """Helper class to manage thread-safe caching of splits across trees and nodes in a random forest"""

    def __init__(self):
        self.lock = Lock()

    def cache_decorator(self, func: callable) -> callable:
        """Decorator to cache the results of a function in a thread-safe manner"""
        cached_func = lru_cache()(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                return cached_func(*args, **kwargs)

        return wrapper
