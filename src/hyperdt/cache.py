"""Cache to keep track of split candidates in HyperDT"""

from functools import lru_cache, wraps
from threading import Lock


class SplitCache:
    """Helper class to manage thread-safe caching of splits across trees and nodes in a random forest"""

    def __init__(self):
        self.cache = lru_cache()
        self.lock = Lock

    def cache_decorator(self, func: callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock():
                return self.cache(func)(*args, **kwargs)

        return wrapper
