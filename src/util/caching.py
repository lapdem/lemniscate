import functools
import numpy as np


cache = {}


def cached(func):
    @functools.wraps(func)
    def cache_wrapper(*args):
        key = tuple(
            map(lambda x: x.tostring() if isinstance(x, np.ndarray) else x, args)
        )
        if func.__name__ not in cache:
            cache[func.__name__] = {}
        if key in cache[func.__name__]:
            return cache[func.__name__][key]
        result = func(*args)
        cache[func.__name__][key] = result
        return result

    return cache_wrapper
