import time
from functools import wraps

def timed(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = fn(*args, **kwargs)
        end_time = time.perf_counter()
        print(f'{fn.__name__} completed in {(end_time-start_time):.4f} s.')
        return result
    return wrapper