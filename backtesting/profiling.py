import cProfile
import io
import pstats
from functools import wraps

from line_profiler import LineProfiler


def profile_function(func):
    """
    Decorator that profiles a function using cProfile and prints stats
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative').print_stats(10)  # Show top 10 entries with module names
        print(s.getvalue())
        return result
    return wrapper

def line_profile_function(func):
    """
    Decorator that profiles a function line by line using line_profiler
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = LineProfiler()
        profiled_func = profiler(func)
        result = profiled_func(*args, **kwargs)
        profiler.print_stats()
        return result
    return wrapper 