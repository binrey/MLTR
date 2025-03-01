import cProfile
import io
import os
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
        
        # Get project root directory (2 levels up from this file)
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Process stats to show relative paths
        stats_dict = ps.stats
        new_stats = {}
        for (filename, line, name), value in stats_dict.items():
            if filename and isinstance(filename, str):
                if filename.startswith(project_dir):
                    filename = os.path.relpath(filename, project_dir)
            new_stats[(filename, line, name)] = value
            
        ps.stats = new_stats
        ps.sort_stats('cumulative').print_stats(10)  # Show top 10 entries with relative paths
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