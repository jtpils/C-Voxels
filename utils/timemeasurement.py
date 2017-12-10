import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        return result, (te - ts)

    return timed


@timeit
def exec_func(function, *args, **kwargs):
    return function(*args, **kwargs)


def time_func(func, *args, **kwargs):
    res, exec_time = exec_func(func, *args, **kwargs)
    print("{} took {} secs".format(func.__name__, exec_time))
    return res
