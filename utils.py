import sys
import time
import contextlib


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        sys.stdout.write(
            '{} : {} ms\n'.format(
                method.__name__, (te - ts) * 1000
            )
        )
        return result
    return timed


@contextlib.contextmanager
def time_measure(ident):
    tstart = time.time()
    yield
    elapsed = time.time() - tstart
    sys.stdout.write('{0}: {1} ms'.format(ident, elapsed))
