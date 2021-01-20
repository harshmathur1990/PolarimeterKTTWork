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


def calculate_d(j1, j2):
    return j1 * (j1 + 1) - j2 * (j2 + 1)


def calculate_lande_g_eff(g1, g2, j1, j2):
    d = calculate_d(j1, j2)

    return 0.5 * (g1 + g2) + 0.25 * (g1 - g2) * d


def get_lande_g_factor_for_level(j, l, s):
    return 1 + (
        0.5 * (
            j * (j + 1) + s * (s + 1) - l * (l + 1)
        ) / (
            j * (j + 1)
        )
    )
