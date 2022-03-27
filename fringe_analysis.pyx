cimport numpy
cimport cython
import sys
import copy
from pathlib import Path
import numpy as np
import scipy.signal
import scipy.ndimage
import sunpy.io.fits
import matplotlib.pyplot as plt
from pathlib import Path
import h5py



base_path = Path('/Volumes/HarshHDD-Data/Documents/CourseworkRepo/Polcal/Fringe Removal')



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
def do_calc():
    cdef numpy.ndarray[numpy.float64_t, ndim = 2] D_matrix
    cdef numpy.ndarray[numpy.float64_t, ndim = 2] C_matrix
    cdef numpy.ndarray[numpy.float64_t, ndim = 2] ej_matrix
    cdef numpy.ndarray[numpy.float64_t, ndim = 2] new_ej_matrix
    cdef numpy.ndarray[numpy.float64_t, ndim = 2] medfiltered_better_flat
    cdef numpy.ndarray[numpy.float64_t, ndim = 2] sel_region
    cdef numpy.ndarray[numpy.float64_t, ndim = 2] corr
    cdef numpy.float64_t score
    cdef numpy.ndarray[numpy.float64_t , ndim=2] new_D
    cdef numpy.ndarray[numpy.float64_t , ndim = 2] mean_new_D
    cdef numpy.float64_t ns

    f = h5py.File(base_path / 'Data.h5', 'r')
    D_matrix = f['D_matrix'][()]
    C_matrix = f['C_matrix'][()]
    ej_matrix = f['ej_matrix'][()]
    medfiltered_better_flat = f['medfiltered_better_flat'][()]
    nnnmean = np.mean(medfiltered_better_flat, 0)
    f.close()
    new_ej_matrix = ej_matrix.copy()
    sel_region = nnnmean[300:500, 50:250]
    corr = scipy.signal.fftconvolve(nnnmean, sel_region[::-1, ::-1], mode='same')
    score = np.mean(np.abs(corr))

    k = 0
    for i in range(D_matrix.shape[1]):
        for j in range(i + 1, D_matrix.shape[1], 1):
            for theta in np.arange(-np.pi, np.pi, 0.1):
                old_ei_xy = new_ej_matrix[:, i].copy()
                old_ej_xy = new_ej_matrix[:, j].copy()
                new_ej_matrix[:, i] = np.cos(theta) * old_ei_xy + np.sin(theta) * old_ej_xy
                new_ej_matrix[:, j] = -np.sin(theta) * old_ei_xy + np.cos(theta) * old_ej_xy
                new_D = np.dot(new_ej_matrix, C_matrix)
                mean_new_D = np.mean(new_D.T.reshape(90, 512, 512), 0)
                ns = np.mean(
                    np.abs(scipy.signal.fftconvolve(nnnmean, mean_new_D[300:500, 50:250][::-1, ::-1], mode='same')))
                if ns < score:
                    score = ns
                    sys.stdout.write('{}\n'.format(k))
                else:
                    new_ej_matrix[:, i] = old_ei_xy
                    new_ej_matrix[:, j] = old_ej_xy
                sys.stdout.write('{}\n'.format(k))
                k += 1

    sunpy.io.fits.write(base_path / 'new_2_ej_matrix.fits', new_ej_matrix, dict())