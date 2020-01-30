from collections import defaultdict
from pathlib import Path
import copy
import numpy as np
import sunpy.io.fits
import sys
"""
In this example we will load the NINO3 sea surface temperature anomaly dataset
between 1871 and 1996. This and other sample data files are kindly provided by
C. Torrence and G. Compo at
<http://paos.colorado.edu/research/wavelets/software.html>.

"""
# We begin by importing the relevant libraries. Please make sure that PyCWT is
# properly installed in your system.
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pycwt as wavelet
from pycwt.helpers import find
import pandas as pd

# Then, we load the dataset and define some data related parameters. In this
# case, the first 19 lines of the data file contain meta-data, that we ignore,
# since we set them manually (*i.e.* title, units).


def plot_wavelet_trandform(
    dat, t,
    period, power, coi, wave,
    scales, dt, dj, mother, sig95,
    glbl_power, glbl_signif,
    scale_avg_signif, scale_avg,
    std, iwave, var,
    fft_theor, fft_power, fftfreqs
):

    title = 'Fringes across row'
    label = 'Fringes in Image'
    units = 'Pixels'

    '''
    Finally, we plot our results in four different subplots containing the
    (i) original series anomaly and the inverse wavelet transform; (ii) the
    wavelet power spectrum (iii) the global wavelet and Fourier spectra ; and
    (iv) the range averaged wavelet spectrum. In all sub-plots the significance
    levels are either included as dotted lines or as filled contour lines.
    '''

    # Prepare the figure
    plt.close('all')
    plt.ioff()
    figprops = dict(figsize=(11, 8), dpi=72)
    # fig = plt.figure(**figprops)
    plt.figure(**figprops)

    # First sub-plot, the original time series anomaly and inverse wavelet
    # transform.
    ax = plt.axes([0.1, 0.75, 0.65, 0.2])
    ax.plot(t, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
    ax.plot(t, dat, 'k', linewidth=1.5)
    ax.set_title('a) {}'.format(title))
    ax.set_ylabel('{} [{}]'.format(label, units))

    # Second sub-plot, the normalized wavelet power spectrum and significance
    # level contour lines and cone of influece hatched area. Note that period
    # scale is logarithmic.
    bx = plt.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    im = bx.contourf(
        t,
        np.log2(period),
        np.log2(power),
        np.log2(levels),
        extend='both',
        cmap=plt.cm.viridis
    )
    plt.colorbar(im, ax=bx)
    extent = [t.min(), t.max(), 0, max(period)]
    bx.contour(
        t,
        np.log2(period),
        sig95,
        [-99, 1],
        colors='k',
        linewidths=2,
        extent=extent
    )
    bx.fill(
        np.concatenate(
            [
                t,
                t[-1:] + dt,
                t[-1:] + dt,
                t[:1] - dt,
                t[:1] - dt
            ]
        ),
        np.concatenate(
            [
                np.log2(coi),
                [1e-9],
                np.log2(period[-1:]),
                np.log2(period[-1:]),
                [1e-9]
            ]
        ),
        'k',
        alpha=0.3,
        hatch='x'
    )
    bx.set_title(
        'b) {} Wavelet Power Spectrum ({})'.format(
            label, mother.name
        )
    )
    bx.set_ylabel('Period (Pixels)')
    #
    Yticks = 2 ** np.arange(
        np.ceil(
            np.log2(
                period.min()
            )
        ),
        np.ceil(
            np.log2(
                period.max()
            )
        )
    )
    bx.set_yticks(np.log2(Yticks))
    bx.set_yticklabels(Yticks)

    '''
    Third sub-plot, the global wavelet and Fourier power spectra
    and theoretical noise spectra. Note that period scale is logarithmic.
    '''
    cx = plt.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
    cx.plot(glbl_signif, np.log2(period), 'k--')
    cx.plot(var * fft_theor, np.log2(period), '--', color='#cccccc')
    cx.plot(
        var * fft_power,
        np.log2(1. / fftfreqs),
        '-',
        color='#cccccc',
        linewidth=1.
    )
    cx.plot(var * glbl_power, np.log2(period), 'k-', linewidth=1.5)
    cx.set_title('c) Global Wavelet Spectrum')
    cx.set_xlabel(r'Power [({})^2]'.format(units))
    cx.set_xlim([0, glbl_power.max() + var])
    cx.set_ylim(np.log2([period.min(), period.max()]))
    cx.set_yticks(np.log2(Yticks))
    cx.set_yticklabels(Yticks)
    plt.setp(cx.get_yticklabels(), visible=False)

    # Fourth sub-plot, the scale averaged wavelet spectrum.
    dx = plt.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
    dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
    dx.plot(t, scale_avg, 'k-', linewidth=1.5)
    dx.set_title('d) {}--{} pixel scale-averaged power'.format(2, 8))
    dx.set_xlabel('Time (pixel)')
    dx.set_ylabel(r'Average variance [{}]'.format(units))
    ax.set_xlim([t.min(), t.max()])

    plt.show()


def do_wavelet_transform(dat, dt):

    t0 = 0
    # dt = 0.25  # In years

    # We also create a time array in years.
    N = dat.size
    t = np.arange(0, N) * dt + t0

    '''
    We write the following code to detrend and normalize the input data by its
    standard deviation. Sometimes detrending is not necessary and simply
    removing the mean value is good enough. However, if your dataset has a well
    defined trend, such as the Mauna Loa CO\ :sub:`2` dataset available in the
    above mentioned website, it is strongly advised to perform detrending.
    Here, we fit a one-degree polynomial function and then subtract it from the
    original data.
    '''
    p = np.polyfit(t - t0, dat, 1)
    dat_notrend = dat - np.polyval(p, t - t0)
    std = dat_notrend.std()  # Standard deviation
    var = std ** 2  # Variance
    dat_norm = dat_notrend / std  # Normalized dataset

    # The next step is to define some parameters of our wavelet analysis. We
    # select the mother wavelet, in this case the Morlet wavelet with
    # :math:`\omega_0=6`.
    mother = wavelet.Morlet(6)
    s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
    dj = 1 / 12  # Twelve sub-octaves per octaves
    J = 7 / dj  # Seven powers of two with dj sub-octaves
    sr = pd.Series(dat)
    alpha = sr.autocorr(lag=1)  # Lag-1 autocorrelation for red noise

    '''
    The following routines perform the wavelet transform and inverse wavelet
    transform using the parameters defined above. Since we have normalized our
    input time-series, we multiply the inverse transform by the standard
    deviation.
    '''
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
        dat_norm, dt, dj, s0, J, mother
    )
    iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

    # We calculate the normalized wavelet and Fourier power spectra, as well as
    # the Fourier equivalent periods for each wavelet scale.
    power = (np.abs(wave)) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs

    '''
    We could stop at this point and plot our results. However we are also
    interested in the power spectra significance test. The power is significant
    where the ratio ``power / sig95 > 1``.
    '''
    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                             significance_level=0.95,
                                             wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95

    # Then, we calculate the global wavelet spectrum and determine its
    # significance level.
    glbl_power = power.mean(axis=1)
    dof = N - scales  # Correction for padding at edges
    glbl_signif, tmp = wavelet.significance(var, dt, scales, 1, alpha,
                                            significance_level=0.95, dof=dof,
                                            wavelet=mother)

    # We also calculate the scale average between 2 years and 8 years, and its
    # significance level.
    sel = find((period >= 2) & (period < 8))
    Cdelta = mother.cdelta
    scale_avg = (scales * np.ones((N, 1))).transpose()
    # As in Torrence and Compo (1998) equation 24
    scale_avg = power / scale_avg
    scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
    scale_avg_signif, tmp = wavelet.significance(var, dt, scales, 2, alpha,
                                                 significance_level=0.95,
                                                 dof=[scales[sel[0]],
                                                      scales[sel[-1]]],
                                                 wavelet=mother)

    return dat, t, \
        period, power, coi, wave, \
        scales, dt, dj, mother, sig95, \
        glbl_power, glbl_signif, \
        scale_avg_signif, scale_avg, \
        std, iwave, var, \
        fft_theor, fft_power, fftfreqs


def stride_vertically(image, window_size):
    nrows = image.shape[0] - window_size + 1
    stride = image.strides[-1]
    return np.lib.stride_tricks.as_strided(
        image,
        shape=(nrows, image.shape[1], window_size),
        strides=(stride * image.shape[0], stride, stride * image.shape[1]),
        writeable=False
    )


def read_flat_file(flat_filename):
    data, header = sunpy.io.fits.read(flat_filename)[0]

    avg_data = np.mean(data, axis=0)

    return avg_data, header


def read_dark_data(dark_filename):
    return sunpy.io.fits.read(dark_filename)[0]


def smooth_image(image, kernel=41):
    return np.median(stride_vertically(image, kernel), axis=2)


def get_dark_corrected_data(flat_filename, dark_filename):
    flat_data, flat_header = read_flat_file(flat_filename)
    dark_data, _ = read_dark_data(dark_filename)

    dark_corrected_flat = flat_data - dark_data

    return dark_corrected_flat, flat_header


def get_minima_points(
    power, wave, j, period, coi,
    period_estimate=None, row_left=None, row_right=None,
    col_left=None, col_right=None
):
    '''
    objective is to find minimas near the period estimate

    I have the period estimate, and old_left and old_right
    which are hard bounds for left and right minimas so that
    I don't consider more than what is required.

    '''
    y = power.T[j]
    peaks = np.where(
        (y[1: -1] > y[0: -2]) * (y[1: -1] > y[2:])
    )[0] + 1
    dips = np.where(
        (y[1: -1] < y[0: -2]) * (y[1: -1] < y[2:])
    )[0] + 1

    if period_estimate is None:
        maxima = peaks[np.argmax(y[peaks])]
    else:
        index_gt_period = np.where(
            np.round(period[peaks]) >= np.round(period_estimate)
        )[0]

        if index_gt_period.size == 0:
            maxima = peaks[-1]
        else:
            maxima = peaks[index_gt_period[0]]

    left_minima_arr = np.where(dips < maxima)[0]
    if left_minima_arr.size == 0:
        left_minima_point = 0
    else:
        left_minima_point = dips[left_minima_arr[-1]]

    right_minima_arr = np.where(dips > maxima)[0]

    if right_minima_arr.size == 0:
        right_minima_point = len(y) - 1
    else:
        right_minima_point = dips[right_minima_arr[0]]

    return left_minima_point, right_minima_point


def do_work(
    work_dict, shape_0, shape_1,
    period_estimate=None, kill_coi=True,
    ref_row=300, ref_col=540
):

    result = np.zeros(shape=(shape_0, shape_1), dtype=np.complex)

    result_dict = defaultdict(dict)

    second_iter = np.array(
        [ref_col] + list(np.arange(ref_col)[::-1]) +
        list(np.arange(ref_col + 1, shape_1))
    )

    first_iter = np.array(
        [ref_row] + list(np.arange(ref_row)[::-1]) +
        list(np.arange(ref_row + 1, shape_0))
    )

    old_left_right_dict = defaultdict(dict)

    for i in first_iter:
        power = work_dict[i]['power'].copy()
        wave = work_dict[i]['wave'].copy()
        scales = work_dict[i]['scales']
        dt = work_dict[i]['dt']
        dj = work_dict[i]['dj']
        mother = work_dict[i]['mother']
        std = work_dict[i]['std']
        period = work_dict[i]['period']
        coi = work_dict[i]['coi']
        for k in second_iter:
            if i == ref_row and k == ref_col:
                prev_col, prev_row = None, None
            elif i == ref_row:
                prev_row = None
                prev_col = k + 1 if k < ref_col else k - 1
            elif k == ref_col:
                prev_col = None
                prev_row = i + 1 if i < ref_row else i - 1
            else:
                prev_col = k + 1 if k < ref_col else k - 1
                prev_row = i + 1 if i < ref_row else i - 1

            row_left, row_right = None, None
            col_left, col_right = None, None

            if prev_row:
                row_left, row_right = old_left_right_dict[prev_row][k]

            if prev_col:
                col_left, col_right = old_left_right_dict[i][prev_col]

            left_minima_point, right_minima_point = get_minima_points(
                power,
                wave,
                k,
                period,
                coi,
                period_estimate=period_estimate[k],
                row_left=row_left,
                row_right=row_right,
                col_left=col_left,
                col_right=col_right
            )
            old_left_right_dict[i][k] = (
                left_minima_point, right_minima_point
            )
            wave.T[k][0: left_minima_point] = 0
            wave.T[k][right_minima_point:] = 0
            if kill_coi:
                wave.T[k][np.where(period > coi[k])] = 0
                power.T[k][np.where(period > coi[k])] = 0
            power.T[k][0: left_minima_point] = 0
            power.T[k][right_minima_point:] = 0
            sys.stdout.write(
                'Worked for Row: {} and Column: {}\n'.format(
                    i, k
                )
            )

        result_dict[i].update(
            dat=work_dict[i]['dat'],
            t=work_dict[i]['t'],
            period=period,
            power=power,
            coi=coi,
            wave=wave,
            scales=scales,
            dt=dt,
            dj=dj,
            mother=mother,
            sig95=work_dict[i]['sig95'],
            glbl_power=work_dict[i]['glbl_power'],
            glbl_signif=work_dict[i]['glbl_signif'],
            scale_avg_signif=work_dict[i]['scale_avg_signif'],
            scale_avg=work_dict[i]['scale_avg'],
            std=work_dict[i]['std'],
            iwave=work_dict[i]['iwave'],
            var=work_dict[i]['var'],
            fft_theor=work_dict[i]['fft_theor'],
            fft_power=work_dict[i]['fft_power'],
            fftfreqs=work_dict[i]['fftfreqs']
        )
        result[i] = wavelet.icwt(
            wave,
            scales,
            dt,
            dj,
            mother
        ) * std

    return result, result_dict, old_left_right_dict


def clean_data(result, kernel=7):
    res = result.copy()
    res[np.where(res == 0)] = np.nan
    mn = np.nanmean(res)
    sd = np.nanstd(res)

    lim_left = mn - 3 * sd
    lim_right = mn + 3 * sd

    rr, cc = np.where((res > lim_left) & (res < lim_right))

    left_win = kernel // 2
    right_win = kernel // 2 + 1

    for x, y in zip(rr, cc):
        left_x = x - left_win if x - left_win >= 0 else 0
        right_x = x + right_win if x + right_win < res.shape[0] \
            else res.shape[0] - 1

        left_y = y - left_win if y - left_win >= 0 else 0
        right_y = y + right_win if y + right_win < res.shape[1] else \
            res.shape[1] - 1

        res[x][y] = np.nanmedian(res[left_x: right_x, left_y: right_y])

    res[np.isnan(res)] = 0

    return res


def do_wavelet_over_image(image):

    first_iteration = np.array(
        [300] + list(np.arange(300)) + list(np.arange(301, image.shape[0]))
    )

    result_dict = defaultdict(dict)

    for i in first_iteration:

        dat, t, period, power, coi, wave, scales, dt, dj, mother, sig95, glbl_power, glbl_signif, scale_avg_signif, scale_avg, std, iwave, var, fft_theor, fft_power, fftfreqs = do_wavelet_transform(
            image[i], 1
        )

        result_dict[i].update(
            dat=dat,
            t=t,
            period=period,
            power=power,
            coi=coi,
            wave=wave,
            scales=scales,
            dt=dt,
            dj=dj,
            mother=mother,
            sig95=sig95,
            glbl_power=glbl_power,
            glbl_signif=glbl_signif,
            scale_avg_signif=scale_avg_signif,
            scale_avg=scale_avg,
            std=std,
            iwave=iwave,
            var=var,
            fft_theor=fft_theor,
            fft_power=fft_power,
            fftfreqs=fftfreqs
        )

    return result_dict


if __name__ == '__main__':
    flat_filename = Path(
        '/Users/harshmathur/Documents/CourseworkRepo' +
        '/20190413/Flats/083523_FLAT.fits'
    )

    dark_filename = Path(
        '/Users/harshmathur/Documents/' +
        'CourseworkRepo/Level-1/083651_DARKMASTER.fits'
    )

    dark_corrected_flat, flat_header = get_dark_corrected_data(
        flat_filename, dark_filename
    )

    image = dark_corrected_flat
