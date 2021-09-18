import sys
from pathlib import Path
import numpy as np
import scipy.ndimage
import modulation_matrix as mm
import sunpy.io.fits
import matplotlib.pyplot as plt

# write_path = Path('/Users/harshmathur/CourseworkRepo/Level-1')


def normalise_modulation_matrix(mod_matrix):
    mod_matrix = np.copy(mod_matrix)
    m_0 = np.zeros(mod_matrix.shape[0])

    for i in np.arange(mod_matrix.shape[0]):
        m_0[i] = mod_matrix[i][0]
        mod_matrix[i] = mod_matrix[i] / m_0[i]

    return mod_matrix, m_0


def is_valid_mod_matrix(mod_matrix):

    for i in np.arange(mod_matrix.shape[0]):
        for j in np.arange(mod_matrix.shape[1]):
            if not -1 <= mod_matrix[i][j] <= 1:
                return False

    return True


def get_squared_dist(source, dest):
    diff = np.abs(np.abs(source) - np.abs(dest))
    least_sq_diff = np.sum(np.square(diff))
    k = len(np.where(diff < 0.4)[0])

    return least_sq_diff - ((k - 4) * 100)


def get_modulation_matrix(offset, top_beam, bottom_beam):
    input_stokes = mm.get_input_stokes(offset)

    input_mod = np.matmul(
        input_stokes.T,
        np.linalg.inv(
            np.matmul(
                input_stokes,
                input_stokes.T
            )
        )
    )
    modulation_matrix_top = np.matmul(
        top_beam,
        input_mod
    )

    normalised_top, m_0_top = normalise_modulation_matrix(
        modulation_matrix_top
    )

    modulation_matrix_bot = np.matmul(
        bottom_beam,
        input_mod
    )

    normalised_bot, m_0_bot = normalise_modulation_matrix(
        modulation_matrix_bot
    )

    return normalised_top, m_0_top, normalised_bot, m_0_bot


def reduce_offset(
    offset_range_start, offset_range_end,
    step, top_beam, bottom_beam, wavelength=8542
):

    im_top, im_bot = mm.get_modulation_matrix(
        mm.config, wavelength=wavelength
    )

    im_top = im_top.astype(np.float64)

    im_bot = im_bot.astype(np.float64)

    score = np.Inf

    res_offset = None

    res_top = None

    res_bot = None

    for offset in np.arange(
        offset_range_start, offset_range_end, step
    ):
        a, b, c, d = get_modulation_matrix(
            offset, top_beam, bottom_beam
        )

        normalised_top, _, normalised_bot, _ = a, b, c, d

        if not is_valid_mod_matrix(normalised_top) or \
                not is_valid_mod_matrix(normalised_bot):
            continue

        curr_score = get_squared_dist(normalised_top, im_top) + \
            get_squared_dist(normalised_bot, im_bot)

        if curr_score < score:
            score = curr_score
            res_offset = offset
            res_top = normalised_top
            res_bot = normalised_bot

        sys.stdout.write(
            'Offset: {} Score: {} Min Score: {} Min Offset: {}\n'.format(
                offset, curr_score, score, res_offset
            )
        )
    return score, res_offset, res_top, res_bot


def read_file_for_observations(filename, top_cut=530, bot_cut=560):
    data, header = sunpy.io.fits.read(filename)[0]

    return np.reshape(
        data[:, 0:top_cut, :],
        (4, data.shape[0] // 4, top_cut, data.shape[2]),
        order='F'
    ), np.reshape(
        data[:, bot_cut:, :],
        (4, data.shape[0] // 4, data.shape[1] - bot_cut, data.shape[2]),
        order='F'
    )


def read_file(
    calib_filename,
    x1=692, x2=835, y1=477, y2=739, BEAMSEP=512
):
    data, header = sunpy.io.fits.read(calib_filename)[0]
    PTS = [[y1, x1], [y2, x2]]
    ROI = [list(map(int, i)) for i in PTS]
    ROI_TOP = np.copy(ROI)
    ROI_BOT = np.copy(ROI_TOP)
    ROI_BOT[0][1] = ROI_TOP[0][1] - BEAMSEP
    ROI_BOT[1][1] = ROI_TOP[1][1] - BEAMSEP
    INTENSITY_TOP = np.sum(
        data[:, ROI_TOP[0][1]:ROI_TOP[1][1], ROI_TOP[0][0]:ROI_TOP[1][0]],
        (1, 2)
    )
    INTENSITY_BOT = np.sum(
        data[:, ROI_BOT[0][1]:ROI_BOT[1][1], ROI_BOT[0][0]:ROI_BOT[1][0]],
        (1, 2)
    )
    return INTENSITY_TOP.reshape(24, 4).T, INTENSITY_BOT.reshape(24, 4).T


def get_demodulation_matrix(mod_matrix):
    return np.matmul(
        np.linalg.inv(
            np.matmul(
                mod_matrix.T,
                mod_matrix
            )
        ),
        mod_matrix.T
    )


def plot_stokes(
    beam_name,
    beam,
    beam_theory
):
    fig1, axs1 = plt.subplots(2, 2)

    l1, = axs1[0][0].plot(beam[0])

    l2, = axs1[0][0].plot(beam_theory[0])

    l3, = axs1[0][1].plot(beam[1])

    l4, = axs1[0][1].plot(beam_theory[1])

    l5, = axs1[1][0].plot(beam[2])

    l6, = axs1[1][0].plot(beam_theory[2])

    l7, = axs1[1][1].plot(beam[3])

    l8, = axs1[1][1].plot(beam_theory[3])

    fig1.legend(
        (l1, l2), ('Mod I Practical', 'Mod I Theory'), 'upper left')

    fig1.legend((
        l3, l4), ('Mod II Practical', 'Mod II Theory'), 'upper right')

    fig1.legend(
        (l5, l6), ('Mod III Practical', 'Mod III Theory'), 'lower left')

    fig1.legend(
        (l7, l8), ('Mod IV Practical', 'Mod IV Theory'), 'lower right')

    fig1.suptitle('{}'.format(beam_name))

    fig1.tight_layout()

    plt.show()


def plot_beam(
    beam_name,
    beam,
    beam_theory
):
    fig1, axs1 = plt.subplots(2, 2)

    l1, = axs1[0][0].plot(list(np.arange(24)), beam[0])

    l2, = axs1[0][0].plot(list(np.arange(24)), beam_theory[0])

    l3, = axs1[0][1].plot(list(np.arange(24)), beam[1])

    l4, = axs1[0][1].plot(list(np.arange(24)), beam_theory[1])

    l5, = axs1[1][0].plot(list(np.arange(24)), beam[2])

    l6, = axs1[1][0].plot(list(np.arange(24)), beam_theory[2])

    l7, = axs1[1][1].plot(list(np.arange(24)), beam[3])

    l8, = axs1[1][1].plot(list(np.arange(24)), beam_theory[3])

    fig1.legend((l1, l2), ('Mod I Practical', 'Mod I Theory'), 'upper left')

    fig1.legend((l3, l4), ('Mod II Practical', 'Mod II Theory'), 'upper right')

    fig1.legend(
        (l5, l6), ('Mod III Practical', 'Mod III Theory'), 'lower left')

    fig1.legend((l7, l8), ('Mod IV Practical', 'Mod IV Theory'), 'lower right')

    fig1.suptitle('{}'.format(beam_name))

    fig1.tight_layout()

    # plt.legend()

    # plt.tight_layout()

    # plt.show()

    fig1.savefig(
        '{} Intensity Variation.png'.format(beam_name),
        dpi=300,
        format='png'
    )


def get_measured_stokes(filename):
    top_beam, bottom_beam = read_file(filename)

    modulation_matrix_top, modulation_matrix_bot = mm.get_modulation_matrix(
        mm.config
    )

    demod_top = get_demodulation_matrix(modulation_matrix_top)

    demod_bot = get_demodulation_matrix(modulation_matrix_bot)

    stokes_top = np.matmul(
        demod_top,
        top_beam
    )

    stokes_bot = np.matmul(
        demod_bot,
        bottom_beam
    )

    return stokes_top, stokes_bot


def get_standard_deviation(
    offset,
    response_matrix_top,
    response_matrix_bot,
    stokes_top,
    stokes_bot
):
    input_stokes = mm.get_input_stokes(offset=offset)

    out_stokes_top = np.matmul(
        response_matrix_top, input_stokes
    )

    out_stokes_bot = np.matmul(
        response_matrix_bot, input_stokes
    )

    return np.sum(np.square(out_stokes_top[3] - stokes_top[3])) + \
        np.sum(np.square(out_stokes_bot[3] - stokes_bot[3]))


def get_stokes_from_observations(
    top_beam, bottom_beam,
    modulation_matrix_top, modulation_matrix_bot
):
    demod_top = get_demodulation_matrix(modulation_matrix_top)

    demod_bot = get_demodulation_matrix(modulation_matrix_bot)

    stokes_top = np.einsum('ij, jklm-> iklm', demod_top, top_beam)

    stokes_bot = np.einsum('ij, jklm-> iklm', demod_bot, bottom_beam)

    return stokes_top, stokes_bot


def get_measured_stokes_observations(filename, calib_filename):
    top_beam, bottom_beam = read_file_for_observations(filename)

    modulation_matrix_top, modulation_matrix_bot = mm.get_modulation_matrix(
        mm.config
    )

    sign_matrix_top = modulation_matrix_top / np.abs(modulation_matrix_top)

    sign_matrix_bot = modulation_matrix_bot / np.abs(modulation_matrix_bot)

    demod_top = get_demodulation_matrix(sign_matrix_top)

    demod_bot = get_demodulation_matrix(sign_matrix_bot)

    stokes_top = np.einsum('ij, jklm-> iklm', demod_top, top_beam)

    stokes_bot = np.einsum('ij, jklm-> iklm', demod_bot, bottom_beam)

    a, b, c = get_response_matrix_add_subtract(calib_filename)

    response_matrix_top, response_matrix_bot, _ = a, b, c

    inverse_response_top = np.linalg.inv(response_matrix_top)

    inverse_response_bot = np.linalg.inv(response_matrix_bot)

    real_stokes_top = np.einsum(
        'ij, jklm-> iklm', inverse_response_top, stokes_top
    )

    real_stokes_bot = np.einsum(
        'ij, jklm-> iklm', inverse_response_bot, stokes_bot
    )

    return real_stokes_top, real_stokes_bot


def get_final_stokes_from_real_stokes(
    real_stokes_top,
    real_stokes_bot,
    header,
    top_s=0,
    top_e=464
):

    cropped_stokes_top = real_stokes_top[:, :, top_s:top_e, :]

    final_stokes = np.zeros_like(cropped_stokes_top)

    final_stokes[0] = cropped_stokes_top[0] + real_stokes_bot[0]

    final_stokes[1] = (cropped_stokes_top[1] / cropped_stokes_top[0]) + \
        (real_stokes_bot[1] / real_stokes_bot[0])

    final_stokes[2] = (cropped_stokes_top[2] / cropped_stokes_top[0]) + \
        (real_stokes_bot[2] / real_stokes_bot[0])

    final_stokes[3] = (cropped_stokes_top[3] / cropped_stokes_top[0]) + \
        (real_stokes_bot[3] / real_stokes_bot[0])

    final_stokes[1] = final_stokes[1] * final_stokes[0] / 2

    final_stokes[2] = final_stokes[2] * final_stokes[0] / 2

    final_stokes[3] = final_stokes[3] * final_stokes[0] / 2

    filename = write_path / 'final_stokes.fits'

    sunpy.io.fits.write(filename, final_stokes, header, overwrite=True)

    return final_stokes


def get_response_matrix_add_subtract(filename):
    top_beam, bottom_beam = read_file(filename)

    modulation_matrix_top, modulation_matrix_bot = mm.get_modulation_matrix(
        mm.config
    )

    sign_matrix_top = modulation_matrix_top / np.abs(modulation_matrix_top)

    sign_matrix_bot = modulation_matrix_bot / np.abs(modulation_matrix_bot)

    demod_top = get_demodulation_matrix(sign_matrix_top)

    demod_bot = get_demodulation_matrix(sign_matrix_bot)

    stokes_top = np.matmul(
        demod_top,
        top_beam
    )

    stokes_top = stokes_top / stokes_top[0]

    stokes_bot = np.matmul(
        demod_bot,
        bottom_beam
    )

    stokes_bot = stokes_bot / stokes_bot[0]

    min_std = np.Inf

    response_matrix_top, response_matrix_bot, res_offset = None, None, None

    for offset in np.arange(-15, -5, 0.01):

        input_stokes = mm.get_input_stokes(offset=offset)

        input_matrix = np.matmul(
            input_stokes.T,
            np.linalg.inv(
                np.matmul(
                    input_stokes,
                    input_stokes.T
                )
            )
        )

        _response_matrix_top = np.matmul(
            stokes_top,
            input_matrix
        )

        _response_matrix_bot = np.matmul(
            stokes_bot,
            input_matrix
        )

        std = get_standard_deviation(
            offset,
            _response_matrix_top,
            _response_matrix_bot,
            stokes_top,
            stokes_bot
        )

        if std < min_std:
            min_std = std
            response_matrix_bot = _response_matrix_bot
            response_matrix_top = _response_matrix_top
            res_offset = offset

    return response_matrix_top, response_matrix_bot, res_offset


def execute(filename, offset_start, offset_end, step):
    top_beam, bottom_beam = read_file(filename)

    total_intensity = top_beam + bottom_beam

    top_beam = top_beam * 2 / np.max(top_beam)

    bottom_beam = bottom_beam * 2 / np.max(bottom_beam)

    intensity_top_theory, intensity_bot_theory = mm.get_modulated_intensity(
        offset=0
    )

    total_intensity_theory = intensity_top_theory + intensity_bot_theory

    total_intensity = total_intensity * 2 / np.max(total_intensity)

    plot_beam(
        'Total Beam',
        total_intensity,
        total_intensity_theory
    )


def plot_stokes_data(data_binned, stokes_bined, pixel_x, pixel_y):
    fig, axs = plt.subplots(2, 2)

    axs[0][0].plot(data_binned[0, pixel_x, pixel_y, :])

    axs[0][0].set_title('Stokes I in Umbra')

    axs[0][1].plot(stokes_bined[1, pixel_x, pixel_y, :])

    axs[0][1].set_title('Stokes Q in Umbra')

    axs[1][0].plot(stokes_bined[2, pixel_x, pixel_y, :])

    axs[1][0].set_title('Stokes U in Umbra')

    axs[1][1].plot(stokes_bined[3, pixel_x, pixel_y, :])

    axs[1][1].set_title('Stokes V in Umbra')

    plt.legend()

    plt.show()


def get_binned_image(stokes_filename):
    data, header = sunpy.io.fits.read(stokes_filename)[0]

    scan_step = header['SCANSTEP']

    binning = int(header['BINNING'])

    pixel_size = 13.5

    image_binning = int(np.rint(1000 * scan_step / (pixel_size * binning)))

    temp_1 = (data.shape[2] % image_binning) // 2

    temp_2 = (data.shape[2] // image_binning) * image_binning

    data_cropped = data[:, :, temp_1: temp_1 + temp_2, :]

    data_binned = np.zeros(
        (
            data_cropped.shape[0],
            data_cropped.shape[1],
            data_cropped.shape[2] // image_binning,
            data_cropped.shape[3]
        )
    )

    for i in range(image_binning):
        data_binned += data_cropped[:, :, i::image_binning, :]

    return data_binned


def get_flat_profile(flat_filename):
    flat_profile = np.loadtxt(flat_filename)

    return scipy.ndimage.gaussian_filter1d(flat_profile, 1)


def get_efficiency_vector(modulation_matrix):
    demod = get_demodulation_matrix(modulation_matrix)
    return 1 / np.sqrt(
        modulation_matrix.shape[0] * np.sum(
            np.square(demod),
            axis=1
        )
    )


if __name__ == '__main__':
    calib_filename = '/Users/harshmathur/Documents/' + \
        'CourseworkRepo/Level-1/calib_data.fits'

    observation_filename = '/Users/harshmathur/Documents/' + \
        'CourseworkRepo/Level-1/observation_data.fits'

    real_stokes_top, real_stokes_bot = get_measured_stokes_observations(
        observation_filename, calib_filename
    )

    stokes_filename = '/Users/harshmathur/CourseworkRepo/' + \
        'Level-1/20190413_093058_STOKESDATA.fits'

    flat_filename = '/Users/harshmathur/CourseworkRepo/' + \
        'Level-1/20190413_083523_LINEPROFILE.txt'

    header = sunpy.io.read_file_header(observation_filename)[0]

    get_final_stokes_from_real_stokes(
        real_stokes_top,
        real_stokes_bot,
        header
    )
