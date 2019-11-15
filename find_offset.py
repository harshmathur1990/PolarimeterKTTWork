import sys
import numpy as np
import modulation_matrix as mm
import sunpy.io.fits
import matplotlib.pyplot as plt


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

    input_stokes = input_stokes.astype(np.float64)

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
    offset_range_start, offset_range_end, step, top_beam, bottom_beam
):

    im_top, im_bot = mm.get_modulation_matrix(mm.config)

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
            # import ipdb;ipdb.set_trace()
            sys.stdout.write('Skipping for Offset {}\n'.format(offset))
            continue

        curr_score = get_squared_dist(normalised_top, im_top) + \
            get_squared_dist(normalised_bot, im_bot)

        sys.stdout.write('\n')
        sys.stdout.write('Offset: {}'.format(offset))
        sys.stdout.write(str(normalised_bot))
        sys.stdout.write('\n')
        sys.stdout.write(str(normalised_bot))
        sys.stdout.write('\n')
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


def read_file(filename):
    data, header = sunpy.io.fits.read(filename)[0]
    BEAMSEP = 512
    PTS = [[477, 692], [739, 835]]
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
    return INTENSITY_TOP.reshape(4, 24), INTENSITY_BOT.reshape(4, 24)


def plot_top_bottom_beam(top_beam, bottom_beam):
    plt.plot(top_beam.reshape(96, ), label='top line')
    plt.plot(bottom_beam.reshape(96, ), label='top line')
    plt.scatter(np.arange(96), top_beam.reshape(96, ), label='top')
    plt.scatter(np.arange(96), bottom_beam.reshape(96, ), label='bot')
    plt.xlabel('Measurements')
    plt.ylabel('Intensities')
    plt.title('Intensity Vs Measurement Curve')
    plt.legend()
    plt.show()


def execute(filename, offset_start, offset_end, step):
    top_beam, bottom_beam = read_file(filename)

    total_intensity = top_beam + bottom_beam

    a, b, c, d = np.polyfit(
        np.arange(96), total_intensity.reshape(96, ), deg=3
    )

    fitted_values = np.polyval((a, b, c, d), np.arange(96))

    top_beam = top_beam / fitted_values.reshape(4, 24)

    bottom_beam = bottom_beam / fitted_values.reshape(4, 24)

    top_beam /= np.max(top_beam)

    top_beam += 0.6

    bottom_beam /= np.max(bottom_beam)

    bottom_beam += 0.6

    plot_top_bottom_beam(top_beam, bottom_beam)

    score, res_offset, res_top, res_bot = reduce_offset(
        offset_range_start=offset_start,
        offset_range_end=offset_end,
        step=step,
        top_beam=top_beam,
        bottom_beam=bottom_beam
    )

    sys.stdout.write(
        'Score: {} Offset: {}\n'.format(score, res_offset)
    )

    sys.stdout.write(str(res_top))
    sys.stdout.write('\n')
    sys.stdout.write(str(res_bot))


if __name__ == '__main__':
    filename = '/Volumes/Harsh 9599771751/' +\
        'Spectropolarimetric Data Kodaikanal' +\
        '/2019/Level-1/20190413_093058_MOD4_CALIB24.fits'
    execute(filename, 0, 180, 1)
