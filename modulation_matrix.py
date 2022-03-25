import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys

config = [[-22.5, 42.6], [-22.5, 69.9], [22.5, 47.4], [22.5, 20.1]]


def get_waveplate_matrix(retardation):
    def retard_waveplate_matrix(theta):
        cos_2_theta = np.cos(2 * theta)
        sin_2_theta = np.sin(2 * theta)
        cos_del = np.cos(retardation)
        sin_del = np.sin(retardation)
        q_1 = (cos_2_theta ** 2) + (sin_2_theta ** 2 * cos_del)
        q_2 = cos_2_theta * sin_2_theta * (1 - cos_del)
        q_3 = -1 * sin_2_theta * sin_del

        u_1 = q_2
        u_2 = (cos_2_theta ** 2 * cos_del) + (sin_2_theta ** 2)
        u_3 = cos_2_theta * sin_del

        v_1 = -1 * q_3
        v_2 = -1 * u_3
        v_3 = cos_del
        return np.array(
            [
                [1, 0, 0, 0],
                [0, q_1, q_2, q_3],
                [0, u_1, u_2, u_3],
                [0, v_1, v_2, v_3]
            ],
            dtype=object
        )
    return retard_waveplate_matrix


def get_linear_polarizer(sign=1):
    return np.array(
        [
            [1, sign, 0, 0],
            [sign, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )


def get_input_angles(offset=0, total=24):
    return np.radians(np.linspace(0, 180, total + 1)[1:] + offset)


def get_initial_stokes():
    return np.array([1, 1, 0, 0]).reshape(4, 1)


def get_input_stokes(offset=0, total=24):
    initial_stokes = get_initial_stokes()

    input_angles = get_input_angles(offset=offset, total=total)

    retardation = 2 * np.pi * 0.249

    waveplate_matrix = get_waveplate_matrix(retardation)

    vec_waveplate_matrix = np.vectorize(waveplate_matrix)

    wp_matrices = vec_waveplate_matrix(input_angles)

    stokes_res = None

    for wp_matrice in wp_matrices:
        _res = np.matmul(wp_matrice, initial_stokes)

        if stokes_res is None:
            stokes_res = _res
        else:
            stokes_res = np.concatenate((stokes_res, _res), axis=1)

    return stokes_res.astype(np.float64)


def get_modulation_matrix(config, original_wavelength=8542, wavelength=8542):

    modulation_matrix_top = None
    modulation_matrix_bottom = None

    quarter_retardation = 2 * np.pi * 0.249 * original_wavelength / wavelength
    half_retardation = 2 * np.pi * 0.249 * 2 * original_wavelength / wavelength

    qwp_matrix_func = get_waveplate_matrix(quarter_retardation)
    hwp_matrix_func = get_waveplate_matrix(half_retardation)

    top_retarder = get_linear_polarizer(1)
    bottom_retarder = get_linear_polarizer(-1)

    for conf in config:
        qwp_angle = np.radians(conf[0])
        hwp_angle = np.radians(conf[1])

        qwp_matrix = qwp_matrix_func(qwp_angle)
        hwp_matrix = hwp_matrix_func(hwp_angle)

        mueller_matrix_top = np.matmul(
            np.matmul(
                top_retarder,
                hwp_matrix
            ),
            qwp_matrix
        )

        mueller_matrix_bottom = np.matmul(
            np.matmul(
                bottom_retarder,
                hwp_matrix
            ),
            qwp_matrix
        )

        if modulation_matrix_top is None:
            modulation_matrix_top = mueller_matrix_top[0]
            modulation_matrix_top = modulation_matrix_top.reshape(1, 4)
        else:
            modulation_matrix_top = np.concatenate(
                (
                    modulation_matrix_top,
                    mueller_matrix_top[0].reshape(1, 4)
                ),
                axis=0
            )

        if modulation_matrix_bottom is None:
            modulation_matrix_bottom = mueller_matrix_bottom[0]
            modulation_matrix_bottom = modulation_matrix_bottom.reshape(1, 4)
        else:
            modulation_matrix_bottom = np.concatenate(
                (
                    modulation_matrix_bottom,
                    mueller_matrix_bottom[0].reshape(1, 4)
                ),
                axis=0
            )

    return modulation_matrix_top.astype(np.float64), \
        modulation_matrix_bottom.astype(np.float64)


def get_modulation_matrix_when_waveplates_exchanged(config, original_wavelength=8542, wavelength=8542):

    modulation_matrix_top = None
    modulation_matrix_bottom = None

    quarter_retardation = 2 * np.pi * 0.249 * original_wavelength / wavelength
    half_retardation = 2 * np.pi * 0.249 * 2 * original_wavelength / wavelength

    qwp_matrix_func = get_waveplate_matrix(quarter_retardation)
    hwp_matrix_func = get_waveplate_matrix(half_retardation)

    top_retarder = get_linear_polarizer(1)
    bottom_retarder = get_linear_polarizer(-1)

    for conf in config:
        hwp_angle = np.radians(conf[0])
        qwp_angle = np.radians(conf[1])

        hwp_matrix = hwp_matrix_func(hwp_angle)
        qwp_matrix = qwp_matrix_func(qwp_angle)

        mueller_matrix_top = np.matmul(
            np.matmul(
                top_retarder,
                qwp_matrix
            ),
            hwp_matrix
        )

        mueller_matrix_bottom = np.matmul(
            np.matmul(
                bottom_retarder,
                qwp_matrix
            ),
            hwp_matrix
        )

        if modulation_matrix_top is None:
            modulation_matrix_top = mueller_matrix_top[0]
            modulation_matrix_top = modulation_matrix_top.reshape(1, 4)
        else:
            modulation_matrix_top = np.concatenate(
                (
                    modulation_matrix_top,
                    mueller_matrix_top[0].reshape(1, 4)
                ),
                axis=0
            )

        if modulation_matrix_bottom is None:
            modulation_matrix_bottom = mueller_matrix_bottom[0]
            modulation_matrix_bottom = modulation_matrix_bottom.reshape(1, 4)
        else:
            modulation_matrix_bottom = np.concatenate(
                (
                    modulation_matrix_bottom,
                    mueller_matrix_bottom[0].reshape(1, 4)
                ),
                axis=0
            )

    return modulation_matrix_top.astype(np.float64), \
        modulation_matrix_bottom.astype(np.float64)


def calculate_efficiency_for_mod_matrix(mod_matrix):
    A = np.dot(mod_matrix.T, mod_matrix)

    B = np.linalg.inv(A)

    return 1/np.sqrt(4 * np.diag(B))


def prepare_modulation_matrix_minimisation_function(
    original_wavelength=8542, wavelength_1=8542.12, wavelength_2=6562.8
):
    def modulation_matrix_minimisation_function(config):

        optimum_mod_matrix = np.array(
            [
                [1., -0.57005914, 0.58262066, -0.57929763],
                [1., 0.57954106, -0.57321769, -0.57926992],
                [1., -0.58776753, -0.56491227, 0.57914028],
                [1., 0.56807082, 0.58468793, 0.57916799]
            ]
        )

        modulation_matrix_top_wave = np.zeros((len(config) // 2, 4))
        modulation_matrix_top_ori_wave = np.zeros((len(config) // 2, 4))

        quarter_retardation_wave = 2 * np.pi * 0.249 * original_wavelength / wavelength_1
        half_retardation_wave = 2 * np.pi * 0.249 * 2 * original_wavelength / wavelength_1

        quarter_retardation_ori_wave = 2 * np.pi * 0.249 * original_wavelength / wavelength_2
        half_retardation_ori_wave = 2 * np.pi * 0.249 * 2 * original_wavelength / wavelength_2

        qwp_matrix_func_wave = get_waveplate_matrix(quarter_retardation_wave)
        hwp_matrix_func_wave = get_waveplate_matrix(half_retardation_wave)

        qwp_matrix_func_ori_wave = get_waveplate_matrix(quarter_retardation_ori_wave)
        hwp_matrix_func_ori_wave = get_waveplate_matrix(half_retardation_ori_wave)

        top_retarder = get_linear_polarizer(1)

        for index in range(0, len(config), 2):
            qwp_angle = np.radians(config[index])
            hwp_angle = np.radians(config[index + 1])

            qwp_matrix_wave = qwp_matrix_func_wave(qwp_angle)
            hwp_matrix_wave = hwp_matrix_func_wave(hwp_angle)

            qwp_matrix_ori_wave = qwp_matrix_func_ori_wave(qwp_angle)
            hwp_matrix_ori_wave = hwp_matrix_func_ori_wave(hwp_angle)

            mueller_matrix_top_wave = np.matmul(
                np.matmul(
                    top_retarder,
                    hwp_matrix_wave
                ),
                qwp_matrix_wave
            )

            mueller_matrix_top_ori_wave = np.matmul(
                np.matmul(
                    top_retarder,
                    hwp_matrix_ori_wave
                ),
                qwp_matrix_ori_wave
            )

            modulation_matrix_top_wave[index // 2] = mueller_matrix_top_wave[0]
            modulation_matrix_top_ori_wave[index // 2] = mueller_matrix_top_ori_wave[0]

        # penalty = 0

        # for i in range(1, 4):
        #     for j in range(1, 4):
        #         if np.abs(modulation_matrix_top_wave[i, j]) < 0.4:
        #             penalty += np.square(
        #                 0.4 - np.abs(modulation_matrix_top_wave[i, j])
        #             )
        #         if np.abs(modulation_matrix_top_wave[i, j]) > 0.6:
        #             penalty += np.square(
        #                 np.abs(modulation_matrix_top_wave[i, j]) - 0.6
        #             )
        #         if np.abs(modulation_matrix_top_ori_wave[i, j]) < 0.4:
        #             penalty += np.square(
        #                 0.4 - np.abs(modulation_matrix_top_ori_wave[i, j])
        #             )
        #         if np.abs(modulation_matrix_top_ori_wave[i, j]) > 0.6:
        #             penalty += np.square(
        #                 np.abs(modulation_matrix_top_ori_wave[i, j]) - 0.6
        #             )

        # penalty = np.sqrt(penalty)

        # merit_value = np.sqrt(
        #     np.sum(
        #         np.square(
        #             np.subtract(
        #                 np.abs(optimum_mod_matrix),
        #                 np.abs(modulation_matrix_top_wave)
        #             )
        #         )
        #     )
        # ) + np.sqrt(
        #     np.sum(
        #         np.square(
        #             np.subtract(
        #                 np.abs(optimum_mod_matrix),
        #                 np.abs(modulation_matrix_top_ori_wave)
        #             )
        #         )
        #     )
        # )

        # print (merit_value, penalty)

        efficiency_vec_top = calculate_efficiency_for_mod_matrix(
            modulation_matrix_top_wave
        )

        efficiency_vec_ori_top = calculate_efficiency_for_mod_matrix(
            modulation_matrix_top_ori_wave
        )

        sys.stdout.write('Wave 1: {}, {}, {}, {}\t'.format(*list(efficiency_vec_top)))
        sys.stdout.write('Wave 2: {}, {}, {}, {}\n'.format(*list(efficiency_vec_ori_top)))

        return np.sum(1/efficiency_vec_top) + np.sum(1/efficiency_vec_ori_top)

    return modulation_matrix_minimisation_function


def find_modulation_scheme():
    modulation_matrix_minimisation_function = prepare_modulation_matrix_minimisation_function(
        original_wavelength=6300,
        wavelength_1=6563,
        wavelength_2=8662
    )

    res = minimize(
        modulation_matrix_minimisation_function,
        config,
        method='Nelder-Mead',
        tol=1e-6,
        options={
            'maxiter': 30000
        }
    )

    cfg = res.x.reshape(4, 2)

    '''
    In [107]: cfg
    Out[107]: 
    array([[-68.10399891,  17.62235885],
           [-30.73527284, 106.02538276],
           [ 62.40551554,  71.15438246],
           [  9.76832059,  57.16230412]])

    In [108]: get_modulation_matrix(cfg, original_wavelength=6300, wavelength=6562)
    Out[108]: 
    (array([[ 1.        ,  0.67782261,  0.58275496, -0.44827799],
            [ 1.        , -0.29009536,  0.49873564,  0.81676646],
            [ 1.        ,  0.44698922, -0.83019619,  0.33312899],
            [ 1.        , -0.84579809, -0.20331839, -0.49324155]]),
     array([[ 1.        , -0.67782261, -0.58275496,  0.44827799],
            [ 1.        ,  0.29009536, -0.49873564, -0.81676646],
            [ 1.        , -0.44698922,  0.83019619, -0.33312899],
            [ 1.        ,  0.84579809,  0.20331839,  0.49324155]]))

    In [109]: get_modulation_matrix(cfg, original_wavelength=6300, wavelength=8662)
    Out[109]: 
    (array([[ 1.        ,  0.82708773,  0.38460758, -0.40988156],
            [ 1.        , -0.20711692,  0.33733864,  0.91831652],
            [ 1.        ,  0.10358601, -0.99227222, -0.06830655],
            [ 1.        , -0.66833404,  0.22666632, -0.7084857 ]]),
     array([[ 1.        , -0.82708773, -0.38460758,  0.40988156],
            [ 1.        ,  0.20711692, -0.33733864, -0.91831652],
            [ 1.        , -0.10358601,  0.99227222,  0.06830655],
            [ 1.        ,  0.66833404, -0.22666632,  0.7084857 ]]))
 
    '''

    pass


def get_modulated_intensity(offset=0, wavelength=8542):
    input_stokes = get_input_stokes(offset=offset)

    modulation_matrix_top, modulation_matrix_bottom = get_modulation_matrix(
        config, wavelength=wavelength
    )

    intensity_top = np.matmul(
        modulation_matrix_top, input_stokes
    )

    intensity_bot = np.matmul(
        modulation_matrix_bottom, input_stokes
    )

    return intensity_top, intensity_bot


def save_top_bottom_intensity_curve():
    intensity_top, intensity_bot = get_modulated_intensity()

    fig = plt.figure()

    plt.scatter(np.arange(96), intensity_top.T.ravel(), label='top')

    plt.scatter(np.arange(96), intensity_bot.T.ravel(), label='bot')

    plt.plot(intensity_top.T.ravel(), label='top line')

    plt.plot(intensity_bot.T.ravel(), label='bot line')

    plt.title('Intensities vs Measurements')

    plt.xlabel('No of Measurement')

    plt.ylabel('Intensity')

    plt.legend()

    fig.tight_layout()

    plt.savefig('intensity_vs_measurements.png', format='png', dpi=300)


# if __name__ == '__main__':
    # save_top_bottom_intensity_curve()