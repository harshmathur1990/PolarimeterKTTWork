"""
Created on Sun May 19 14:27:01 2024

@author: Raveena Khan, Harsh Mathur
"""


import numpy as np
from scipy.optimize import minimize


def Rot_mat_opt(angle):
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(2 * angle), np.sin(2 * angle), 0],
            [0, -np.sin(2 * angle), np.cos(2 * angle), 0],
            [0, 0, 0, 1]
        ]
    )


def calculate_efficiency_for_mod_matrix(mod_matrix):
    efficiency = np.sqrt(
        np.sum(
            np.square(mod_matrix),
            0
        ) / mod_matrix.shape[0]
    )

    return efficiency


def get_modulation_matrix(mod_ang):

    P = np.array(
        [
            [0.49065484, -0.2468864, -0.09962462, 0.14381742],
            [-0.22544269, 0.204288, 0.0314095, -0.39648331],
            [0.19553534, -0.31721242, -0.28480333, -0.07469363],
            [0.05005178, -0.25996385, 0.27726322, -0.08368225]
        ]
    )

    a1, a2, a3, a4 = mod_ang
    f1 = Rot_mat_opt(-a1) @ P @ Rot_mat_opt(a1)
    f2 = Rot_mat_opt(-a2) @ P @ Rot_mat_opt(a2)
    f3 = Rot_mat_opt(-a3) @ P @ Rot_mat_opt(a3)
    f4 = Rot_mat_opt(-a4) @ P @ Rot_mat_opt(a4)

    f_mod = np.array(
        [
            np.divide(f1[0, 0:4], f1[0, 0]),
            np.divide(f2[0, 0:4], f2[0, 0]),
            np.divide(f3[0, 0:4], f3[0, 0]),
            np.divide(f4[0, 0:4], f4[0, 0])
        ]
    )

    mod_matrix = f_mod

    return mod_matrix


def modulation_matrix_minimisation_function(mod_ang):

    mod_matrix = get_modulation_matrix(mod_ang)

    efficiency = calculate_efficiency_for_mod_matrix(mod_matrix)

    return 1 / np.sum(efficiency)


def find_modulation_scheme():

    mod_ang = [0, 15, 30, 45]

    config = np.deg2rad(mod_ang)

    res = minimize(
        modulation_matrix_minimisation_function,
        config,
        method='Nelder-Mead',
        tol=1e-6,
        options={
            'maxiter': 30000
        }
    )

    print(res)

    cfg = res.x

    print('Angles: {}'.format(np.round(np.rad2deg(cfg), 1)))

    mod_matrix = get_modulation_matrix(cfg)

    print('Modulation matrix: {}'.format(mod_matrix))

    print('Efficiency: {}'.format(calculate_efficiency_for_mod_matrix(mod_matrix)))


if __name__ == '__main__':
    find_modulation_scheme()
