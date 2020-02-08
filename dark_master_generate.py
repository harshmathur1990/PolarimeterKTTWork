import sys
import copy
from pathlib import Path
import numpy as np
import scipy.signal
import scipy.ndimage
import sunpy.io.fits
import matplotlib.pyplot as plt


# write_path = Path('/Users/harshmathur/Documents/CourseworkRepo/Kodai Visit/20200207')
# write_path = Path('/Users/harshmathur/Documents/CourseworkRepo/Level-1')
write_path = Path('/Volumes/Harsh 9599771751/Kodai Visit Processed/20200208')


def generate_master_dark(dark_filename):
    data, header = sunpy.io.fits.read(dark_filename)[0]
    average_data = np.average(data, axis=0)
    smoothed_data = scipy.signal.medfilt2d(average_data, kernel_size=3)
    smoothed_data[np.where(smoothed_data == 0)] = np.mean(smoothed_data)
    smoothed_data[
        np.where(
            np.abs(
                smoothed_data - smoothed_data.mean()
            ) > 4 * smoothed_data.std()
        )
    ] = smoothed_data.mean()
    save_path = write_path / dark_filename.name
    sunpy.io.fits.write(save_path, smoothed_data, header)


def generate_fringe_flat(fringe_filename, dark_master):
    data, header = sunpy.io.fits.read(fringe_filename)[0]

    fringe_data = np.mean(data, axis=0)

    dark_data, _ = sunpy.io.fits.read(dark_master)[0]

    fringe_data -= dark_data

    no_of_points = 50

    plt.imshow(fringe_data, cmap='gray', origin='lower')

    plt.title(
        'Select {} points in bright fringe in top beam across the image'
        .format(
            no_of_points
        )
    )

    points = plt.ginput(no_of_points)

    intensity = np.zeros(no_of_points)

    x_position = np.zeros(no_of_points)

    x_bin, y_bin = 2, 2

    for i, point in enumerate(points):
        x_beg, x_end = int(point[0] - x_bin), int(point[0] + x_bin)
        y_beg, y_end = int(point[1] - y_bin), int(point[1] + y_bin)
        intensity[i] = np.median(fringe_data[y_beg:y_end, x_beg:x_end])
        x_position[i] = point[0]

    intensity = intensity / np.max(intensity)

    c = np.polyfit(x_position, intensity, 3, w=intensity)

    x = np.arange(fringe_data.shape[1])

    intensity_fit = c[0] * x**3 + c[1] * x**2 + c[2] * x + c[3]

    intensity_mask = np.ones(fringe_data.shape) * np.array([intensity_fit])

    intensity_mask = intensity_mask / intensity_mask.max()

    fringe_corrected = fringe_data / intensity_mask

    fringe_write_path = write_path / (
        fringe_filename.name.split('.')[-2] + 'FRINGEFLAT.fits'
    )

    sunpy.io.fits.write(fringe_write_path, fringe_corrected, header)


def get_y_shift(x_corrected_flat):
    max_shift, extent, up_sampling = 5, 20, 10

    no_of_vertical_pixels = x_corrected_flat.shape[0]

    vertical_indices = np.arange(no_of_vertical_pixels)

    shift_horizontal = np.zeros(no_of_vertical_pixels)

    shift_horizontal_fit = np.zeros(no_of_vertical_pixels)

    weights = np.ones(no_of_vertical_pixels)

    display = copy.deepcopy(x_corrected_flat)

    plt.figure('Click on the line profile to trace')

    plt.imshow(display, cmap='gray', origin='lower')

    point = plt.ginput(1)

    plt.clf()

    plt.cla()

    point_as_a_list = list(map(int, point[0]))

    reference_row = int(point_as_a_list[1])

    x_beg = int(point_as_a_list[0] - extent / 2)

    x_end = int(point_as_a_list[0] + extent / 2)

    slit_reference = np.mean(
        display[
            reference_row - 10:reference_row + 10, x_beg:x_end
        ],
        axis=0
    )

    normalised_slit = (
        slit_reference - slit_reference.mean()
    ) / slit_reference.std()

    for j in vertical_indices:
        this_slit = display[j, x_beg - max_shift:x_end + max_shift]

        weights[j] = np.sqrt(this_slit.mean()**2)

        this_slit_normalised = (this_slit - this_slit.mean()) / this_slit.std()

        correlation = np.correlate(
            scipy.ndimage.zoom(this_slit_normalised, up_sampling),
            scipy.ndimage.zoom(normalised_slit, up_sampling),
            mode='valid'
        )

        shift_horizontal[j] = np.argmax(correlation)

    shift_horizontal = shift_horizontal / up_sampling - max_shift

    valid_x_points = np.argwhere(np.abs(shift_horizontal) < max_shift)

    shifts_for_valid_points = shift_horizontal[valid_x_points]

    c = np.polyfit(
        valid_x_points.ravel(),
        shifts_for_valid_points.ravel(),
        2,
        w=np.nan_to_num(weights)[valid_x_points].ravel()
    )

    shift_horizontal_fit = c[0] * vertical_indices**2 + \
        c[1] * vertical_indices + c[2]

    shift_horizontal_apply = -shift_horizontal_fit

    plt.plot(
        valid_x_points,
        shifts_for_valid_points,
        'k-',
        shift_horizontal_fit,
        'k-'
    )

    plt.show()

    return shift_horizontal_apply


def get_x_shift(dark_corrected_flat):
    max_shift, extent, up_sampling = 10, 20, 10

    no_of_horizontal_pixels = dark_corrected_flat.shape[1]

    horizontal_indices = np.arange(no_of_horizontal_pixels)

    shift_vertical = np.zeros(no_of_horizontal_pixels)

    shift_vertical_fit = np.zeros(no_of_horizontal_pixels)

    weights = np.ones(no_of_horizontal_pixels)

    display = copy.deepcopy(dark_corrected_flat)

    plt.figure('Click on the slit profile to trace')

    plt.imshow(display, cmap='gray', origin='lower')

    point = plt.ginput(1)

    plt.clf()

    plt.cla()

    point_as_a_list = list(map(int, point[0]))

    reference_column = int(point_as_a_list[0])

    y_beg = int(point_as_a_list[1] - extent / 2)

    y_end = int(point_as_a_list[1] + extent / 2)

    slit_reference = display[y_beg:y_end, reference_column]

    normalised_slit = (
        slit_reference - slit_reference.mean()
    ) / slit_reference.std()

    for j in horizontal_indices:
        this_slit = display[y_beg - max_shift:y_end + max_shift, j]

        weights[j] = np.sqrt(this_slit.mean()**2)

        this_slit_normalised = (this_slit - this_slit.mean()) / this_slit.std()

        correlation = np.correlate(
            scipy.ndimage.zoom(this_slit_normalised, up_sampling),
            scipy.ndimage.zoom(normalised_slit, up_sampling),
            mode='valid'
        )

        shift_vertical[j] = np.argmax(correlation)

    shift_vertical = shift_vertical / up_sampling - max_shift

    valid_x_points = np.argwhere(np.abs(shift_vertical) < max_shift)

    shifts_for_valid_points = shift_vertical[valid_x_points]

    c = np.polyfit(
        valid_x_points.ravel(),
        shifts_for_valid_points.ravel(),
        1,
        w=np.nan_to_num(weights)[valid_x_points].ravel()
    )

    shift_vertical_fit = c[0] * horizontal_indices + c[1]

    shift_vertical_apply = -shift_vertical_fit

    plt.plot(
        valid_x_points,
        shifts_for_valid_points,
        'k-',
        shift_vertical_fit,
        'k-'
    )

    plt.show()

    return shift_vertical_apply


def apply_x_shift(dark_corrected_flat, shifts):
    result = np.zeros_like(dark_corrected_flat)

    for i in np.arange(dark_corrected_flat.shape[1]):
        scipy.ndimage.shift(
            dark_corrected_flat[:, i],
            shifts[i],
            result[:, i],
            mode='nearest'
        )

    # plt.imshow(result, cmap='gray', origin='lower')

    # plt.show()

    return result


def apply_y_shift(dark_corrected_flat, shifts):
    result = np.zeros_like(dark_corrected_flat)

    for i in np.arange(dark_corrected_flat.shape[0]):
        scipy.ndimage.shift(
            dark_corrected_flat[i, :],
            shifts[i],
            result[i, :],
            mode='nearest'
        )

    # plt.imshow(result, cmap='gray', origin='lower')

    # plt.show()

    return result


def remove_line_profile(inclination_corrected_flat):
    rows_1 = np.arange(100, 400)
    rows_2 = np.arange(600, 900)
    profile = np.append(
        inclination_corrected_flat[rows_1], inclination_corrected_flat[rows_2],
        axis=0
    )
    line_median = np.median(profile, 0)
    normalised_median = line_median / line_median.max()
    filtered_line = scipy.ndimage.gaussian_filter1d(normalised_median, 2)
    result = np.divide(inclination_corrected_flat, filtered_line)

    return result, normalised_median


def get_master_flat_x_y_inclinations_and_line_profile(
    flat_filename, dark_master, fringe_master
):
    flat_data, flat_header = sunpy.io.fits.read(flat_filename)[0]
    dark_data, dark_header = sunpy.io.fits.read(dark_master)[0]
    fringe_data, fringe_header = sunpy.io.fits.read(fringe_master)[0]

    mean_flat = np.mean(flat_data, axis=0)

    dark_corrected_flat = mean_flat - dark_data

    shift_vertical_apply = get_x_shift(dark_corrected_flat)

    x_corrected_flat = apply_x_shift(dark_corrected_flat, shift_vertical_apply)

    x_corrected_fringe = apply_x_shift(fringe_data, shift_vertical_apply)

    shift_horizontal_apply = get_y_shift(x_corrected_flat)

    y_corrected_flat = apply_y_shift(
        x_corrected_flat, shift_horizontal_apply
    )

    y_corrected_fringe = apply_y_shift(
        x_corrected_fringe, shift_horizontal_apply
    )

    _, line_median = remove_line_profile(y_corrected_flat / y_corrected_fringe)

    flat_master = y_corrected_flat / line_median

    flat_master_name = flat_filename.name.split('.')[-2] + 'FLATMASTER.fits'

    write_path_flat_master = write_path / flat_master_name

    sunpy.io.fits.write(
        write_path_flat_master,
        flat_master,
        flat_header,
        overwrite=True
    )

    np.savetxt(write_path / 'x_inclinations.txt', shift_vertical_apply)

    np.savetxt(write_path / 'y_inclinations.txt', shift_horizontal_apply)

    np.savetxt(write_path / 'flat_profile.txt', line_median)


if __name__ == '__main__':
    dark_filename = Path(
        '/Users/harshmathur/Documents/CourseworkRepo' +
        '/Kodai Visit/20200204/102402_DARK.fits'
    )

    flat_filename = Path(
        '/Volumes/Harsh 9599771751/Kodai Visit ' +
        '31 Jan - 12 Feb/20200204/Flats/114059_FLAT.fits'
    )

    fringe_filename = Path(
        '/Volumes/Harsh 9599771751/Kodai Visit ' +
        '31 Jan - 12 Feb/20200207/Flats/082259_FLAT.fits'
    )

    fringe_master = Path(
        '/Users/harshmathur/Documents/CourseworkRepo' +
        '/Kodai Visit/20200204/103556_FLATFRINGEFLAT.fits'
    )

    dark_master = Path(
        '/Users/harshmathur/Documents/CourseworkRepo' +
        '/Kodai Visit/20200207/102402_DARK.fits'
    )
