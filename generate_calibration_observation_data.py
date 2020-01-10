from pathlib import Path
import numpy as np
import sunpy.io.fits
from dark_master_generate import apply_x_shift, apply_y_shift


write_path = Path('/Users/harshmathur/CourseworkRepo/Level-1')


def get_keys_from_name(filepath):
    name = filepath.name
    name_array = name.split('_')
    primary = int(name_array[2])
    sec_array = name_array[-1].split('.')
    secondary = int(sec_array[0])
    return [primary, secondary]


def get_calib_key_from_name(filepath):
    keys = get_keys_from_name(filepath)
    return keys[1], keys[0]


def get_observation_key_from_name(filepath):
    keys = get_keys_from_name(filepath)
    return keys[0], keys[1]


def get_flat_corrected_data(
    data_files,
    x_inclination_file,
    y_inclination_file,
    flat_master,
    dark_master
):
    temp_data, first_header = sunpy.io.fits.read(data_files[0])[0]
    if len(temp_data.shape) == 3:
        resultant = np.zeros(
            shape=(
                len(data_files),
                temp_data.shape[1],
                temp_data.shape[2]
            )
        )
    else:
        resultant = np.zeros(
            shape=(
                len(data_files),
                temp_data.shape[0],
                temp_data.shape[1]
            )
        )
    dark_data, dark_header = sunpy.io.fits.read(dark_master)[0]
    flat_data, flat_header = sunpy.io.fits.read(flat_master)[0]
    shift_vertical_apply = np.loadtxt(x_inclination_file)
    shift_horizontal_apply = np.loadtxt(y_inclination_file)

    for index, data_file in enumerate(data_files):
        data, header = sunpy.io.fits.read(data_file)[0]

        if len(data.shape) == 3:
            data = data[0]

        data = data - dark_data

        data[np.where(data < 0)] = 0

        x_corrected_data = apply_x_shift(data, shift_vertical_apply)

        y_corrected_data = apply_y_shift(
            x_corrected_data, shift_horizontal_apply
        )

        flat_corrected_data = y_corrected_data / flat_data

        resultant[index] = flat_corrected_data

    return resultant, first_header


def save_calibration_data(
    calibration_folder,
    x_inclination_file,
    y_inclination_file,
    flat_master,
    dark_master
):
    everything = calibration_folder.glob('**/*')

    calibration_files = [
        x for x in everything if x.is_file() and
        x.name.endswith('.fits')
    ]

    calibration_files.sort(key=get_calib_key_from_name)

    resultant, header = get_flat_corrected_data(
        data_files=calibration_files,
        x_inclination_file=x_inclination_file,
        y_inclination_file=y_inclination_file,
        flat_master=flat_master,
        dark_master=dark_master
    )

    sunpy.io.fits.write(
        write_path / 'calib_data.fits',
        resultant,
        header,
        overwrite=True
    )


def save_observation_data(
    observation_folder,
    x_inclination_file,
    y_inclination_file,
    flat_master,
    dark_master
):
    everything = observation_folder.glob('**/*')

    observation_files = [
        x for x in everything if x.is_file() and
        x.name.endswith('.fits')
    ]

    observation_files.sort(key=get_observation_key_from_name)

    resultant, header = get_flat_corrected_data(
        data_files=observation_files,
        x_inclination_file=x_inclination_file,
        y_inclination_file=y_inclination_file,
        flat_master=flat_master,
        dark_master=dark_master
    )

    sunpy.io.fits.write(
        write_path / 'observation_data.fits',
        resultant,
        header,
        overwrite=True
    )


if __name__ == '__main__':
    calibration_folder = Path(
        '/Volumes/Harsh 9599771751/Spectropolarimetric' +
        ' Data Kodaikanal/2019/20190413/CalibrationAlt/093027'
    )

    observation_folder = Path(
        '/Volumes/Harsh 9599771751/Spectropolarimetric' +
        ' Data Kodaikanal/2019/20190413/Observation/080619'
    )

    x_inclination_file = Path(
        '/Users/harshmathur/Documents/' +
        'CourseworkRepo/Level-1/x_inclinations.txt'
    )

    y_inclination_file = Path(
        '/Users/harshmathur/Documents/' +
        'CourseworkRepo/Level-1/y_inclinations.txt'
    )

    flat_master = Path(
        '/Users/harshmathur/Documents/CourseworkRepo' +
        '/Level-1/083523_FLATFLATMASTER.fits'
    )

    dark_master = Path(
        '/Users/harshmathur/Documents/CourseworkRepo' +
        '/Level-1/083651_DARKMASTER.fits'
    )

    save_calibration_data(
        calibration_folder,
        x_inclination_file,
        y_inclination_file,
        flat_master,
        dark_master
    )

    save_observation_data(
        observation_folder,
        x_inclination_file,
        y_inclination_file,
        flat_master,
        dark_master
    )
