import numpy as np
import scipy.special as special
import scipy.integrate as integrate
import tqdm

def calculate_mla_pattern(focal_length_ml, wave_number, grid_x, grid_y, num_lenses_x, num_lenses_y,
                          use_round_lenses=False):
    """
    Calculates the microlens array (MLA) diffraction pattern.

    Parameters:
    - focal_length_ml: Focal length of the microlenses.
    - wave_number: Wave number (k).
    - grid_x, grid_y: Grids for the microlenses (x and y coordinates).
    - num_lenses_x, num_lenses_y: Number of microlenses along x and y axes.
    - use_round_lenses: Boolean indicating whether to use round microlenses.

    Returns:
    - mla_array: Diffraction pattern of the microlens array.
    """
    x_grid, y_grid = np.meshgrid(grid_x, grid_y)
    lens_norm = x_grid ** 2 + y_grid ** 2
    lens_pattern = np.exp(-1j * wave_number / (2 * focal_length_ml) * lens_norm)

    if use_round_lenses:
        lens_radius = min(np.max(grid_x), np.max(grid_y)) / 2
        lens_mask = lens_norm <= lens_radius ** 2
        lens_pattern *= lens_mask

    mla_array = np.tile(lens_pattern, [num_lenses_y, num_lenses_x])
    return mla_array


def fresnel_propagation_2d(input_wave, grid_spacing, propagation_distance, wavelength):
    """
    Performs 2D Fresnel propagation of a wave.

    Parameters:
    - input_wave: Input wave.
    - grid_spacing: Spacing between grid points.
    - propagation_distance: Propagation distance.
    - wavelength: Wavelength of light.

    Returns:
    - propagated_wave: Propagated wave.
    - new_grid_spacing: New grid spacing.
    - grid_coordinates: New grid coordinates.
    """
    num_points_x = input_wave.shape[1]
    num_points_y = input_wave.shape[0]
    wave_number = 2 * np.pi / wavelength

    freq_x_spacing = 1. / (num_points_x * grid_spacing)
    freq_x = np.concatenate((np.arange(0, np.ceil(num_points_x / 2)),
                             np.arange(np.ceil(-num_points_x / 2), 0))) * freq_x_spacing
    freq_y_spacing = 1. / (num_points_y * grid_spacing)
    freq_y = np.concatenate((np.arange(0, np.ceil(num_points_y / 2)),
                             np.arange(np.ceil(-num_points_y / 2), 0))) * freq_y_spacing
    freq_x_grid, freq_y_grid = np.meshgrid(freq_x, freq_y)

    fresnel_transfer_function = np.exp(
        -1j * 2 * np.pi ** 2 * (freq_x_grid ** 2 + freq_y_grid ** 2) * propagation_distance / wave_number)

    propagated_wave = np.exp(1j * wave_number * propagation_distance) * np.fft.ifft2(
        np.fft.fft2(input_wave) * fresnel_transfer_function)

    new_grid_spacing = grid_spacing
    grid_coordinates = np.arange(-num_points_x / 2, num_points_x / 2-1) * new_grid_spacing

    return propagated_wave, new_grid_spacing, grid_coordinates


def compute_point_spread_function_debye(obj_x, obj_y, obj_z, na, grid_x, grid_y, wavelength, mag, focus_mag, refr_index):
    """
    Calculates the point spread function (PSF) based on the Debye theory.

    Parameters:
    - obj_x, obj_y, obj_z: Coordinates of the point source in object space.
    - na: Numerical aperture of the optical system.
    - grid_x, grid_y: Coordinate grids in image space.
    - wavelength: Wavelength of light.
    - mag: Magnification of the optical system.
    - focus_mag: Magnification in focus.
    - refr_index: Refractive index of the medium.

    Returns:
    - psf_result: Complex point spread function (PSF).
    """
    k_wave = 2 * np.pi * refr_index / wavelength
    half_angle = np.arcsin(na / refr_index)
    coord_grid_x, coord_grid_y = np.meshgrid(grid_x, grid_y)

    lateral_dist = (
        ((coord_grid_x + focus_mag * obj_x) ** 2 + (coord_grid_y + focus_mag * obj_y) ** 2) ** 0.5
    ) / mag

    radial_dist = k_wave * lateral_dist * np.sin(half_angle)
    axial_dist = 4 * k_wave * obj_z * (np.sin(half_angle / 2) ** 2)

    theta_values = np.linspace(0, half_angle, 50)
    integrand_vals = integrand(
        theta=theta_values,
        axial_distance=axial_dist,
        radial_distance=np.repeat(np.expand_dims(radial_dist, axis=-1), 50, axis=-1),
        aperture_angle=half_angle
    )

    system_const = (
        (mag * refr_index ** 2) / (na ** 2 * wavelength ** 2)
        * np.exp(-1j * axial_dist / (4 * (np.sin(half_angle / 2) ** 2)))
    )

    integral_res = integrate.trapezoid(integrand_vals, dx=half_angle / 50, axis=-1)
    psf_result = system_const * integral_res

    return psf_result


def integrand(theta, axial_distance, radial_distance, aperture_angle):
    """
    Calculates the integrand for the Debye theory-based PSF calculation.

    Parameters:
    - theta: Angle describing scattering behavior (in radians).
    - radial_distance: Radial distance (v).
    - axial_distance: Axial distance (u).
    - aperture_angle: Half opening angle of the numerical aperture (alpha).

    Returns:
    - result: Value of the integrand function.
    """
    result = (
        np.sqrt(np.cos(theta))
        * (1 + np.cos(theta))
        * np.exp(-(1j * axial_distance / 2) * (np.sin(theta / 2) ** 2) / (np.sin(aperture_angle / 2) ** 2))
        * special.j0(np.sin(theta) / np.sin(aperture_angle) * radial_distance)
        * np.sin(theta)
    )
    return result



def generate_wave(image_height, image_width, object_coords_xyz, focal_length_ml, x_grid, y_grid, x_shift_grid, y_shift_grid, scaling_factor, wavelength, microlens_array, microlens_to_sensor_distance, microlens_to_ml_distance, mainlens_diameter, refractive_index, oversample_x, oversample_y, ml_focus_position):
    """
    Generates wavefront and PSFs before the microlens array.

    Parameters:
    - image_height: Height of the resulting image.
    - image_width: Width of the resulting image.
    - object_coords_xyz: 3D coordinates of the objects.
    - focal_length_ml: Focal length of the microlenses.
    - x_grid, y_grid: Grids for x and y axes.
    - x_shift_grid, y_shift_grid: Shift grids for x and y axes.
    - scaling_factor: Scaling factor for the simulation.
    - wavelength: Wavelength of light.
    - microlens_array: Microlens array.
    - microlens_to_sensor_distance: Distance between microlens array and sensor.
    - microlens_to_ml_distance: Distance between microlenses and the main lens array.
    - mainlens_diameter: Diameter of the main lens.
    - refractive_index: Refractive index of the medium.
    - oversample_x, oversample_y: Oversampling factors for x and y axes.
    - ml_focus_position: Focus position of the microlenses.

    Returns:
    - wavefront: Simulated wavefront.
    - psfs_before_mla: Point spread functions (PSFs) before the microlens array.
    """
    wavefront = np.zeros((image_height, image_width, object_coords_xyz.shape[0], object_coords_xyz.shape[1], object_coords_xyz.shape[2]), dtype=np.float32)
    psfs_before_mla = np.zeros((y_shift_grid.shape[0], x_shift_grid.shape[0], object_coords_xyz.shape[2]), dtype=np.complex128)

    for ml_idx, ml_position in enumerate(tqdm.tqdm(object_coords_xyz[0, 0, :, 2])):
        z_position = ml_position - ml_focus_position
        theta = np.arctan(mainlens_diameter / (2 * ml_position))
        numerical_aperture = refractive_index * np.sin(theta)
        magnification = microlens_to_ml_distance / ml_position
        focus_magnification = microlens_to_ml_distance / ml_focus_position

        psf_before_mla = compute_point_spread_function_debye(
            0, 0, z_position, numerical_aperture, x_shift_grid, y_shift_grid, wavelength, magnification, focus_magnification, refractive_index
        )
        psfs_before_mla[:, :, ml_idx] = psf_before_mla

        psf_height, psf_width = psf_before_mla.shape
        cropped_width, cropped_height = len(x_grid), len(y_grid)
        crop_left = (psf_width - cropped_width) // 2
        crop_top = (psf_height - cropped_height) // 2
        crop_right = (psf_width + cropped_width) // 2
        crop_bottom = (psf_height + cropped_height) // 2

        for x_idx, x_position in enumerate(tqdm.tqdm(object_coords_xyz[:, 0, 0, 0])):
            for y_idx, y_position in enumerate(object_coords_xyz[0, :, 0, 1]):
                x_shift = np.around(x_position / np.abs(x_grid[1] - x_grid[0]) * magnification).astype(int)

                if len(y_grid) > 1:
                    y_shift = np.around(y_position / np.abs(y_grid[1] - y_grid[0]) * magnification).astype(int)
                else:
                    y_shift = 0

                shifted_psf = np.roll(psf_before_mla, (-y_shift, -x_shift), axis=(0, 1))
                cropped_psf = shifted_psf[crop_top:crop_bottom, crop_left:crop_right]

                psf_after_mla = cropped_psf * microlens_array
                psf_sensor, _, _ = fresnel_propagation_2d(psf_after_mla, scaling_factor, microlens_to_sensor_distance, wavelength)

                intensity = np.abs(psf_sensor ** 2)
                intensity = intensity.reshape(image_height, oversample_y, image_width, oversample_x).mean(-1).mean(1)
                wavefront[:, :, x_idx, y_idx, ml_idx] = intensity

    wavefront = np.transpose(wavefront, axes=(0, 1, 3, 2, 4))
    wavefront /= wavefront.sum(axis=1, keepdims=True).sum(axis=0, keepdims=True)

    return wavefront, psfs_before_mla


def generate_wave_with_na(image_height, image_width, object_coords_xyz, x_grid, y_grid, x_shift_space, y_shift_space, scaling_factor, wavelength, microlens_array, microlens_to_sensor_distance, refractive_index, oversample_x, oversample_y, ml_focus_position, numerical_aperture, magnification):
    """
    Generates wavefront and PSFs before the microlens array with a numerical aperture.

    Parameters:
    (Same as the `generate_wave` function but with numerical aperture and magnification included.)

    Returns:
    - wavefront: Simulated wavefront.
    - psfs_before_mla: PSFs before the microlens array.
    """
    wavefront = np.zeros((image_height, image_width, object_coords_xyz.shape[0], object_coords_xyz.shape[1], object_coords_xyz.shape[2]), dtype=np.float32)
    psfs_before_mla = np.zeros((y_shift_space.shape[0], x_shift_space.shape[0], object_coords_xyz.shape[2]), dtype=np.complex128)

    for microlens_idx, microlens_position in enumerate(tqdm.tqdm(object_coords_xyz[0, 0, :, 2])):
        z_position = microlens_position - ml_focus_position

        psf_before_mla = compute_point_spread_function_debye(
            0, 0, z_position, numerical_aperture, x_shift_space, y_shift_space, wavelength, magnification, magnification, refractive_index
        )
        psfs_before_mla[:, :, microlens_idx] = psf_before_mla

        psf_height, psf_width = psf_before_mla.shape
        cropped_width, cropped_height = len(x_grid), len(y_grid)
        crop_left = (psf_width - cropped_width) // 2
        crop_top = (psf_height - cropped_height) // 2
        crop_right = (psf_width + cropped_width) // 2
        crop_bottom = (psf_height + cropped_height) // 2

        for x_idx, x_position in enumerate(tqdm.tqdm(object_coords_xyz[:, 0, 0, 0])):
            for y_idx, y_position in enumerate(object_coords_xyz[0, :, 0, 1]):
                x_shift = np.around(x_position / np.abs(x_grid[1] - x_grid[0]) * magnification).astype(int)
                if len(y_grid) > 1:
                    y_shift = np.around(y_position / np.abs(y_grid[1] - y_grid[0]) * magnification).astype(int)
                else:
                    y_shift = 0

                shifted_psf = np.roll(psf_before_mla, (-y_shift, -x_shift), axis=(0, 1))
                cropped_psf = shifted_psf[crop_top:crop_bottom, crop_left:crop_right]

                psf_after_mla = cropped_psf * microlens_array
                psf_sensor, _, _ = fresnel_propagation_2d(psf_after_mla, scaling_factor, microlens_to_sensor_distance, wavelength)

                intensity = np.abs(psf_sensor ** 2)
                intensity = intensity.reshape(image_height, oversample_y, image_width, oversample_x).mean(-1).mean(1)
                wavefront[:, :, x_idx, y_idx, microlens_idx] = intensity

    wavefront = np.transpose(wavefront, axes=(0, 1, 3, 2, 4))
    wavefront /= wavefront.sum(axis=1, keepdims=True).sum(axis=0, keepdims=True)

    return wavefront, psfs_before_mla
