import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import tqdm
import lightfieldpackage
import os
from shutil import copyfile

# Set the colormap for image visualizations
colormap = 'gray'

# Import settings module for configuration parameters
import settings

# Define the results directory to store output images and data
results_directory = r"results/LF_Deconv_USAF"

# Create the results directory if it doesn't already exist
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# Save a copy of the current script and settings file in the results directory for reproducibility
copyfile(os.path.abspath(__file__), os.path.join(results_directory, os.path.basename(__file__)))
copyfile(settings.params_path, os.path.join(results_directory, os.path.basename(settings.params_path)))

# Directory containing input irradiance images
input_directory = r"data/USAF_lf"
input_file_paths = sorted(glob.glob(os.path.join(input_directory, "irradiance_*.npy")))

# Load irradiance images into a lightfield array
lightfield_array = []
for file_path in input_file_paths:
    image = np.load(file_path)  # Load each irradiance image
    lightfield_array.append(image)
lightfield_array = np.array(lightfield_array)  # Convert to a NumPy array

# Load the calibration image
calibration_file_path = os.path.join(input_directory, r"irradiance_calibration.npy")
calibration_image = np.load(calibration_file_path)

# Display the calibration image
plt.figure(), plt.imshow(calibration_image)

# Resize the calibration image using a predefined scaling factor
scaling_factor = 31 / 33
resized_calibration_image = cv2.resize(calibration_image, None, fx=scaling_factor, fy=scaling_factor)

# Display the resized calibration image
plt.figure(), plt.imshow(resized_calibration_image)

# Pad the resized calibration image to match the original dimensions
padded_calibration_image = np.zeros_like(calibration_image)
padded_calibration_image[(padded_calibration_image.shape[0] - resized_calibration_image.shape[0]) // 2:
                         (padded_calibration_image.shape[0] - resized_calibration_image.shape[0]) // 2 +
                         resized_calibration_image.shape[0],
                         (padded_calibration_image.shape[1] - resized_calibration_image.shape[1]) // 2:
                         (padded_calibration_image.shape[1] - resized_calibration_image.shape[1]) // 2 +
                         resized_calibration_image.shape[1]] = resized_calibration_image
plt.figure(), plt.imshow(padded_calibration_image)

# Extract parameters from settings
distance_micro_lens_mla = settings.B_ML_MLA * 1e-3
diameter_main_lens = settings.D_ML * 1e-3
focal_length_micro_lens = settings.f_ML * 1e-3
g_main_lens = settings.g_ML * 1e-3
g_main_lens_focus = settings.g_ML_focus * 1e-3
focal_length_mla = settings.f_MLA * 1e-3
distance_mla_sensor = settings.f_MLA * 1e-3
refractive_index = 1.0
wavelength = 520 * 1e-9
wave_number = 2 * np.pi / wavelength
pixel_pitch = settings.p * 1e-3
oversampling_rate = 3
num_micro_pixels = settings.N_MP_int
scaling_factor_per_pixel = pixel_pitch / oversampling_rate

# Load point light source positions
plqs_positions = settings.plqs_pos
object_coordinates = plqs_positions[:, :, :, :] * 1e-3

# Extract point light source coordinates
plq_coordinates = np.array([[plq.sourceX, plq.sourceY] for plq in settings.plqs])
x1objspace = np.unique(plq_coordinates[:, 0]) * 1e-3
x2objspace = np.unique(plq_coordinates[:, 1]) * 1e-3
x3objspace = g_main_lens - g_main_lens_focus
object_space = np.ones((object_coordinates.shape[0], object_coordinates.shape[1], len(x3objspace)))

# Calculate the PSF (Point Spread Function) size at micro-lens
psf_size_micro_lens_mla = diameter_main_lens * np.abs(
    1 / (1 / focal_length_micro_lens - 1 / g_main_lens) - distance_micro_lens_mla) / (
                                      1 / (1 / focal_length_micro_lens - 1 / g_main_lens))
num_micro_lens_rois = psf_size_micro_lens_mla / (settings.D_MiL * 1e-3)
num_micro_lens_roi = int(np.ceil(np.max(num_micro_lens_rois)) // 2 * 2 + 1) if np.mod(settings.N_MLA, 2) else int(
    np.ceil(np.max(num_micro_lens_rois)))

# Print number of micro-lens regions of interest
print("N_MiL_RoI is ", num_micro_lens_roi)

# Define image dimensions and spatial grids
image_width = num_micro_lens_roi * num_micro_pixels
image_height = num_micro_lens_roi * num_micro_pixels
x1space = np.linspace(start=-image_width * pixel_pitch / 2 + pixel_pitch / oversampling_rate / 2,
                      stop=image_width * pixel_pitch / 2 - pixel_pitch / oversampling_rate / 2,
                      num=num_micro_pixels * num_micro_lens_roi * oversampling_rate)
x2space = np.linspace(start=-image_width * pixel_pitch / 2 + pixel_pitch / oversampling_rate / 2,
                      stop=image_width * pixel_pitch / 2 - pixel_pitch / oversampling_rate / 2,
                      num=num_micro_pixels * num_micro_lens_roi * oversampling_rate)
assert len(x2space) == num_micro_pixels * num_micro_lens_roi * oversampling_rate

# Calculate magnification and shifts for object space
magnification_factor = distance_micro_lens_mla / g_main_lens[0]
shift_offset_x1_top = np.min(x1objspace) * magnification_factor
shift_offset_x1_bottom = np.max(x1objspace) * magnification_factor
shift_offset_x2_left = np.min(x2objspace) * magnification_factor
shift_offset_x2_right = np.max(x2objspace) * magnification_factor

# Define spatial grids for shift calculations
x_space_for_shift = np.arange(start=-image_width * pixel_pitch / 2 + pixel_pitch / oversampling_rate / 2 + shift_offset_x1_top,
                              stop=image_width * pixel_pitch / 2 - pixel_pitch / oversampling_rate / 2 + shift_offset_x1_bottom + pixel_pitch / oversampling_rate,
                              step=pixel_pitch / oversampling_rate)
y_space_for_shift = np.arange(start=-image_width * pixel_pitch / 2 + pixel_pitch / oversampling_rate / 2 + shift_offset_x2_left,
                              stop=image_width * pixel_pitch / 2 - pixel_pitch / oversampling_rate / 2 + shift_offset_x2_right + pixel_pitch / oversampling_rate,
                              step=pixel_pitch / oversampling_rate)

# Micro-lens array grid for lightfield processing
x1ml_space = np.linspace(start=-num_micro_pixels * pixel_pitch / 2 + pixel_pitch / oversampling_rate / 2,
                         stop=num_micro_pixels * pixel_pitch / 2 - pixel_pitch / oversampling_rate / 2,
                         endpoint=True, num=oversampling_rate * num_micro_pixels)
x2ml_space = np.linspace(start=-num_micro_pixels * pixel_pitch / 2 + pixel_pitch / oversampling_rate / 2,
                         stop=num_micro_pixels * pixel_pitch / 2 - pixel_pitch / oversampling_rate / 2,
                         endpoint=True, num=oversampling_rate * num_micro_pixels)

# Calculate micro-lens array pattern
micro_lens_array = lightfieldpackage.utils_optics.calculate_mla_pattern(
    focal_length_mla, wave_number, x1ml_space, x2ml_space, num_lenses_x=num_micro_lens_roi, num_lenses_y=num_micro_lens_roi)

# Extract super-resolution factor
super_res_factor = settings.super_resolution_factor

# Process each lightfield image
for idx in tqdm.tqdm(range(object_coordinates.shape[2])):
    # Extract object coordinates for current depth
    current_object_coordinates = object_coordinates[:, :, [idx], :]
    current_object_coordinates_quarter = current_object_coordinates[
                                         :current_object_coordinates.shape[0] // 2 + 1,
                                         :current_object_coordinates.shape[1] // 2 + 1, :, :]

    # Generate wave propagation kernel for the current depth
    H_wave_quarter, psf_before_MLA_character = lightfieldpackage.utils_optics.generate_wave(image_height, image_width,
                                                                                          current_object_coordinates_quarter,
                                                                                          focal_length_micro_lens,
                                                                                          x1space, x2space,
                                                                                          x_shift_grid=x_space_for_shift,
                                                                                          y_shift_grid=y_space_for_shift,
                                                                                          scaling_factor=scaling_factor_per_pixel,
                                                                                          wavelength=wavelength,
                                                                                          microlens_array=micro_lens_array,
                                                                                          microlens_to_sensor_distance=distance_mla_sensor,
                                                                                          microlens_to_ml_distance=distance_micro_lens_mla,
                                                                                          mainlens_diameter=diameter_main_lens,
                                                                                          refractive_index=refractive_index,
                                                                                          oversample_x=oversampling_rate,
                                                                                          oversample_y=oversampling_rate,
                                                                                          ml_focus_position=g_main_lens_focus)

    # Reconstruct the full wave kernel
    H_wave = np.zeros((image_width, image_height, super_res_factor, super_res_factor, 1))
    a = current_object_coordinates.shape[0] // 2 + 1
    b = (current_object_coordinates.shape[0] // 2 - 1) if (current_object_coordinates.shape[0] // 2 - 1) > 0 else 0
    H_wave[:, :, :a, :a, :] = H_wave_quarter
    H_wave[:, :, a:, :a, :] = H_wave_quarter[::-1, :, b::-1, :, :]
    H_wave[:, :, a:, a:, :] = H_wave_quarter[::-1, ::-1, b::-1, b::-1, :]
    H_wave[:, :, :a, a:, :] = H_wave_quarter[:, ::-1, :, b::-1, :]
    H_wave_multiple_depth = np.flip(H_wave, axis=(0, 1))
    super_res_factor_wave_x = H_wave_multiple_depth.shape[3]
    super_res_factor_wave_y = H_wave_multiple_depth.shape[2]
    H_wave_single_depth = H_wave_multiple_depth[:, :, :, :, 0]

    # Define forward and backward functions for lightfield deconvolution
    forward_function = lambda object_field: lightfieldpackage.utils_deconv.forward_prevedel_super_resolution(
        object_field, H_wave_single_depth, super_res_factor_wave_x, super_res_factor_wave_y, microlens_pitch_x=num_micro_pixels,
        microlens_pitch_y=num_micro_pixels)

    backward_function = lambda projection: lightfieldpackage.utils_deconv.backward_prevedel_super_resolution(
        projection, np.flip(H_wave_single_depth, (0, 1)), super_res_factor_x=super_res_factor_wave_x,
        super_res_factor_y=super_res_factor_wave_y, microlens_pitch_x=num_micro_pixels, microlens_pitch_y=num_micro_pixels)

    # Load the irradiance image for the current depth
    img_usaf = np.load(input_file_paths[idx])
    resized_image = cv2.resize(img_usaf, None, fx=scaling_factor, fy=scaling_factor)
    zero_padded_image = np.zeros_like(img_usaf)
    zero_padded_image[(zero_padded_image.shape[0] - resized_image.shape[0]) // 2:
                      (zero_padded_image.shape[0] - resized_image.shape[0]) // 2 + resized_image.shape[0],
                      (zero_padded_image.shape[1] - resized_image.shape[1]) // 2:
                      (zero_padded_image.shape[1] - resized_image.shape[1]) // 2 + resized_image.shape[1]] = resized_image

    # Perform iterative Richardson-Lucy deconvolution
    iterations = 3
    LF_image = zero_padded_image
    reconstructed_wave = np.ones([zero_padded_image.shape[1] // num_micro_pixels * super_res_factor_wave_x,
                                  zero_padded_image.shape[1] // num_micro_pixels * super_res_factor_wave_x])

    for i in range(iterations):
        reconstructed_wave = lightfieldpackage.utils_deconv.deconvRL(
            forward_function, backward_function, LF_image, 1, reconstructed_wave)

        # Save intermediate reconstructed images for each iteration
        plt.imsave(os.path.join(results_directory,
                                f"Reconstructed_{idx:04d}_iter_{i:02d}.png"), reconstructed_wave, cmap=colormap)

    # Save rectified raw image
    plt.imsave(os.path.join(results_directory, f"Raw_Rectified_{idx:04d}.png"), zero_padded_image, cmap=colormap)
