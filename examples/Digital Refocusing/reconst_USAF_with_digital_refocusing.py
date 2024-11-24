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

colormap = 'gray'

import settings

results_directory = r"../../results/digital_refocusing_USAF"

if not os.path.exists(results_directory):
    os.makedirs(results_directory)
copyfile(os.path.basename(__file__), os.path.join(results_directory, os.path.basename(__file__)))
copyfile(settings.params_path, os.path.join(results_directory, os.path.basename(settings.params_path)))

input_directory = r"..\..\data\usaf_lf"
input_file_paths = sorted(glob.glob(os.path.join(input_directory, "irradiance_*.npy")))

lightfield_array = []
for file_path in input_file_paths:
    image = np.load(file_path)
    lightfield_array.append(image)
lightfield_array = np.array(lightfield_array)

calibration_file_path = os.path.join(input_directory, r"irradiance_calibration.npy")
calibration_image = np.load(calibration_file_path)
plt.figure(), plt.imshow(calibration_image)
scaling_factor = 31 / 33
resized_calibration_image = cv2.resize(calibration_image, None, fx=scaling_factor, fy=scaling_factor)
plt.figure(), plt.imshow(resized_calibration_image)
padded_calibration_image = np.zeros_like(calibration_image)
padded_calibration_image[(padded_calibration_image.shape[0] - resized_calibration_image.shape[0]) // 2:
                         (padded_calibration_image.shape[0] - resized_calibration_image.shape[0]) // 2 +
                         resized_calibration_image.shape[0],
                         (padded_calibration_image.shape[1] - resized_calibration_image.shape[1]) // 2:
                         (padded_calibration_image.shape[1] - resized_calibration_image.shape[1]) // 2 +
                         resized_calibration_image.shape[1]] = resized_calibration_image
plt.figure(), plt.imshow(padded_calibration_image)

distance_micro_lens_mla = settings.B_ML_MLA * 1e-3
diameter_micro_lens = settings.D_ML * 1e-3
distance_sensor = settings.D_S * 1e-3
focal_length_micro_lens = settings.f_ML * 1e-3
g_main_lens = settings.g_ML * 1e-3
g_main_lens_focus = settings.g_ML_focus * 1e-3
focal_length_mla = settings.f_MLA * 1e-3
distance_mla_sensor = settings.f_MLA * 1e-3
base_micro_lens = settings.b_ML * 1e-3
refractive_index = 1.0
wavelength = 520 * 1e-9
wave_number = 2 * np.pi / wavelength
pixel_pitch = settings.p * 1e-3
oversampling_rate = 3
num_micro_pixels = settings.N_MP_int
num_mlas = settings.N_MLA
scaling_factor_per_pixel = pixel_pitch / oversampling_rate

lightfield_positions = settings.plqs_pos
object_coordinates = lightfield_positions[:, :, :, :] * 1e-3

psf_size_micro_lens_mla = diameter_micro_lens * np.abs(
    1 / (1 / focal_length_micro_lens - 1 / g_main_lens) - distance_micro_lens_mla) / (
                                  1 / (1 / focal_length_micro_lens - 1 / g_main_lens))
magnification_factor = distance_micro_lens_mla / g_main_lens[0]

super_res_factor = settings.super_resolution_factor
input_file_paths = sorted(glob.glob(os.path.join(input_directory, "irradiance_*.npy")))
forward_projections = []
reconstructed_images = []
lightfield = lightfieldpackage.utils.LightfieldProcessor(is_1D=False)

for idx in tqdm.tqdm(range(object_coordinates.shape[2])):
    current_object_coordinates = object_coordinates[:, :, [idx], :]
    current_object_coordinates_quarter = current_object_coordinates[:current_object_coordinates.shape[0] // 2 + 1,
                                         :current_object_coordinates.shape[1] // 2 + 1, :, :]
    focal_distance_current_micro_lens = base_micro_lens[idx]
    focal_distance_mla = distance_micro_lens_mla
    scaling_alpha = focal_distance_current_micro_lens / focal_distance_mla
    current_image = np.load(input_file_paths[idx])
    resized_image = cv2.resize(current_image, None, fx=scaling_factor, fy=scaling_factor)
    zero_padded_image = np.zeros_like(current_image)
    zero_padded_image[(zero_padded_image.shape[0] - resized_image.shape[0]) // 2:
                      (zero_padded_image.shape[0] - resized_image.shape[0]) // 2 + resized_image.shape[0],
                      (zero_padded_image.shape[1] - resized_image.shape[1]) // 2:
                      (zero_padded_image.shape[1] - resized_image.shape[1]) // 2 + resized_image.shape[1]] = resized_image
    lightfield_4D = lightfield.raw_to_lf(zero_padded_image, num_micro_pixels)
    lightfield_4D = lightfield_4D.astype(np.float32)

    backward_projection = lambda projection: lightfield.refocus_2D(projection[::-1, ::-1, :, :], scaling_alpha,
                                                                    coord_u=num_micro_pixels, coord_v=num_micro_pixels,
                                                                    coord_s=num_mlas, coord_t=num_mlas, downsample_factor=1,
                                                                    lenslet_pitch=distance_sensor, mainlens_pitch=diameter_micro_lens,
                                                                    apply_scaling=False, interpolation_method="linear")
    reconstructed_image = backward_projection(lightfield_4D)
    plt.imsave(os.path.join(results_directory,
                            "Reconstructed_Zemax_Digital_refocusing_" + str(idx).zfill(4) + ".png"), reconstructed_image,
               cmap=colormap)
    plt.imsave(os.path.join(results_directory,
                            "" + str(idx).zfill(4) + ".png"), zero_padded_image, cmap=colormap)

forward_projections = np.array(forward_projections)
reconstructed_images = np.array(reconstructed_images)
