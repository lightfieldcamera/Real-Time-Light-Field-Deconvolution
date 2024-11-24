import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import lightfieldpackage
import os
from shutil import copyfile

colormap = 'hot'  # gray

import settings

results_directory = r"../../results/USAF"
input_image_path = r"../../data/usaf.png"

if not os.path.exists(results_directory):
    os.makedirs(results_directory)
copyfile(os.path.basename(__file__), os.path.join(results_directory, os.path.basename(__file__)))
copyfile(settings.params_path, os.path.join(results_directory, os.path.basename(settings.params_path)))

image = cv2.imread(input_image_path, 0).astype(np.float32) / 255.0
plt.figure(), plt.imshow(image), plt.title("gt"), plt.colorbar()
resized_image = cv2.resize(image, dsize=(33 * 57, 33 * 57))

scaling_factor = ((settings.B_ML_Sensor - settings.f_MLA) / settings.g_ML_focus) / (settings.B_ML_Sensor / settings.g_ML_focus)
distance_micro_lens_mla = settings.B_ML_MLA * 1e-3
diameter_main_lens = settings.D_ML * 1e-3
focal_length_micro_lens = settings.f_ML * 1e-3
g_main_lens = settings.g_ML * 1e-3
g_main_lens_focus = settings.g_ML_focus * 1e-3
focal_length_mla = settings.f_MLA * 1e-3
distance_mla_sensor = settings.B_MLA_Sensor * 1e-3
refractive_index = 1.0
wavelength = 520 * 1e-9
wave_number = 2 * np.pi / wavelength
pixel_pitch = settings.p * 1e-3
oversampling_rate = 3
num_micro_pixels = settings.N_MP_int
scaling_factor_per_pixel = pixel_pitch / oversampling_rate

lightfield_positions = settings.plqs_pos
object_coordinates = lightfield_positions[:, :, :, :] * 1e-3

plq_coordinates = np.array([[plq.sourceX, plq.sourceY] for plq in settings.plqs])
x1objspace = np.unique(plq_coordinates[:, 0]) * 1e-3
x2objspace = np.unique(plq_coordinates[:, 1]) * 1e-3
x3objspace = g_main_lens - g_main_lens_focus
object_space = np.ones((object_coordinates.shape[0], object_coordinates.shape[1], len(x3objspace)))

psf_size_micro_lens_mla = diameter_main_lens * np.abs(
    1 / (1 / focal_length_micro_lens - 1 / g_main_lens) - distance_micro_lens_mla) / (
                                      1 / (1 / focal_length_micro_lens - 1 / g_main_lens))
num_micro_lens_rois = psf_size_micro_lens_mla / (settings.D_MiL * 1e-3)
num_micro_lens_roi = int(np.ceil(np.max(num_micro_lens_rois)) // 2 * 2 + 1) if np.mod(settings.N_MLA, 2) else int(
    np.ceil(np.max(num_micro_lens_rois)))

print("N_MiL_RoI is ", num_micro_lens_roi)
image_width = num_micro_lens_roi * num_micro_pixels
image_height = num_micro_lens_roi * num_micro_pixels
x1space = np.linspace(start=-image_width * pixel_pitch / 2 + pixel_pitch / oversampling_rate / 2,
                      stop=image_width * pixel_pitch / 2 - pixel_pitch / oversampling_rate / 2,
                      num=num_micro_pixels * num_micro_lens_roi * oversampling_rate)
x2space = np.linspace(start=-image_width * pixel_pitch / 2 + pixel_pitch / oversampling_rate / 2,
                      stop=image_width * pixel_pitch / 2 - pixel_pitch / oversampling_rate / 2,
                      num=num_micro_pixels * num_micro_lens_roi * oversampling_rate)
assert len(x2space) == num_micro_pixels * num_micro_lens_roi * oversampling_rate

magnification_factor = distance_micro_lens_mla / g_main_lens[0]
shift_offset_x1_top = np.min(x1objspace) * magnification_factor
shift_offset_x1_bottom = np.max(x1objspace) * magnification_factor
shift_offset_x2_left = np.min(x2objspace) * magnification_factor
shift_offset_x2_right = np.max(x2objspace) * magnification_factor
x_space_for_shift = np.arange(start=-image_width * pixel_pitch / 2 + pixel_pitch / oversampling_rate / 2 + shift_offset_x1_top,
                              stop=image_width * pixel_pitch / 2 - pixel_pitch / oversampling_rate / 2 + shift_offset_x1_bottom + pixel_pitch / oversampling_rate,
                              step=pixel_pitch / oversampling_rate)
y_space_for_shift = np.arange(start=-image_width * pixel_pitch / 2 + pixel_pitch / oversampling_rate / 2 + shift_offset_x2_left,
                              stop=image_width * pixel_pitch / 2 - pixel_pitch / oversampling_rate / 2 + shift_offset_x2_right + pixel_pitch / oversampling_rate,
                              step=pixel_pitch / oversampling_rate)

x1ml_space = np.linspace(start=-num_micro_pixels * pixel_pitch / 2 + pixel_pitch / oversampling_rate / 2,
                         stop=num_micro_pixels * pixel_pitch / 2 - pixel_pitch / oversampling_rate / 2,
                         endpoint=True, num=oversampling_rate * num_micro_pixels)
x2ml_space = np.linspace(start=-num_micro_pixels * pixel_pitch / 2 + pixel_pitch / oversampling_rate / 2,
                         stop=num_micro_pixels * pixel_pitch / 2 - pixel_pitch / oversampling_rate / 2,
                         endpoint=True, num=oversampling_rate * num_micro_pixels)

micro_lens_array = lightfieldpackage.utils_optics.calculate_mla_pattern(focal_length_mla, wave_number, x1ml_space, x2ml_space,
                                                          num_lenses_x=num_micro_lens_roi, num_lenses_y=num_micro_lens_roi)

super_res_factor = settings.super_resolution_factor

i_curr_obj_coords_xyz = 0
current_object_coordinates = object_coordinates[:, :, [i_curr_obj_coords_xyz], :]
current_object_coordinates_quarter = current_object_coordinates[
                                     :current_object_coordinates.shape[0] // 2 + 1,
                                     :current_object_coordinates.shape[1] // 2 + 1, :, :]

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
H_wave = np.zeros((image_width, image_height, super_res_factor, super_res_factor, 1))
a = current_object_coordinates.shape[0] // 2 + 1
b = (current_object_coordinates.shape[0] // 2 - 1) if (current_object_coordinates.shape[0] // 2 - 1) > 0 else 0
H_wave[:, :, :a, :a, :] = H_wave_quarter
H_wave[:, :, a:, :a, :] = H_wave_quarter[::-1, :, b::-1, :, :]
H_wave[:, :, a:, a:, :] = H_wave_quarter[::-1, ::-1, b::-1, b::-1, :]
H_wave[:, :, :a, a:, :] = H_wave_quarter[:, ::-1, :, b::-1, :]
H_wave_multiple_depth = np.flip(H_wave, axis=(0, 1))
n_depth_current = 0
H_wave_single_depth = H_wave_multiple_depth[:, :, :, :, n_depth_current]

super_res_factor_wave_x = H_wave_single_depth.shape[3]
super_res_factor_wave_y = H_wave_single_depth.shape[2]
forward_function = lambda object_field: lightfieldpackage.utils_deconv.forward_prevedel_super_resolution(
    object_field, H_wave_single_depth, super_res_factor_x=super_res_factor_wave_x,
    super_res_factor_y=super_res_factor_wave_y, microlens_pitch_x=num_micro_pixels, microlens_pitch_y=num_micro_pixels)

lightfield_image = forward_function(resized_image)
plt.figure(), plt.imshow(lightfield_image), plt.title("Lightfield Image"), plt.colorbar()

plt.imsave(os.path.join(results_directory, "H_wave_single_depth.png"), H_wave_single_depth[:, :, 0, 0], cmap=colormap)
plt.imsave(os.path.join(results_directory, "lightfield_image.png"), lightfield_image, cmap=colormap)
plt.imsave(os.path.join(results_directory, "lightfield_image_gray.png"), lightfield_image, cmap="gray")