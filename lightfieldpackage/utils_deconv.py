import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import cv2
import tqdm
import time

def forward_prevedel_super_resolution(input_image, psf_single_depth, super_res_factor_x, super_res_factor_y, microlens_pitch_x, microlens_pitch_y):
    """
    Performs a forward projection for super-resolution lightfield imaging following the Prevedel method.

    Parameters:
    - input_image: Input 2D image.
    - psf_single_depth: Point Spread Function (PSF) for a single depth, dimensions (H, W, super_res_factor_y, super_res_factor_x).
    - super_res_factor_x: Super-resolution factor in the x direction.
    - super_res_factor_y: Super-resolution factor in the y direction.
    - microlens_pitch_x: Microlens spacing in the x direction.
    - microlens_pitch_y: Microlens spacing in the y direction.

    Returns:
    - output_image: Result of the forward projection.
    """
    print('\nForward Prevedel Single Depth Super Resolution')

    cropped_input = input_image[:input_image.shape[0] // super_res_factor_y * super_res_factor_y,
                                 :input_image.shape[1] // super_res_factor_x * super_res_factor_x]

    step_size = np.array((microlens_pitch_y, microlens_pitch_x), dtype=int)
    output_shape = np.array(np.ceil(np.array(cropped_input.shape) / (super_res_factor_y, super_res_factor_x)) * (microlens_pitch_y, microlens_pitch_x), dtype=int)
    output_image = np.zeros(output_shape)

    for y_factor in tqdm.tqdm(range(super_res_factor_y)):  #
        for x_factor in range(super_res_factor_x):  #
            temp_object_space = np.zeros(output_shape)

            temp_object_space[
                (step_size[0] // 2)::step_size[0],
                (step_size[1] // 2)::step_size[1]
            ] = cropped_input[y_factor::super_res_factor_y, x_factor::super_res_factor_x]

            if np.any(temp_object_space) and np.any(psf_single_depth[:, :, y_factor, x_factor]):
                result = cv2.filter2D(
                    src=np.expand_dims(temp_object_space, axis=-1),
                    ddepth=-1,
                    kernel=np.flip(psf_single_depth[:, :, y_factor, x_factor], axis=(0, 1)),
                    borderType=cv2.BORDER_CONSTANT
                )
                result = result.clip(min=0)
                output_image += result

    return output_image


def backward_prevedel_super_resolution(projection_image, psf_single_depth, super_res_factor_x, super_res_factor_y, microlens_pitch_x, microlens_pitch_y):
    """
    Performs a backward projection for super-resolution lightfield imaging following the Prevedel method.

    Parameters:
    - projection_image: Input 2D projection.
    - psf_single_depth: Point Spread Function (PSF) for a single depth, dimensions (H, W, super_res_factor_y, super_res_factor_x).
    - super_res_factor_x: Super-resolution factor in the x direction.
    - super_res_factor_y: Super-resolution factor in the y direction.
    - microlens_pitch_x: Microlens spacing in the x direction.
    - microlens_pitch_y: Microlens spacing in the y direction.

    Returns:
    - reconstructed_image: Result of the backward projection.
    """
    print('\nBackward Prevedel Single Depth Super Resolution')

    reconstructed_image = np.zeros((
        super_res_factor_y * (projection_image.shape[0] // microlens_pitch_y),
        super_res_factor_x * (projection_image.shape[1] // microlens_pitch_x)
    ))

    for y_factor in tqdm.tqdm(range(super_res_factor_y)):  # Iteriere über y-PLQ-Faktor
        for x_factor in range(super_res_factor_x):  # Iteriere über x-PLQ-Faktor
            if np.any(projection_image) and np.any(psf_single_depth[:, :, y_factor, x_factor]):
                kernel = np.flip(psf_single_depth[:, :, y_factor, x_factor], axis=(0, 1))

                temp_result = cv2.filter2D(
                    src=np.expand_dims(projection_image, axis=-1),
                    ddepth=-1,
                    kernel=kernel,
                    borderType=cv2.BORDER_CONSTANT
                )
                temp_result = temp_result.clip(min=0)

                reconstructed_image[
                    y_factor::super_res_factor_y,
                    x_factor::super_res_factor_x
                ] += temp_result[
                    (microlens_pitch_y // 2)::microlens_pitch_y,
                    (microlens_pitch_x // 2)::microlens_pitch_x
                ]
    return reconstructed_image


def deconvRL(forwardFUN, backwardFUN, img, iter, init):
    """
    Performs Richardson-Lucy deconvolution.

    Parameters:
    - forwardFUN: Function for the forward projection.
    - backwardFUN: Function for the backward projection.
    - img: Input image.
    - iter: Number of iterations.
    - init: Initial guess for the reconstruction.

    Returns:
    - recon: Deconvolved image.
    """
    recon = init
    print('\nDeconvolution:')
    for i in range(iter):
        time_start = time.time()
        fpj = forwardFUN(recon)
        errorBack = img / fpj
        errorBack[np.isnan(errorBack)] = 0
        errorBack[np.isinf(errorBack)] = 0
        errorBack[errorBack < 0] = 0
        errorBack[errorBack > 1e+10] = 0
        bpjError = backwardFUN(errorBack)

        recon = recon * bpjError

        elapsed = time.time() - time_start
        print('\niter ', i + 1, ' | ', iter, ', took ', elapsed, ' secs')
    return recon
def backward_with_indices_super_resolution_with_mask(projection_image, psf_mask, lens_array_roi, super_res_factor_x, super_res_factor_y, microlens_pitch_x, microlens_pitch_y):
    """
    Performs backward projection with PSF mask and indices for super-resolution lightfield imaging.

    Parameters:
    - projection_image: Input 2D projection image.
    - psf_mask: PSF mask with indices, dimensions (H, W, super_res_factor_y, super_res_factor_x).
    - lens_array_roi: Region of Interest (ROI) for microlenses.
    - super_res_factor_x: Super-resolution factor in the x direction.
    - super_res_factor_y: Super-resolution factor in the y direction.
    - microlens_pitch_x: Microlens spacing in the x direction.
    - microlens_pitch_y: Microlens spacing in the y direction.

    Returns:
    - reconstructed_image: Reconstructed image after backward projection.
    """
    reconstructed_image = np.zeros((
        super_res_factor_y * (projection_image.shape[0] // microlens_pitch_y),
        super_res_factor_x * (projection_image.shape[1] // microlens_pitch_x)
    ))

    num_shifts = projection_image.shape[0] // microlens_pitch_x

    for i in range(psf_mask.shape[2]):  # Iterate over the y super-resolution factor
        for j in range(psf_mask.shape[3]):  # Iterate over the x super-resolution factor
            psf_indices = np.array(np.nonzero(psf_mask[:, :, j, i]))

            shifted_indices = psf_indices[None, None, ...] + \
                              np.transpose(
                                  np.array(
                                      np.meshgrid(
                                          np.arange(-lens_array_roi, num_shifts - lens_array_roi),
                                          np.arange(-lens_array_roi, num_shifts - lens_array_roi)
                                      )
                                  ),
                                  [2, 1, 0]
                              )[..., None] * microlens_pitch_x

            shifted_indices_reshaped = shifted_indices.transpose((2, 0, 1, 3)).reshape(
                (2, num_shifts * num_shifts * psf_indices.shape[-1])
            )
            shifted_indices_1d = np.ravel_multi_index(shifted_indices_reshaped, projection_image.shape, mode="wrap")

            temp_projection_1d = np.take(projection_image, shifted_indices_1d, mode="wrap")
            temp_projection_reshaped = temp_projection_1d.reshape(
                (num_shifts, num_shifts, psf_indices.shape[-1])
            )
            temp_projection_summed = np.sum(temp_projection_reshaped, axis=-1)

            reconstructed_image[
                j::super_res_factor_y,
                i::super_res_factor_x
            ] += temp_projection_summed

    return reconstructed_image


def forward_prevedel_multi_depth_super_res(input_multi_depth, psf_multi_depth, super_res_factor_x, super_res_factor_y,
                                           microlens_pitch_x, microlens_pitch_y):
    """
    Performs forward projection for multiple depth levels in super-resolution lightfield imaging following the Prevedel method.

    Parameters:
    - input_multi_depth: Input 3D image for multiple depth levels.
    - psf_multi_depth: PSF for multiple depth levels, dimensions (H, W, super_res_factor_y, super_res_factor_x, D).
    - super_res_factor_x: Super-resolution factor in the x direction.
    - super_res_factor_y: Super-resolution factor in the y direction.
    - microlens_pitch_x: Microlens spacing in the x direction.
    - microlens_pitch_y: Microlens spacing in the y direction.

    Returns:
    - output_image: Result of the forward projection.
    """
    output_shape = np.array(
        np.ceil(np.array(input_multi_depth.shape[0:2]) / (super_res_factor_y, super_res_factor_x))
        * (microlens_pitch_y, microlens_pitch_x),
        dtype=int
    )
    output_image = np.zeros(output_shape)

    for depth_idx in range(psf_multi_depth.shape[-1]):  # Iterate over depth levels
        start_time = time.time()
        input_single_depth = input_multi_depth[:, :, depth_idx]

        if np.sum(input_single_depth) == 0:
            continue

        psf_single_depth = psf_multi_depth[:, :, :, :, depth_idx]
        output_image += forward_prevedel_super_resolution(
            input_single_depth, psf_single_depth, super_res_factor_x, super_res_factor_y, microlens_pitch_x,
            microlens_pitch_y
        )
        elapsed_time = time.time() - start_time
        print(f'\nOne forward pass {depth_idx + 1} | {psf_multi_depth.shape[-1]}, took {elapsed_time:.2f} secs')

    return output_image


def backward_prevedel_multi_depth_super_res(input_projection, psf_multi_depth, super_res_factor_x, super_res_factor_y,
                                            microlens_pitch_x, microlens_pitch_y):
    """
    Performs backward projection for multiple depth levels in super-resolution lightfield imaging following the Prevedel method.

    Parameters:
    - input_projection: Input 2D projection image.
    - psf_multi_depth: PSF for multiple depth levels, dimensions (H, W, super_res_factor_y, super_res_factor_x, D).
    - super_res_factor_x: Super-resolution factor in the x direction.
    - super_res_factor_y: Super-resolution factor in the y direction.
    - microlens_pitch_x: Microlens spacing in the x direction.
    - microlens_pitch_y: Microlens spacing in the y direction.

    Returns:
    - output_multi_depth: Result of the backward projection for multiple depth levels.
    """
    output_shape = np.array(
        np.ceil(np.array(input_projection.shape) * (super_res_factor_y, super_res_factor_x))
        / (microlens_pitch_y, microlens_pitch_x),
        dtype=int
    )
    output_multi_depth = np.zeros([*output_shape, psf_multi_depth.shape[-1]])

    for depth_idx in range(psf_multi_depth.shape[-1]):  # Iterate over depth levels
        start_time = time.time()
        psf_single_depth = psf_multi_depth[:, :, :, :, depth_idx]

        output_multi_depth[:, :, depth_idx] = backward_prevedel_super_resolution(
            input_projection, psf_single_depth, super_res_factor_x, super_res_factor_y, microlens_pitch_x,
            microlens_pitch_y
        )
        elapsed_time = time.time() - start_time
        print(f'\nOne backward pass {depth_idx + 1} | {psf_multi_depth.shape[-1]}, took {elapsed_time:.2f} secs')

    return output_multi_depth
