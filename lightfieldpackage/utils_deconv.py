import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import cv2
import tqdm
import time

def forward_prevedel_super_resolution(input_image, psf_single_depth, super_res_factor_x, super_res_factor_y, microlens_pitch_x, microlens_pitch_y):
    '''
    Führt eine Vorwärtsprojektion für superauflösendes Lichtfeld-Bildgebung nach Prevedel durch.

    Parameter:
    input_image: Eingangsbild (2D)
    psf_single_depth: Punktspreizfunktion (PSF) für eine einzelne Tiefe, Dimensionen (H, W, super_res_factor_y, super_res_factor_x)
    super_res_factor_x: Superauflösungsfaktor in x-Richtung
    super_res_factor_y: Superauflösungsfaktor in y-Richtung
    microlens_pitch_x: Abstand zwischen Mikrolinsen in x-Richtung
    microlens_pitch_y: Abstand zwischen Mikrolinsen in y-Richtung

    Rückgabe:
    output_image: Ergebnis der Vorwärtsprojektion
    '''
    print('\nForward Prevedel Single Depth Super Resolution')

    # Größe des Eingangsbilds an Superauflösungsfaktor anpassen
    cropped_input = input_image[:input_image.shape[0] // super_res_factor_y * super_res_factor_y,
                                 :input_image.shape[1] // super_res_factor_x * super_res_factor_x]

    # Schritte und Ausgabeform definieren
    step_size = np.array((microlens_pitch_y, microlens_pitch_x), dtype=int)
    output_shape = np.array(np.ceil(np.array(cropped_input.shape) / (super_res_factor_y, super_res_factor_x)) * (microlens_pitch_y, microlens_pitch_x), dtype=int)
    output_image = np.zeros(output_shape)

    for y_factor in tqdm.tqdm(range(super_res_factor_y)):  # Iteriere über y-PLQ-Faktor
        for x_factor in range(super_res_factor_x):  # Iteriere über x-PLQ-Faktor
            # Erstelle einen leeren Objektraum
            temp_object_space = np.zeros(output_shape)

            # Werte für die entsprechende PLQ (Plenoptische Ansicht) einfügen
            temp_object_space[
                (step_size[0] // 2)::step_size[0],
                (step_size[1] // 2)::step_size[1]
            ] = cropped_input[y_factor::super_res_factor_y, x_factor::super_res_factor_x]

            # PSF anwenden, wenn gültige Werte vorhanden sind
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
    '''
    Führt eine Rückwärtsprojektion für superauflösendes Lichtfeld-Bildgebung nach Prevedel durch.

    Parameter:
    projection_image: Eingangsprojektion (2D)
    psf_single_depth: Punktspreizfunktion (PSF) für eine einzelne Tiefe, Dimensionen (H, W, super_res_factor_y, super_res_factor_x)
    super_res_factor_x: Superauflösungsfaktor in x-Richtung
    super_res_factor_y: Superauflösungsfaktor in y-Richtung
    microlens_pitch_x: Abstand zwischen Mikrolinsen in x-Richtung
    microlens_pitch_y: Abstand zwischen Mikrolinsen in y-Richtung

    Rückgabe:
    reconstructed_image: Ergebnis der Rückwärtsprojektion
    '''
    print('\nBackward Prevedel Single Depth Super Resolution')

    # Initialisiere das Ergebnisbild
    reconstructed_image = np.zeros((
        super_res_factor_y * (projection_image.shape[0] // microlens_pitch_y),
        super_res_factor_x * (projection_image.shape[1] // microlens_pitch_x)
    ))

    for y_factor in tqdm.tqdm(range(super_res_factor_y)):  # Iteriere über y-PLQ-Faktor
        for x_factor in range(super_res_factor_x):  # Iteriere über x-PLQ-Faktor
            if np.any(projection_image) and np.any(psf_single_depth[:, :, y_factor, x_factor]):
                # PSF umkehren
                kernel = np.flip(psf_single_depth[:, :, y_factor, x_factor], axis=(0, 1))

                # Faltung mit der Projektion
                temp_result = cv2.filter2D(
                    src=np.expand_dims(projection_image, axis=-1),
                    ddepth=-1,
                    kernel=kernel,
                    borderType=cv2.BORDER_CONSTANT
                )
                temp_result = temp_result.clip(min=0)

                # Rekonstruktion aufteilen und summieren
                reconstructed_image[
                    y_factor::super_res_factor_y,
                    x_factor::super_res_factor_x
                ] += temp_result[
                    (microlens_pitch_y // 2)::microlens_pitch_y,
                    (microlens_pitch_x // 2)::microlens_pitch_x
                ]

    return reconstructed_image


def deconvRL(forwardFUN, backwardFUN, img, iter, init):
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
    '''
    Führt eine Rückwärtsprojektion mit PSF-Maske und Indizes für superauflösendes Lichtfeld-Bildgebung durch.

    Parameter:
    projection_image: Eingangsprojektion (2D)
    psf_mask: PSF-Maske mit Indizes, Dimensionen (H, W, super_res_factor_y, super_res_factor_x)
    lens_array_roi: Region of Interest (ROI) der Mikrolinsen
    super_res_factor_x: Superauflösungsfaktor in x-Richtung
    super_res_factor_y: Superauflösungsfaktor in y-Richtung
    microlens_pitch_x: Abstand zwischen Mikrolinsen in x-Richtung
    microlens_pitch_y: Abstand zwischen Mikrolinsen in y-Richtung

    Rückgabe:
    reconstructed_image: Rekonstruiertes Bild nach Rückwärtsprojektion
    '''
    # Initialisiere das Ergebnisbild
    reconstructed_image = np.zeros((
        super_res_factor_y * (projection_image.shape[0] // microlens_pitch_y),
        super_res_factor_x * (projection_image.shape[1] // microlens_pitch_x)
    ))

    num_shifts = projection_image.shape[0] // microlens_pitch_x

    for i in range(psf_mask.shape[2]):  # Iteriere über y-PLQ-Faktor
        for j in range(psf_mask.shape[3]):  # Iteriere über x-PLQ-Faktor
            # Extrahiere die PSF-Indizes
            psf_indices = np.array(np.nonzero(psf_mask[:, :, j, i]))

            # Verschiebe Indizes entsprechend der ROI und Gitter
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

            # Reshape für 1D-Indexierung
            shifted_indices_reshaped = shifted_indices.transpose((2, 0, 1, 3)).reshape(
                (2, num_shifts * num_shifts * psf_indices.shape[-1])
            )
            shifted_indices_1d = np.ravel_multi_index(shifted_indices_reshaped, projection_image.shape, mode="wrap")

            # Werte extrahieren und rekonstruieren
            temp_projection_1d = np.take(projection_image, shifted_indices_1d, mode="wrap")
            temp_projection_reshaped = temp_projection_1d.reshape(
                (num_shifts, num_shifts, psf_indices.shape[-1])
            )
            temp_projection_summed = np.sum(temp_projection_reshaped, axis=-1)

            # Ergebnis hinzufügen
            reconstructed_image[
                j::super_res_factor_y,
                i::super_res_factor_x
            ] += temp_projection_summed

    return reconstructed_image


def forward_prevedel_multi_depth_super_res(input_multi_depth, psf_multi_depth, super_res_factor_x, super_res_factor_y,
                                           microlens_pitch_x, microlens_pitch_y):
    '''
    Führt eine Vorwärtsprojektion für mehrere Tiefenebenen mit superauflösender Lichtfeld-Bildgebung nach Prevedel durch.

    Parameter:
    input_multi_depth: Eingangsbild für mehrere Tiefenebenen (3D)
    psf_multi_depth: Punktspreizfunktion (PSF) für mehrere Tiefenebenen, Dimensionen (H, W, super_res_factor_y, super_res_factor_x, D)
    super_res_factor_x: Superauflösungsfaktor in x-Richtung
    super_res_factor_y: Superauflösungsfaktor in y-Richtung
    microlens_pitch_x: Abstand zwischen Mikrolinsen in x-Richtung
    microlens_pitch_y: Abstand zwischen Mikrolinsen in y-Richtung

    Rückgabe:
    output_image: Ergebnis der Vorwärtsprojektion
    '''
    # Bestimme die Ausgabeform
    output_shape = np.array(
        np.ceil(np.array(input_multi_depth.shape[0:2]) / (super_res_factor_y, super_res_factor_x))
        * (microlens_pitch_y, microlens_pitch_x),
        dtype=int
    )
    output_image = np.zeros(output_shape)

    for depth_idx in range(psf_multi_depth.shape[-1]):  # Iteriere über Tiefenebenen
        start_time = time.time()
        input_single_depth = input_multi_depth[:, :, depth_idx]

        # Überspringe, wenn keine Werte vorhanden sind
        if np.sum(input_single_depth) == 0:
            continue

        psf_single_depth = psf_multi_depth[:, :, :, :, depth_idx]
        # Rufe die Vorwärtsprojektion für eine Tiefe auf
        output_image += forward_prevedel_super_resolution(
            input_single_depth, psf_single_depth, super_res_factor_x, super_res_factor_y, microlens_pitch_x,
            microlens_pitch_y
        )
        elapsed_time = time.time() - start_time
        print(f'\nOne forward pass {depth_idx + 1} | {psf_multi_depth.shape[-1]}, took {elapsed_time:.2f} secs')

    return output_image


def backward_prevedel_multi_depth_super_res(input_projection, psf_multi_depth, super_res_factor_x, super_res_factor_y,
                                            microlens_pitch_x, microlens_pitch_y):
    '''
    Führt eine Rückwärtsprojektion für mehrere Tiefenebenen mit superauflösender Lichtfeld-Bildgebung nach Prevedel durch.

    Parameter:
    input_projection: Eingangsprojektion (2D)
    psf_multi_depth: Punktspreizfunktion (PSF) für mehrere Tiefenebenen, Dimensionen (H, W, super_res_factor_y, super_res_factor_x, D)
    super_res_factor_x: Superauflösungsfaktor in x-Richtung
    super_res_factor_y: Superauflösungsfaktor in y-Richtung
    microlens_pitch_x: Abstand zwischen Mikrolinsen in x-Richtung
    microlens_pitch_y: Abstand zwischen Mikrolinsen in y-Richtung

    Rückgabe:
    output_multi_depth: Ergebnis der Rückwärtsprojektion für mehrere Tiefenebenen
    '''
    # Bestimme die Form der Ausgabe
    output_shape = np.array(
        np.ceil(np.array(input_projection.shape) * (super_res_factor_y, super_res_factor_x))
        / (microlens_pitch_y, microlens_pitch_x),
        dtype=int
    )
    output_multi_depth = np.zeros([*output_shape, psf_multi_depth.shape[-1]])

    for depth_idx in range(psf_multi_depth.shape[-1]):  # Iteriere über Tiefenebenen
        start_time = time.time()
        psf_single_depth = psf_multi_depth[:, :, :, :, depth_idx]

        # Rufe die Rückwärtsprojektion für eine Tiefe auf
        output_multi_depth[:, :, depth_idx] = backward_prevedel_super_resolution(
            input_projection, psf_single_depth, super_res_factor_x, super_res_factor_y, microlens_pitch_x,
            microlens_pitch_y
        )
        elapsed_time = time.time() - start_time
        print(f'\nOne backward pass {depth_idx + 1} | {psf_multi_depth.shape[-1]}, took {elapsed_time:.2f} secs')

    return output_multi_depth
