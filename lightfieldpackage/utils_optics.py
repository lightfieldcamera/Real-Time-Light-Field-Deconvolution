import numpy as np
import scipy.special as special
import scipy.integrate as integrate
import tqdm

def calculate_mla_pattern(focal_length_ml, wave_number, grid_x, grid_y, num_lenses_x, num_lenses_y,
                          use_round_lenses=False):
    '''
    Berechnet das Mikrolinsenarray (MLA)-Beugungsmuster.

    Parameter:
    focal_length_ml: Brennweite der Mikrolinsen
    wave_number: Wellenzahl (k)
    grid_x, grid_y: Gitter für die Mikrolinsen (x und y Koordinaten)
    num_lenses_x, num_lenses_y: Anzahl der Mikrolinsen entlang x und y
    use_round_lenses: Boolean, ob runde Mikrolinsen verwendet werden sollen

    Rückgabe:
    mla_array: Beugungsmuster des Mikrolinsenarrays
    '''
    x_grid, y_grid = np.meshgrid(grid_x, grid_y)
    lens_norm = x_grid ** 2 + y_grid ** 2
    lens_pattern = np.exp(-1j * wave_number / (2 * focal_length_ml) * lens_norm)

    if use_round_lenses:
        # Maske für runde Mikrolinsen
        lens_radius = min(np.max(grid_x), np.max(grid_y)) / 2
        lens_mask = lens_norm <= lens_radius ** 2
        lens_pattern *= lens_mask

    mla_array = np.tile(lens_pattern, [num_lenses_y, num_lenses_x])
    return mla_array


def fresnel_propagation_2d(input_wave, grid_spacing, propagation_distance, wavelength):
    '''
    Führt die 2D-Fresnel-Propagation einer Welle durch.

    Parameter:
    input_wave: Eingabewelle
    grid_spacing: Abstand zwischen Punkten im Gitter
    propagation_distance: Propagationsentfernung
    wavelength: Wellenlänge des Lichts

    Rückgabe:
    propagated_wave: Propagierte Welle
    new_grid_spacing: Neuer Gitterabstand
    grid_coordinates: Neue Gitterkoordinaten
    '''
    num_points_x = input_wave.shape[1]
    num_points_y = input_wave.shape[0]
    wave_number = 2 * np.pi / wavelength

    # Frequenzraum-Gitter
    freq_x_spacing = 1. / (num_points_x * grid_spacing)
    freq_x = np.concatenate((np.arange(0, np.ceil(num_points_x / 2)),
                             np.arange(np.ceil(-num_points_x / 2), 0))) * freq_x_spacing
    freq_y_spacing = 1. / (num_points_y * grid_spacing)
    freq_y = np.concatenate((np.arange(0, np.ceil(num_points_y / 2)),
                             np.arange(np.ceil(-num_points_y / 2), 0))) * freq_y_spacing
    freq_x_grid, freq_y_grid = np.meshgrid(freq_x, freq_y)

    # Fresnel-Transferfunktion
    fresnel_transfer_function = np.exp(
        -1j * 2 * np.pi ** 2 * (freq_x_grid ** 2 + freq_y_grid ** 2) * propagation_distance / wave_number)

    # Fresnel-Transformation
    propagated_wave = np.exp(1j * wave_number * propagation_distance) * np.fft.ifft2(
        np.fft.fft2(input_wave) * fresnel_transfer_function)

    new_grid_spacing = grid_spacing
    grid_coordinates = np.arange(-num_points_x / 2, num_points_x / 2-1) * new_grid_spacing

    return propagated_wave, new_grid_spacing, grid_coordinates


def compute_point_spread_function_debye(obj_x, obj_y, obj_z, na, grid_x, grid_y, wavelength, mag, focus_mag, refr_index):
    '''
    Berechnet die Punktspreizfunktion (PSF) basierend auf der Debye-Theorie.

    Parameter:
    obj_x, obj_y, obj_z: Koordinaten der Punktlichtquelle im Objektraum
    na: Numerische Apertur des optischen Systems
    grid_x, grid_y: Koordinatengitter im Bildraum
    wavelength: Wellenlänge des Lichts
    mag: Vergrößerung des optischen Systems
    focus_mag: Vergrößerung im Fokus
    refr_index: Brechungsindex des Mediums

    Rückgabe:
    psf_result: Komplexe Punktspreizfunktion (PSF)
    '''
    # Berechnung der Wellenzahl
    k_wave = 2 * np.pi * refr_index / wavelength

    # Halböffnungswinkel
    half_angle = np.arcsin(na / refr_index)

    # Erstellung eines 2D-Koordinatengitters
    coord_grid_x, coord_grid_y = np.meshgrid(grid_x, grid_y)

    # Seitlicher Abstand
    lateral_dist = (
        ((coord_grid_x + focus_mag * obj_x) ** 2 + (coord_grid_y + focus_mag * obj_y) ** 2) ** 0.5
    ) / mag

    # Radialkoordinate
    radial_dist = k_wave * lateral_dist * np.sin(half_angle)

    # Axialkoordinate
    axial_dist = 4 * k_wave * obj_z * (np.sin(half_angle / 2) ** 2)

    # Integration
    theta_values = np.linspace(0, half_angle, 50)
    integrand_vals = integrand(
        theta=theta_values,
        axial_distance=axial_dist,
        radial_distance=np.repeat(np.expand_dims(radial_dist, axis=-1), 50, axis=-1),
        aperture_angle=half_angle
    )

    # Systemkonstante
    system_const = (
        (mag * refr_index ** 2) / (na ** 2 * wavelength ** 2)
        * np.exp(-1j * axial_dist / (4 * (np.sin(half_angle / 2) ** 2)))
    )

    # Numerische Integration
    integral_res = integrate.trapezoid(integrand_vals, dx=half_angle / 50, axis=-1)

    # Berechnung des Vignetting-Effekts
    #vignetting = calculate_vignetting(x1_grid, x2_grid, focal_length_ML, numerical_aperture)

    # Berechnung der Punktspreizfunktion (PSF)
    psf_result = system_const * integral_res# * vignetting

    return psf_result



def calculate_vignetting(x1_grid, x2_grid, focal_length_ML, numerical_aperture):
    """
    Berechnet den Vignetting-Effekt basierend auf der Cosinus-hoch-vier-Funktion.

    Parameter:
    x1_grid, x2_grid: Raumkoordinaten im Bildraum
    focal_length_ML: Brennweite der Mikrolinse
    numerical_aperture: Numerische Apertur des Systems

    Rückgabe:
    vignetting: Vignetting-Werte
    """
    # Berechnung der radialen Entfernung vom Bildzentrum
    radial_distance = np.sqrt(x1_grid ** 2 + x2_grid ** 2)

    # Berechnung des maximalen Winkels
    max_angle = np.arctan(numerical_aperture / focal_length_ML)

    # Normierte radiale Entfernung
    normalized_radial_distance = radial_distance / np.max(radial_distance)

    # Berechnung des Vignetting-Effekts
    vignetting = np.cos(normalized_radial_distance * max_angle) ** 4

    return vignetting

def integrand(theta, axial_distance, radial_distance, aperture_angle):
    '''
    :param theta: Winkel (in Radiant), der das Streuverhalten beschreibt
    :param radial_distance: Radialer Abstand (v)
    :param axial_distance: Axialer Abstand (u)
    :param aperture_angle: Halber Öffnungswinkel der numerischen Apertur (alpha)
    :return: Wert der Integrandfunktion
    '''
    result = (
        np.sqrt(np.cos(theta))  # Gewichtung durch den cosinus
        * (1 + np.cos(theta))  # Reflexionskoeffizient
        * np.exp(-(1j * axial_distance / 2) * (np.sin(theta / 2)**2) / (np.sin(aperture_angle / 2)**2))  # Phasenfaktor
        * special.j0(np.sin(theta) / np.sin(aperture_angle) * radial_distance)  # Bessel-Funktion der Ordnung 0
        * np.sin(theta)  # Gewichtung durch sin(theta) für polare Integration
    )
    return result



def generate_wave(image_height, image_width, object_coords_xyz, focal_length_ml, x_grid, y_grid, x_shift_grid, y_shift_grid, scaling_factor, wavelength, microlens_array, microlens_to_sensor_distance, microlens_to_ml_distance, mainlens_diameter, refractive_index, oversample_x, oversample_y, ml_focus_position):
    '''
    :param image_height: Höhe des resultierenden Bildes
    :param image_width: Breite des resultierenden Bildes
    :param object_coords_xyz: 3D-Koordinaten der Objekte
    :param focal_length_ml: Brennweite der Mikroobjektive
    :param x_grid: Gitter für die x-Achse
    :param y_grid: Gitter für die y-Achse
    :param x_shift_grid: Verschiebungsraum für x
    :param y_shift_grid: Verschiebungsraum für y
    :param scaling_factor: Skalierungsfaktor für die Simulation
    :param wavelength: Wellenlänge des Lichts
    :param microlens_array: Mikroobjektivarray
    :param microlens_to_sensor_distance: Abstand zwischen Mikroobjektivarray und Sensor
    :param microlens_to_ml_distance: Abstand zwischen Mikroobjektiven und Hauptobjektivarray
    :param mainlens_diameter: Durchmesser der Mikroobjektive
    :param refractive_index: Brechungsindex des Mediums
    :param oversample_x: Oversampling-Faktor für x
    :param oversample_y: Oversampling-Faktor für y
    :param ml_focus_position: Fokusposition der Mikroobjektive
    :return: Wellenfront und PSFs vor dem Mikroobjektivarray
    '''
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
    '''
    :param image_height: Höhe des resultierenden Bildes
    :param image_width: Breite des resultierenden Bildes
    :param object_coords_xyz: 3D-Koordinaten der Objekte
    :param x_grid: Gitter für die x-Achse
    :param y_grid: Gitter für die y-Achse
    :param x_shift_space: Verschiebungsraum für x
    :param y_shift_space: Verschiebungsraum für y
    :param scaling_factor: Skalierungsfaktor für die Simulation
    :param wavelength: Wellenlänge des Lichts
    :param microlens_array: Mikroobjektivarray (MLA)
    :param microlens_to_sensor_distance: Abstand zwischen Mikroobjektivarray und Sensor
    :param refractive_index: Brechungsindex des Mediums
    :param oversample_x: Oversampling-Faktor für x
    :param oversample_y: Oversampling-Faktor für y
    :param ml_focus_position: Fokusposition des Mikroobjektivarrays
    :param numerical_aperture: Numerische Apertur
    :param magnification: Vergrößerungsfaktor
    :return: Wellenfront H_wave und PSFs vor dem Mikroobjektivarray
    '''
    # Initialisiere Arrays
    wavefront = np.zeros((image_height, image_width, object_coords_xyz.shape[0], object_coords_xyz.shape[1], object_coords_xyz.shape[2]), dtype=np.float32)
    psfs_before_mla = np.zeros((y_shift_space.shape[0], x_shift_space.shape[0], object_coords_xyz.shape[2]), dtype=np.complex128)

    for microlens_idx, microlens_position in enumerate(tqdm.tqdm(object_coords_xyz[0, 0, :, 2])):
        z_position = microlens_position - ml_focus_position

        # Berechnung der PSF vor dem Mikroobjektivarray
        psf_before_mla = compute_point_spread_function_debye(
            0, 0, z_position, numerical_aperture, x_shift_space, y_shift_space, wavelength, magnification, magnification, refractive_index
        )
        psfs_before_mla[:, :, microlens_idx] = psf_before_mla

        # Zuschneiden der PSF
        psf_height, psf_width = psf_before_mla.shape
        cropped_width, cropped_height = len(x_grid), len(y_grid)
        crop_left = (psf_width - cropped_width) // 2
        crop_top = (psf_height - cropped_height) // 2
        crop_right = (psf_width + cropped_width) // 2
        crop_bottom = (psf_height + cropped_height) // 2

        for x_idx, x_position in enumerate(tqdm.tqdm(object_coords_xyz[:, 0, 0, 0])):
            for y_idx, y_position in enumerate(object_coords_xyz[0, :, 0, 1]):
                # Berechnung der Verschiebung
                x_shift = np.around(x_position / np.abs(x_grid[1] - x_grid[0]) * magnification).astype(int)
                if len(y_grid) > 1:
                    y_shift = np.around(y_position / np.abs(y_grid[1] - y_grid[0]) * magnification).astype(int)
                else:
                    y_shift = 0

                # Verschieben und Zuschneiden der PSF
                shifted_psf = np.roll(psf_before_mla, (-y_shift, -x_shift), axis=(0, 1))
                cropped_psf = shifted_psf[crop_top:crop_bottom, crop_left:crop_right]

                # PSF nach dem Mikroobjektivarray
                psf_after_mla = cropped_psf * microlens_array
                psf_sensor, _, _ = fresnel_propagation_2d(psf_after_mla, scaling_factor, microlens_to_sensor_distance, wavelength)

                # Berechnung der Intensität und Normalisierung
                intensity = np.abs(psf_sensor ** 2)
                intensity = intensity.reshape(image_height, oversample_y, image_width, oversample_x).mean(-1).mean(1)
                wavefront[:, :, x_idx, y_idx, microlens_idx] = intensity

    # Transposition und Normalisierung der Wellenfront
    wavefront = np.transpose(wavefront, axes=(0, 1, 3, 2, 4))
    wavefront /= wavefront.sum(axis=1, keepdims=True).sum(axis=0, keepdims=True)

    return wavefront, psfs_before_mla
