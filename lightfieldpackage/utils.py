import lightfieldpackage
import numpy as np
import cv2
class LightfieldProcessor:
    def __init__(self, is_1D):
        super(LightfieldProcessor, self).__init__()
        self.raw_to_lf = lightfieldpackage.utils.convert_raw_to_epi if is_1D else lightfieldpackage.utils.convert_raw_to_lightfield
        self.prepare_lf = self.prepare_lightfield_1D if is_1D else self.prepare_lightfield_2D
        self.refocus = self.refocus_1D if is_1D else self.refocus_2D
        self.shear = self.shear_1D if is_1D else self.shear_2D

    def prepare_lightfield_2D(self, lightfield_data, downsample_factor, num_s, num_t, num_u, num_v, interp_method=cv2.INTER_LINEAR):
        reshaped_data = lightfield_data.reshape((-1, *lightfield_data.shape[-2:]))
        interpolated_data = np.array(
            [cv2.resize(sub, (downsample_factor * num_s, downsample_factor * num_t), interpolation=interp_method)
             for sub in reshaped_data]
        )
        reshaped_interpolated = interpolated_data.reshape((num_u, num_v, *interpolated_data.shape[-2:]))

        # Energieerhaltung vor und nach Interpolation
        normalized_data = reshaped_interpolated * np.sum(lightfield_data) / np.sum(reshaped_interpolated)
        return normalized_data

    def prepare_lightfield_1D(self, lightfield_data, downsample_factor, num_s, num_u, interp_method=cv2.INTER_LINEAR):
        interpolated_data = cv2.resize(lightfield_data, (downsample_factor * num_s, num_u), interpolation=interp_method)
        # Energieerhaltung vor und nach Interpolation
        normalized_data = interpolated_data * np.sum(lightfield_data) / np.sum(interpolated_data)
        return normalized_data

    def refocus_1D(self, interpolated_lightfield, focus_factor, num_pixels_per_microlens, num_microlenses, downsample_factor, lenslet_pitch, mainlens_pitch, apply_scaling=True, interpolation_method="linear"):
        refocused = lightfieldpackage.utils.refocus_1D_lightfield(
            epipolar_image=interpolated_lightfield,
            focus_factor=focus_factor,
            coord_u=num_pixels_per_microlens,
            coord_v=num_pixels_per_microlens,
            coord_s=num_microlenses * downsample_factor,
            coord_t=num_microlenses * downsample_factor,
            pitch_t=lenslet_pitch / (num_microlenses * downsample_factor),
            pitch_s=lenslet_pitch / (num_microlenses * downsample_factor),
            pitch_v=mainlens_pitch / num_pixels_per_microlens,
            pitch_u=mainlens_pitch / num_pixels_per_microlens,
            domain='image',
            focal_distance=None,
            z1=None,
            apply_scaling=apply_scaling,
            interpolation_method=interpolation_method
        )[0]
        return refocused

    def refocus_2D(self, interpolated_lightfield, focus_factor, coord_u, coord_v, coord_s, coord_t, downsample_factor, lenslet_pitch, mainlens_pitch, apply_scaling=True, interpolation_method="linear"):
        refocused = lightfieldpackage.utils.refocus_lightfield_2D(
            lightfield=interpolated_lightfield,
            focus_factor=focus_factor,
            u_coord=coord_u,
            v_coord=coord_v,
            s_coord=coord_s * downsample_factor,
            t_coord=coord_t * downsample_factor,
            pitch_t=lenslet_pitch / (coord_t * downsample_factor),
            pitch_s=lenslet_pitch / (coord_s * downsample_factor),
            pitch_v=mainlens_pitch / coord_v,
            pitch_u=mainlens_pitch / coord_u,
            domain='image',
            focal_distance=None,
            z1=None,
            apply_scaling=apply_scaling,
            interpolation_method=interpolation_method
        )[0]
        return refocused

    def shear_1D(self, interpolated_lightfield, focus_factor, num_pixels_per_microlens, num_microlenses, downsample_factor, lenslet_pitch, mainlens_pitch, apply_scaling=True, interpolation_method="linear"):
        sheared_data = lightfieldpackage.utils.shear_epipolar_image(
            epipolar_image=interpolated_lightfield,
            focus_factor=focus_factor,
            coord_u=num_pixels_per_microlens,
            coord_v=num_pixels_per_microlens,
            coord_s=num_microlenses * downsample_factor,
            coord_t=num_microlenses * downsample_factor,
            pitch_t=lenslet_pitch / (num_microlenses * downsample_factor),
            pitch_s=lenslet_pitch / (num_microlenses * downsample_factor),
            pitch_v=mainlens_pitch / num_pixels_per_microlens,
            pitch_u=mainlens_pitch / num_pixels_per_microlens,
            apply_scaling=apply_scaling,
            interpolation_method=interpolation_method
        )[0]
        return sheared_data

    def shear_2D(self, interpolated_lightfield, focus_factor, num_pixels_per_microlens, num_microlenses, downsample_factor, lenslet_pitch, mainlens_pitch, apply_scaling=True, interpolation_method="linear"):
        sheared_data = lightfieldpackage.utils.shear_lightfield(
            lightfield=interpolated_lightfield,
            focus_factor=focus_factor,
            coord_u=num_pixels_per_microlens,
            coord_v=num_pixels_per_microlens,
            coord_s=num_microlenses * downsample_factor,
            coord_t=num_microlenses * downsample_factor,
            pitch_t=lenslet_pitch / (num_microlenses * downsample_factor),
            pitch_s=lenslet_pitch / (num_microlenses * downsample_factor),
            pitch_v=mainlens_pitch / num_pixels_per_microlens,
            pitch_u=mainlens_pitch / num_pixels_per_microlens,
            scaling_factor=apply_scaling,
            interpolation_method=interpolation_method
        )[0]
        return sheared_data


def shear_epipolar_image(epipolar_image, focus_factor, coord_u, coord_v, coord_s, coord_t, pitch_t, pitch_s, pitch_v,
                         pitch_u, apply_scaling=True, interpolation_method="linear", scaling_factor=1):
    '''
    :param epipolar_image: Eingabe-Epipolarbild mit Dimensionen (u, s)
    :param focus_factor: Scherungsfaktor (Alpha)
    :param coord_u: Koordinaten entlang der u-Achse
    :param coord_v: Koordinaten entlang der v-Achse
    :param coord_s: Koordinaten entlang der s-Achse
    :param coord_t: Koordinaten entlang der t-Achse
    :param pitch_t: Mikroobjektivabstand entlang der t-Achse
    :param pitch_s: Mikroobjektivabstand entlang der s-Achse
    :param pitch_v: Mikrobildschrittweite entlang der v-Achse
    :param pitch_u: Mikrobildschrittweite entlang der u-Achse
    :param apply_scaling: Ob Skalierung angewandt wird (Standard: True)
    :param interpolation_method: Methode der Interpolation (Standard: "linear")
    :param scaling_factor: Skalierungsfaktor (Standard: 1)
    :return: Gescherte Epipolarbilddaten und Normalisierungsfaktor
    '''
    normalization_factor_epi = coord_u

    # Skalierungsfaktoren berechnen:
    scale_factor_s = np.float32(pitch_s * scaling_factor)
    scale_factor_u = np.float32(pitch_u)
    oversampling = 1

    data_dimensions = np.concatenate((np.array([coord_t, coord_s]) * oversampling, [coord_v, coord_u]))
    data_span = np.float32((data_dimensions - 1) / 2)

    sample_s = np.linspace(-data_span[1], data_span[1], data_dimensions[1]) * scale_factor_s / oversampling
    sample_u = np.linspace(-data_span[3], data_span[3], data_dimensions[3]) * scale_factor_u

    import scipy as sp
    interpolator_epi = sp.interpolate.RegularGridInterpolator(
        (sample_u, sample_s), epipolar_image, bounds_error=False, fill_value=0, method=interpolation_method
    )
    (out_sample_u, out_sample_s) = np.meshgrid(sample_u, sample_s, indexing='ij')

    if apply_scaling:
        out_sample_s_adjusted = out_sample_s * focus_factor + out_sample_u * (1 - focus_factor)
    else:
        out_sample_s_adjusted = out_sample_s + out_sample_u * (1 - focus_factor)

    output_grid_epi = np.array((out_sample_u, out_sample_s_adjusted))
    sheared_coordinates_epi = np.transpose(output_grid_epi, axes=(1, 2, 0))
    sheared_epipolar_data = interpolator_epi(sheared_coordinates_epi)

    return sheared_epipolar_data, normalization_factor_epi


def refocus_1D_lightfield(epipolar_image, focus_factor, coord_u, coord_v, coord_s, coord_t, pitch_t, pitch_s, pitch_v, pitch_u, domain, focal_distance, z1, apply_scaling=True, interpolation_method="linear"):
    '''
    :param epipolar_image: Epipolarbild, z. B. np.flipud(np.reshape(slice, (-1, N_MP)).transpose())
    :param focus_factor: Refokussierungsparameter (Alpha)
    :param coord_u: Koordinaten entlang der u-Achse
    :param coord_v: Koordinaten entlang der v-Achse
    :param coord_s: Koordinaten entlang der s-Achse
    :param coord_t: Koordinaten entlang der t-Achse
    :param pitch_t: Mikroobjektivabstand entlang der t-Achse
    :param pitch_s: Mikroobjektivabstand entlang der s-Achse
    :param pitch_v: Mikrobildschrittweite entlang der v-Achse
    :param pitch_u: Mikrobildschrittweite entlang der u-Achse
    :param domain: Domain (z. B. Raum oder Frequenzraum)
    :param focal_distance: Fokusabstand
    :param z1: Abstand zur Bildaufnahmeebene
    :param apply_scaling: Ob Skalierung angewandt wird (Standard: True)
    :param interpolation_method: Methode der Interpolation (Standard: "linear")
    :return: Refokussiertes Projektionsergebnis, gescherte Epipolarbilddaten, Eingabe-Epipolarbild und unfokussierte Projektionen
    '''
    sheared_epipolar_data, normalization_factors_epi = shear_epipolar_image(
        epipolar_image, focus_factor, coord_u, coord_v, coord_s, coord_t,
        pitch_t, pitch_s, pitch_v, pitch_u,
        apply_scaling=apply_scaling, interpolation_method=interpolation_method
    )
    # Integration:
    refocused_projection = np.sum(sheared_epipolar_data, axis=(0))
    refocused_projection /= normalization_factors_epi

    unfocused_projections = np.sum(epipolar_image, axis=(0))
    unfocused_projections /= normalization_factors_epi

    return refocused_projection, sheared_epipolar_data, epipolar_image, unfocused_projections


def shear_lightfield(lightfield, focus_factor, coord_s, coord_t, coord_u, coord_v, pitch_t, pitch_s, pitch_v, pitch_u,
                     apply_scaling=True, interpolation_method="linear", scaling_factor=1):
    '''
    :param lightfield: Eingabe-Lightfield (N_MP, N_MP, N_MLA, N_MLA)
    :param focus_factor: Alpha-Wert für die Scherung
    :param coord_s: Anzahl der Mikrolinsen entlang der s-Achse (N_MLA)
    :param coord_t: Anzahl der Mikrolinsen entlang der t-Achse (N_MLA)
    :param coord_u: Anzahl der Mikrobildpixel entlang der u-Achse (N_MP)
    :param coord_v: Anzahl der Mikrobildpixel entlang der v-Achse (N_MP)
    :param pitch_t: Mikroobjektivabstand entlang der t-Achse
    :param pitch_s: Mikroobjektivabstand entlang der s-Achse
    :param pitch_v: Mikrobildschrittweite entlang der v-Achse
    :param pitch_u: Mikrobildschrittweite entlang der u-Achse
    :param apply_scaling: Ob Skalierung angewandt wird (Standard: True)
    :param interpolation_method: Methode der Interpolation (Standard: "linear")
    :param scaling_factor: Skalierungsfaktor (Standard: 1)
    :return: Gescherte Lightfield-Daten und Normalisierungsfaktor
    credits to plenoptomos
    '''
    lightfield_image = lightfield
    lenslet_dimensions = np.array([coord_v, coord_u], dtype=np.int32)

    scale_factor_t = np.float32(pitch_t * scaling_factor)  # D_S / t * scaling_factor
    scale_factor_s = np.float32(pitch_s * scaling_factor)  # D_S / s * scaling_factor
    scale_factor_v = np.float32(pitch_v)  # D_ML / v
    scale_factor_u = np.float32(pitch_u)  # D_ML / u
    oversampling = 1
    data_dimensions = np.concatenate((np.array([coord_t, coord_s]) * oversampling, [coord_v, coord_u]))
    data_span = np.float32((data_dimensions - 1) / 2)
    sample_t = np.linspace(-data_span[0], data_span[0], data_dimensions[0],
                           dtype=np.float32) * scale_factor_t / oversampling
    sample_s = np.linspace(-data_span[1], data_span[1], data_dimensions[1],
                           dtype=np.float32) * scale_factor_s / oversampling
    sample_v = np.linspace(-data_span[2], data_span[2], data_dimensions[2], dtype=np.float32) * scale_factor_v
    sample_u = np.linspace(-data_span[3], data_span[3], data_dimensions[3], dtype=np.float32) * scale_factor_u

    import scipy as sp
    interpolator_4D = sp.interpolate.RegularGridInterpolator(
        (sample_v, sample_u, sample_t, sample_s), lightfield_image, bounds_error=False, fill_value=0,
        method=interpolation_method
    )
    (out_sample_v, out_sample_u, in_sample_t, in_sample_s) = np.meshgrid(sample_v, sample_u, sample_t, sample_s,
                                                                         indexing='ij')

    if apply_scaling:
        out_sample_t = in_sample_t / focus_factor + out_sample_v * (1 - 1 / focus_factor)
        out_sample_s = in_sample_s / focus_factor + out_sample_u * (1 - 1 / focus_factor)
    else:
        out_sample_t = in_sample_t + out_sample_v * (1 - 1 / focus_factor)
        out_sample_s = in_sample_s + out_sample_u * (1 - 1 / focus_factor)

    output_grid = np.array((out_sample_v, out_sample_u, out_sample_t, out_sample_s))
    sheared_coordinates = np.transpose(output_grid, axes=(1, 2, 3, 4, 0))
    sheared_lightfield_data = interpolator_4D(sheared_coordinates)

    normalization_factor = np.prod(lenslet_dimensions)
    return sheared_lightfield_data, normalization_factor


def refocus_lightfield_2D(lightfield, focus_factor, u_coord, v_coord, s_coord, t_coord, pitch_t, pitch_s, pitch_v, pitch_u, domain, focal_distance, z1, apply_scaling=True, interpolation_method="linear"):
    '''
    :param lightfield: Dimensionen sind u,v,s,t oder v,u,t,s
    :param focus_factor: Refokussierungsparameter
    :param u_coord: Koordinaten entlang der u-Achse
    :param v_coord: Koordinaten entlang der v-Achse
    :param s_coord: Koordinaten entlang der s-Achse
    :param t_coord: Koordinaten entlang der t-Achse
    :param pitch_t: Mikroobjektivabstand entlang der t-Achse
    :param pitch_s: Mikroobjektivabstand entlang der s-Achse
    :param pitch_v: Mikrobildschrittweite entlang der v-Achse
    :param pitch_u: Mikrobildschrittweite entlang der u-Achse
    :param domain: Domain (Raum, in dem die Operationen durchgeführt werden)
    :param focal_distance: Fokusabstand
    :param z1: Abstand zur Bildaufnahmeebene
    :param apply_scaling: Ob Skalierung angewandt wird (Standard: True)
    :param interpolation_method: Interpolationsmethode (Standard: "linear")
    :return: Refokussiertes Projektionsergebnis und unfokussierte Projektionen
    '''
    sheared_lightfield, normalization_factors = shear_lightfield(
        lightfield, focus_factor, s_coord, t_coord, u_coord, v_coord,
        pitch_t, pitch_s, pitch_v, pitch_u, apply_scaling, interpolation_method, scaling_factor=1
    )
    # Integration:
    refocused_result = np.sum(sheared_lightfield, axis=(0, 1))
    refocused_result /= normalization_factors

    unfocused_results = np.sum(lightfield, axis=(0, 1))
    unfocused_results /= normalization_factors

    return refocused_result, unfocused_results

def convert_raw_to_epi(raw_input, microlens_pitch):
    epipolar_image = np.reshape(raw_input, (microlens_pitch, -1), order='F')
    return epipolar_image


def convert_raw_to_lightfield(raw_input, microlens_pitch):
    input_data = np.expand_dims(raw_input, -1)
    microlens_raw_dimensions = np.array([microlens_pitch, microlens_pitch], dtype=np.int32)
    array_start_offsets = np.array([0, 0], dtype=np.int32)

    start_offsets = array_start_offsets
    microlens_spacing = microlens_raw_dimensions
    microlens_end_indices_x = np.arange((start_offsets[1] + microlens_spacing[1]), input_data.shape[1] + 1, microlens_spacing[1])
    microlens_end_indices_y = np.arange((start_offsets[0] + microlens_spacing[0]), input_data.shape[0] + 1, microlens_spacing[0])

    input_data = input_data[start_offsets[0]:microlens_end_indices_y[-1], start_offsets[1]:microlens_end_indices_x[-1]]

    lightfield_array_size = np.array([len(microlens_end_indices_y), len(microlens_end_indices_x)])
    reshaped_dimensions = np.concatenate((lightfield_array_size, microlens_spacing))
    (intermediate_lightfield, reshaped_lightfield_dimensions) = (input_data, reshaped_dimensions)
    ###
    reshaped_dimensions = reshaped_lightfield_dimensions
    # Organize the lightfield as a 4-D structure:
    intermediate_reshape = (reshaped_dimensions[2], reshaped_dimensions[0], reshaped_dimensions[3], reshaped_dimensions[1])
    intermediate_lightfield = np.reshape(intermediate_lightfield, intermediate_reshape, order='F')
    intermediate_lightfield = np.transpose(intermediate_lightfield, axes=(1, 3, 0, 2))
    ###

    permutation_order = (2, 3, 0, 1)
    lightfield = np.transpose(intermediate_lightfield, permutation_order)  # from micro-image mode to sub-aperture mode
    return lightfield
