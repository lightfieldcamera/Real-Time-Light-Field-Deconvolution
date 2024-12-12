# Real-Time Light Field Deconvolution

This repository contains the code and datasets used for the paper **"Real-Time Deconvolution of Light Fields Through Pixel Selection in the Point Spread Function and Direct Inversion of the Image Formation Process"**, currently under review at *Applied Optics*.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Reproducing Results](#reproducing-results)
- [Citation](#citation)

## Installation

To set up the environment:

1. Clone the repository:
   git clone https://github.com/lightfieldcamera/Real-Time-Light-Field-Deconvolution.git \
   cd Real-Time-Light-Field-Deconvolution

2. Install the project:
   pip install .

3. Install dependencies:
   pip install -r requirements.txt

Make sure you're using **Python 3.10** for compatibility.

## Dataset

The dataset required for the experiments is hosted on Zenodo. Download it using the following link:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14210625.svg)](https://doi.org/10.5281/zenodo.14210625)

Once downloaded, move the dataset to the `data/` directory within the project: \
```console
mv <path_to_downloaded_dataset> ./data
```

## Reproducing Results

To reproduce the results presented in the paper, run the scripts provided in the `examples/` folder.

Example to generate a light field:
```console
python examples/Generate_LF_from_Image/generate_lf_with_wave_optics.py
```
Example to use digital refocusing:
```console
python examples/Digital_Refocusing/reconst_USAF_with_digital_refocusing.py
```
Example to back projection:
```console
python examples/LF_Direct_Deconvolution_w_back_projection/reconst_USAF_with_direct_deconv_and_H_wave.py
```
Example to back projection with pixel Selection:
```console
python examples/LF_Direct_Deconvolution_w_back_projection_w_Pixel_Selection/reconst_USAF_with_backprojection_and_H_wave_with_indices.py
```

## Citation

If you use this code or dataset in your research, please cite our paper:

tbd