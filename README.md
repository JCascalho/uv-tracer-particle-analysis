# UV Tracer Particle Analysis

Python routine for identifying and quantifying fluorescent sediment tracer
particles in UV-light photographs.

The workflow segments tracer pixels in Lab colour space, applies morphological
filtering, counts detected particles, estimates particle density, measures
particle areas from binary masks, and exports masks, outlined diagnostic images,
and Excel workbooks.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Basic Usage

Process one folder that directly contains UV photographs:

```bash
python uv_tracer_particle_analysis.py "path/to/image_folder"
```

Process sample subfolders under a parent directory:

```bash
python uv_tracer_particle_analysis.py "path/to/parent_folder" --by-folders
```

Use a photo-reference Excel database:

```bash
python uv_tracer_particle_analysis.py "path/to/image_folder" --single-folder --reference-excel "path/to/photo_reference.xlsx"
```

Tune segmentation thresholds:

```bash
python uv_tracer_particle_analysis.py "path/to/image_folder" --l-threshold 35 --a-threshold 80 --b-threshold 130
```

## Inputs

Supported image formats:

```text
.jpg, .jpeg, .png, .tif, .tiff, .bmp
```

Optional reference Excel columns:

```text
C, S, G, x, y, SS, M, P
```

The `P` value is matched to the last integer found in each image filename.


## Example Images

The `examples` folder includes one or two small demonstration UV-light images.

These images are provided only to illustrate the expected input format and to allow users to test the routine.
Real unpublished field photographs or complete sample datasets are not included for data-protection and publication reasons.

Users should replace the example images with their own UV-light photographs when applying the workflow to real samples.
## Outputs

For each processed image folder, the routine creates:

- `OUTPUT_MASK_*`: binary tracer masks.
- `OUTPUT_OUTLINED_*`: original images with detected particle outlines.
- `OUTPUT_TRACER_COUNTS_*.xlsx`: particle counts and image-level summaries.
- `OUTPUT_PARTICLE_AREAS_*.xlsx`: particle-area measurements and summaries.

## Calibration Note

The default Lab thresholds and pixel-area calibration reproduce the original
study setup:

```text
L > 35
a > 80
b > 130
pixel_area_mm2 = 4.1698e-03
```

Users should validate or adjust these parameters for their own camera,
illumination, exposure, tracer colour, and sediment background.
