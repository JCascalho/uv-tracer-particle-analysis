# Vilamoura Lower Shoreface — M2

## Fluorescent Tracer Detection and Quantification

This directory contains the standalone M2 Python routine developed for identifying and quantifying fluorescent sediment tracer particles in UV-light photographs.

The routine produces binary masks, outlined photographs, particle counts, tracer concentrations, and individual particle areas.

The image-processing algorithm preserves the validated Lab colour thresholds, morphological filtering, and pixel-area calibration.

## Contents

- `M2_fluorescent_tracer_detection.py` — standalone Python routine.

## Requirements

Python with NumPy, pandas, OpenCV, and openpyxl.

## Execution

Run the M2 script with the example photographs stored in the `examples` directory.

No geographical coordinates or external reference spreadsheet are required.
