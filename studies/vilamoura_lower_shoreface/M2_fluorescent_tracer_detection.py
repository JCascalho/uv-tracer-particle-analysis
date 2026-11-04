from __future__ import annotations

"""
M2 — Fluorescent Tracer Detection and Quantification | Vilamoura

Author: João Cascalho (FCUL / IDL, University of Lisbon)
Based on: M4_tracer_identification_quantification(3).py
Demonstration version: November 2026

Standalone analysis of UV photographs in one selected folder. Each photograph
is identified by its filename. The routine exports binary tracer masks,
outlined photographs, particle counts, tracer-pixel concentrations and
individual particle areas. No spatial database or sample-level statistics
are required.

The validated Lab segmentation thresholds, morphology, external-contour
counting, connected-component area measurement and pixel calibration are
preserved from the original M4 routine.

Usage:
    python M2_fluorescent_tracer_detection.py "PATH_TO_PHOTOS"

If no folder is supplied, the script uses an 'examples' folder next to itself,
if present, otherwise the current working directory.

Requires: opencv-python, numpy, pandas, openpyxl.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
LORANGE = 35
AORANGE = 80
BORANGE = 130
DISK_RADIUS = 2
PIXEL_AREA_MM2 = 4.1698e-03
BIN_THRESH = 128
CONNECTIVITY = 8
MIN_AREA_PX = 0


# Validated M4 image-processing functions (unchanged).
def disk_kernel(radius: int) -> np.ndarray:
    """Binary disk kernel similar to MATLAB strel('disk', r)."""
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return ((x * x + y * y) <= r * r).astype(np.uint8)


def bwfilter_sedphoto(bw_255: np.ndarray, se: np.ndarray) -> np.ndarray:
    """MATLAB SedPhoto BWFilter: erode -> dilate -> dilate -> erode."""
    bw = bw_255
    bw = cv2.erode(bw, se, iterations=1)
    bw = cv2.dilate(bw, se, iterations=1)
    bw = cv2.dilate(bw, se, iterations=1)
    bw = cv2.erode(bw, se, iterations=1)
    return bw


def segment_sedphoto(bgr: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Segment tracer pixels in uint8 Lab and apply BWFilter."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    bw = ((L > LORANGE) & (a > AORANGE) & (b > BORANGE)).astype(np.uint8) * 255
    return bwfilter_sedphoto(bw, se)


def count_particles_bwboundaries_like(bw_255: np.ndarray) -> int:
    """Approx MATLAB bwboundaries(BW,'noholes'): count external contours only."""
    contours, _ = cv2.findContours(bw_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return int(len(contours))


def outline_image(bgr: np.ndarray, bw_255: np.ndarray) -> np.ndarray:
    """Draw external boundaries in white."""
    out = bgr.copy()
    contours, _ = cv2.findContours(bw_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 1)
    return out


def mask_filename_png(image_filename: str) -> str:
    """Always save masks as PNG (lossless), keeping 'mask_' prefix."""
    base = Path(image_filename).stem
    return f"mask_{base}.png"


def process_photographs(folder: str | Path) -> None:
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Photograph folder not found: {folder}")

    photographs = sorted(p for p in folder.iterdir()
                         if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not photographs:
        raise RuntimeError(f"No supported photographs found in: {folder}")

    masks_dir = folder / f"OUTPUT_MASK_{folder.name}"
    outlined_dir = folder / f"OUTPUT_OUTLINED_{folder.name}"
    masks_dir.mkdir(exist_ok=True)
    outlined_dir.mkdir(exist_ok=True)

    kernel = disk_kernel(DISK_RADIUS)
    count_rows = []
    area_rows = []
    image_area_rows = []

    for photo in photographs:
        bgr = cv2.imread(str(photo), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Skipping unreadable photograph: {photo.name}")
            continue

        pix_photo = int(bgr.shape[0] * bgr.shape[1])
        bw = segment_sedphoto(bgr, kernel)
        pix_particles = int(np.sum(bw > 0))
        concentration = float(pix_particles / pix_photo)
        n_particles = count_particles_bwboundaries_like(bw)
        image_area_m2 = float(pix_photo) * PIXEL_AREA_MM2 * 1e-6

        mask_path = masks_dir / mask_filename_png(photo.name)
        if not cv2.imwrite(str(mask_path), bw):
            raise OSError(f"Could not write mask: {mask_path}")
        outline_path = outlined_dir / f"outlined_{photo.name}"
        if not cv2.imwrite(str(outline_path), outline_image(bgr, bw)):
            raise OSError(f"Could not write outlined photograph: {outline_path}")

        count_rows.append({
            "image_name": photo.name,
            "np": n_particles,
            "concentration": concentration,
            "particles_per_m2": float(n_particles) / image_area_m2,
            "pix_particles": pix_particles,
            "pix_photo": pix_photo,
            "image_area_m2": image_area_m2,
        })

        # Same connected-component procedure and area calibration as M4.
        bw01 = (bw >= BIN_THRESH).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(
            bw01, connectivity=CONNECTIVITY
        )
        particle_id = 0
        total_area_px = 0
        for label in range(1, n):
            area_px = int(stats[label, cv2.CC_STAT_AREA])
            if area_px < MIN_AREA_PX:
                continue
            particle_id += 1
            total_area_px += area_px
            area_rows.append({
                "image_name": photo.stem,
                "particle_id": particle_id,
                "area_pixels": area_px,
                "area_mm2": area_px * PIXEL_AREA_MM2,
            })

        image_area_rows.append({
            "image_name": photo.stem,
            "particle_count_components": particle_id,
            "total_area_pixels": total_area_px,
            "total_area_mm2": total_area_px * PIXEL_AREA_MM2,
        })
        print(f"{photo.name}: {n_particles} particles; concentration={concentration:.6f}")

    if not count_rows:
        raise RuntimeError("None of the photographs could be read.")

    count_file = folder / f"OUTPUT_TRACER_COUNTS_{folder.name}.xlsx"
    area_file = folder / f"OUTPUT_PARTICLE_AREAS_{folder.name}.xlsx"
    metadata = pd.DataFrame({
        "parameter": ["module", "L_threshold", "a_threshold", "b_threshold",
                      "disk_radius", "pixel_area_mm2", "binary_threshold",
                      "connectivity", "minimum_area_pixels"],
        "value": ["M2", LORANGE, AORANGE, BORANGE, DISK_RADIUS,
                  PIXEL_AREA_MM2, BIN_THRESH, CONNECTIVITY, MIN_AREA_PX],
    })
    with pd.ExcelWriter(count_file, engine="openpyxl") as writer:
        pd.DataFrame(count_rows).to_excel(writer, sheet_name="Tracer_Counts", index=False)
        metadata.to_excel(writer, sheet_name="metadata", index=False)

    with pd.ExcelWriter(area_file, engine="openpyxl") as writer:
        pd.DataFrame(area_rows, columns=[
            "image_name", "particle_id", "area_pixels", "area_mm2"
        ]).to_excel(writer, sheet_name="Particle_Areas", index=False)
        pd.DataFrame(image_area_rows).to_excel(writer, sheet_name="Image_Summary", index=False)
        metadata.to_excel(writer, sheet_name="metadata", index=False)

    print(f"\nCompleted: {len(count_rows)} photograph(s)")
    print(f"Particle counts: {count_file}")
    print(f"Particle areas:  {area_file}")
    print(f"Binary masks:    {masks_dir}")
    print(f"Outlined photos: {outlined_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", default=None,
                        help="Folder containing the UV photographs")
    args = parser.parse_args()
    default_examples = Path(__file__).resolve().parent / "examples"
    selected = args.folder or (default_examples if default_examples.is_dir() else Path.cwd())
    process_photographs(selected)


if __name__ == "__main__":
    main()
