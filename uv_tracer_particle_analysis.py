"""UV fluorescent tracer particle identification and quantification.

This command-line routine identifies fluorescent tracer particles in UV-light
photographs using Lab colour-space thresholds. It exports binary masks,
outlined diagnostic images, particle counts, particle concentrations, particle
areas, and optional summaries linked to a photo-reference database.

The default thresholds reproduce the original M4 workflow. Users should
validate or tune thresholds for their own camera, UV illumination, exposure,
tracer colour, and sediment background.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd


IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
OUTPUT_PREFIX = "OUTPUT_"


@dataclass
class Config:
    """Runtime settings for UV tracer image analysis."""

    l_threshold: int = 35
    a_threshold: int = 80
    b_threshold: int = 130
    disk_radius: int = 2
    pixel_area_mm2: float = 4.1698e-03
    bin_threshold: int = 128
    connectivity: int = 8
    min_area_px: int = 0
    output_prefix: str = OUTPUT_PREFIX
    save_masks: bool = True
    save_outlines: bool = True


def normalize_name_token(value: str) -> str:
    """Return a filesystem-friendly name token."""
    return str(value).strip().replace(" ", "_")


def detect_campaign_name(folder_path: str | Path) -> str:
    """Infer campaign name such as C1, C2, ... from the folder name."""
    folder_name = Path(folder_path).name
    match = re.search(r"(C\d+)", folder_name, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return normalize_name_token(folder_name)


def list_image_files(input_dir: str | Path) -> list[Path]:
    """List image files directly inside a folder."""
    input_path = Path(input_dir)
    return sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def is_output_folder(folder_path: str | Path, output_prefix: str = OUTPUT_PREFIX) -> bool:
    """Return True for generated output folders that should be ignored."""
    name = Path(folder_path).name.upper()
    return name.startswith(output_prefix.upper()) or name.startswith("MASK_") or name.startswith("OUTLINED_")


def folder_has_images(folder_path: str | Path, output_prefix: str = OUTPUT_PREFIX) -> bool:
    """Return True when a folder directly contains supported image files."""
    folder = Path(folder_path)
    if not folder.is_dir() or is_output_folder(folder, output_prefix=output_prefix):
        return False
    return any(p.is_file() and p.suffix.lower() in IMG_EXTS for p in folder.iterdir())


def find_image_folders(parent_dir: str | Path, recursive: bool = False, output_prefix: str = OUTPUT_PREFIX) -> list[Path]:
    """Find sample folders containing images under a parent folder."""
    parent = Path(parent_dir).resolve()
    if not parent.is_dir():
        raise RuntimeError(f"Parent folder not found: {parent}")

    folders: list[Path] = []
    if recursive:
        for root, dirnames, _ in os.walk(parent):
            dirnames[:] = [d for d in dirnames if not is_output_folder(d, output_prefix=output_prefix)]
            root_path = Path(root)
            if root_path == parent:
                continue
            if folder_has_images(root_path, output_prefix=output_prefix):
                folders.append(root_path)
    else:
        for child in sorted(parent.iterdir()):
            if child.is_dir() and folder_has_images(child, output_prefix=output_prefix):
                folders.append(child)

    return sorted(folders)


def build_output_paths(input_dir: str | Path, output_dir: Optional[str | Path], cfg: Config) -> dict[str, Path]:
    """Build output folders and workbook paths for one image folder."""
    input_path = Path(input_dir).resolve()
    root = Path(output_dir).resolve() if output_dir is not None else input_path
    folder_name = normalize_name_token(input_path.name)

    return {
        "mask_dir": root / f"{cfg.output_prefix}MASK_{folder_name}",
        "outline_dir": root / f"{cfg.output_prefix}OUTLINED_{folder_name}",
        "excel_tracer_counts": root / f"{cfg.output_prefix}TRACER_COUNTS_{folder_name}.xlsx",
        "excel_particle_areas": root / f"{cfg.output_prefix}PARTICLE_AREAS_{folder_name}.xlsx",
    }


def image_area_m2_from_pixels(pixel_count: int, cfg: Config) -> float:
    """Convert image pixel count to square metres."""
    return float(pixel_count) * float(cfg.pixel_area_mm2) * 1e-6


def particles_per_m2(n_particles: float, pixel_count: int, cfg: Config) -> float:
    """Particle count normalized by photographed area."""
    area_m2 = image_area_m2_from_pixels(pixel_count, cfg)
    if area_m2 <= 0:
        return np.nan
    return float(n_particles) / area_m2


def normalize_image_name(name: str) -> str:
    """Normalize image/mask names for summaries."""
    base = os.path.basename(str(name))
    base = os.path.splitext(base)[0]
    if base.lower().startswith("mask_"):
        base = base[5:]
    return base


def parse_pair_label(image_name: str) -> pd.Series:
    """Parse replicate labels ending in a/b from image names."""
    base = normalize_image_name(image_name)
    match = re.match(r"^(.*?)([ab])(?:_IMG.*)?$", base, flags=re.IGNORECASE)
    if not match:
        return pd.Series({"pair_label": base, "replicate": np.nan})

    prefix = match.group(1)
    replicate = match.group(2).lower()
    if not re.search(r"\d", prefix):
        return pd.Series({"pair_label": base, "replicate": np.nan})
    return pd.Series({"pair_label": prefix, "replicate": replicate})


def add_pair_columns(df: pd.DataFrame, image_col: str = "image_name") -> pd.DataFrame:
    """Add pair_label and replicate columns when image names support it."""
    out = df.copy()
    if image_col not in out.columns:
        out["pair_label"] = np.nan
        out["replicate"] = np.nan
        return out
    pair_info = out[image_col].apply(parse_pair_label)
    out["pair_label"] = pair_info["pair_label"]
    out["replicate"] = pair_info["replicate"]
    return out


def round_half_up(value):
    """Round half up, preserving NaN."""
    if pd.isna(value):
        return np.nan
    return int(np.floor(float(value) + 0.5))


def round_numeric_columns(df: pd.DataFrame, suffix: str = "_round1") -> pd.DataFrame:
    """Add rounded versions of numeric summary columns."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    count_like_cols = {"np_mean", "particle_count_mean", "particles_per_m2_mean"}
    for col in numeric_cols:
        new_col = f"{col}{suffix}"
        if col == "concentration_mean":
            out[new_col] = out[col]
        elif col in count_like_cols:
            out[new_col] = out[col].apply(round_half_up)
        else:
            out[new_col] = out[col].round(0)
    return out


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that contain only missing values."""
    if df is None or df.empty:
        return df
    return df.dropna(axis=1, how="all")


def disk_kernel(radius: int) -> np.ndarray:
    """Binary disk kernel similar to MATLAB strel('disk', r)."""
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return ((x * x + y * y) <= r * r).astype(np.uint8)


def bwfilter_sedphoto(bw_255: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """MATLAB SedPhoto-style morphology: erode, dilate, dilate, erode."""
    bw = cv2.erode(bw_255, kernel, iterations=1)
    bw = cv2.dilate(bw, kernel, iterations=1)
    bw = cv2.dilate(bw, kernel, iterations=1)
    bw = cv2.erode(bw, kernel, iterations=1)
    return bw


def segment_tracer_pixels(bgr: np.ndarray, kernel: np.ndarray, cfg: Config) -> np.ndarray:
    """Segment fluorescent tracer pixels in uint8 Lab colour space."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    bw = (
        (l_channel > int(cfg.l_threshold))
        & (a_channel > int(cfg.a_threshold))
        & (b_channel > int(cfg.b_threshold))
    ).astype(np.uint8) * 255
    return bwfilter_sedphoto(bw, kernel)


def count_particles_external_contours(bw_255: np.ndarray) -> int:
    """Count particles as external contours, equivalent to no-holes boundaries."""
    contours, _ = cv2.findContours(bw_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return int(len(contours))


def outline_image(bgr: np.ndarray, bw_255: np.ndarray) -> np.ndarray:
    """Draw detected external boundaries in white on the original image."""
    out = bgr.copy()
    contours, _ = cv2.findContours(bw_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 1)
    return out


def mask_filename_png(image_filename: str | Path) -> str:
    """Mask filename preserving source stem and using PNG for lossless masks."""
    return f"mask_{Path(image_filename).stem}.png"


def clean_image_name_from_mask(mask_name: str | Path) -> str:
    """Return original image stem from mask filename."""
    name = Path(mask_name).name
    if name.lower().startswith("mask_"):
        name = name[5:]
    return Path(name).stem


def extract_photo_number(image_name: str):
    """Extract the last integer from an image filename as photo number P."""
    base = Path(str(image_name)).stem
    numbers = re.findall(r"\d+", base)
    return int(numbers[-1]) if numbers else np.nan


def load_photo_reference_database(reference_excel: str | Path) -> pd.DataFrame:
    """Load an Excel database linking photo number P to C, S, G, x, y, SS and M."""
    sheets = pd.read_excel(reference_excel, sheet_name=None)
    ref = pd.concat(sheets.values(), ignore_index=True)
    required = ["C", "S", "G", "x", "y", "SS", "M", "P"]
    missing = [col for col in required if col not in ref.columns]
    if missing:
        raise ValueError("Reference Excel file is missing required columns: " + ", ".join(missing))
    ref = ref.copy()
    ref["P"] = pd.to_numeric(ref["P"], errors="coerce")
    ref = ref.dropna(subset=["P"])
    ref["P"] = ref["P"].astype(int)
    return ref


def add_photo_reference_columns(df: pd.DataFrame, reference_excel: Optional[str | Path]) -> pd.DataFrame:
    """Add C, S, G, x, y, SS and M to photo-level tables using image_name."""
    out = df.copy()
    if "image_name" not in out.columns:
        return out

    out["P"] = out["image_name"].apply(extract_photo_number)
    if reference_excel is None or out.empty:
        return out

    ref = load_photo_reference_database(reference_excel)
    out = out.dropna(subset=["P"])
    out["P"] = out["P"].astype(int)
    return out.merge(ref[["C", "S", "G", "x", "y", "SS", "M", "P"]], on="P", how="left")


def _flatten_stat_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in out.columns
    ]
    return out


def _round_unity_columns(df: pd.DataFrame, source_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in source_cols:
        if col in out.columns:
            out[f"{col}_round"] = out[col].apply(round_half_up)
    return out


def _first_or_join(series):
    values = pd.Series(series).dropna().unique().tolist()
    if not values:
        return np.nan
    return values[0] if len(values) == 1 else ",".join(map(str, values))


def build_sample_mean_from_counts(df_counts: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    """One-row mean summary for the whole image folder."""
    summary = pd.DataFrame([{
        "sample_name": sample_name,
        "n_photos": int(len(df_counts)),
        "np_mean": df_counts["np"].mean() if "np" in df_counts.columns and len(df_counts) else np.nan,
        "concentration_mean": df_counts["concentration"].mean() if "concentration" in df_counts.columns and len(df_counts) else np.nan,
        "particles_per_m2_mean": df_counts["particles_per_m2"].mean() if "particles_per_m2" in df_counts.columns and len(df_counts) else np.nan,
        "pix_particles_mean": df_counts["pix_particles"].mean() if "pix_particles" in df_counts.columns and len(df_counts) else np.nan,
        "pix_photo_mean": df_counts["pix_photo"].mean() if "pix_photo" in df_counts.columns and len(df_counts) else np.nan,
        "image_area_m2_mean": df_counts["image_area_m2"].mean() if "image_area_m2" in df_counts.columns and len(df_counts) else np.nan,
    }])
    return round_numeric_columns(summary)


def build_sample_count_stats_summary(
    df_counts: pd.DataFrame,
    sample_name: str,
    reference_excel: Optional[str | Path] = None,
) -> pd.DataFrame:
    """One-row median, mean, std, min and max summary for one image folder."""
    work = df_counts.copy()
    if reference_excel is not None and not work.empty:
        ref = load_photo_reference_database(reference_excel)
        work["P"] = work["image_name"].apply(extract_photo_number)
        work = work.dropna(subset=["P"])
        work["P"] = work["P"].astype(int)
        work = work.merge(ref[["C", "S", "G", "x", "y", "SS", "M", "P"]], on="P", how="left")

    row = {
        "sample_name": sample_name,
        "C": _first_or_join(work["C"]) if "C" in work else np.nan,
        "S": _first_or_join(work["S"]) if "S" in work else np.nan,
        "G": _first_or_join(work["G"]) if "G" in work else np.nan,
        "x": _first_or_join(work["x"]) if "x" in work else np.nan,
        "y": _first_or_join(work["y"]) if "y" in work else np.nan,
        "n_photos": int(len(work)),
        "n_subsamples": int(work["SS"].nunique(dropna=True)) if "SS" in work else np.nan,
        "photo_numbers": ",".join(map(str, sorted(work["P"].dropna().astype(int).unique().tolist()))) if "P" in work else np.nan,
    }
    for col in ["np", "concentration", "particles_per_m2", "pix_particles", "pix_photo", "image_area_m2"]:
        if col in work:
            values = pd.to_numeric(work[col], errors="coerce")
            row[f"{col}_median"] = values.median()
            row[f"{col}_mean"] = values.mean()
            row[f"{col}_std"] = values.std()
            row[f"{col}_min"] = values.min()
            row[f"{col}_max"] = values.max()

    out = pd.DataFrame([row])
    return _round_unity_columns(out, ["np_median", "np_mean", "pix_particles_median", "pix_particles_mean"])


def build_ss_count_summary(df_counts: pd.DataFrame, reference_excel: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge detected counts with reference database and summarize by subsample SS."""
    ref = load_photo_reference_database(reference_excel)
    work = df_counts.copy()
    work["P"] = work["image_name"].apply(extract_photo_number)
    work = work.dropna(subset=["P"])
    work["P"] = work["P"].astype(int)
    merged = work.merge(ref[["C", "S", "G", "x", "y", "SS", "M", "P"]], on="P", how="inner")
    if merged.empty:
        raise ValueError("No detected-photo rows matched the reference Excel database by P.")

    group_cols = ["C", "S", "G", "x", "y", "SS"]
    value_cols = [
        col for col in ["np", "concentration", "particles_per_m2", "pix_particles", "pix_photo", "image_area_m2"]
        if col in merged.columns
    ]
    stats = merged.groupby(group_cols, as_index=False, dropna=False)[value_cols].agg(["median", "mean", "std", "min", "max"]).reset_index()
    stats = _flatten_stat_columns(stats)
    photo_info = merged.groupby(group_cols, as_index=False, dropna=False).agg(
        n_photos=("image_name", "count"),
        photo_numbers=("P", lambda s: ",".join(map(str, sorted(s.astype(int).tolist())))),
    )
    ss_summary = photo_info.merge(stats, on=group_cols, how="left")
    ss_summary = _round_unity_columns(ss_summary, ["np_median", "np_mean"])
    preferred = group_cols + ["n_photos", "photo_numbers"]
    return merged, ss_summary[preferred + [col for col in ss_summary.columns if col not in preferred]]


def summarize_particle_areas(df_areas: pd.DataFrame, all_expected_images=None) -> pd.DataFrame:
    """Summarize particle areas per image."""
    if "image_name" not in df_areas.columns:
        df_areas = df_areas.copy()
        df_areas["image_name"] = pd.Series(dtype="object")

    work = add_pair_columns(df_areas, "image_name")
    if work.empty:
        summary = pd.DataFrame(columns=[
            "image_name", "pair_label", "replicate", "particle_count",
            "total_area_pixels", "total_area_mm2",
            "mean_area_pixels", "mean_area_mm2",
            "median_area_pixels", "median_area_mm2",
            "std_area_pixels", "std_area_mm2",
            "min_area_pixels", "min_area_mm2",
            "max_area_pixels", "max_area_mm2",
        ])
    else:
        summary = work.groupby(["image_name", "pair_label", "replicate"], dropna=False, sort=False).agg(
            particle_count=("particle_id", "count"),
            total_area_pixels=("area_pixels", "sum"),
            total_area_mm2=("area_mm2", "sum"),
            mean_area_pixels=("area_pixels", "mean"),
            mean_area_mm2=("area_mm2", "mean"),
            median_area_pixels=("area_pixels", "median"),
            median_area_mm2=("area_mm2", "median"),
            std_area_pixels=("area_pixels", "std"),
            std_area_mm2=("area_mm2", "std"),
            min_area_pixels=("area_pixels", "min"),
            min_area_mm2=("area_mm2", "min"),
            max_area_pixels=("area_pixels", "max"),
            max_area_mm2=("area_mm2", "max"),
        ).reset_index()

    if all_expected_images is not None:
        expected = pd.DataFrame({"image_name": [normalize_image_name(x) for x in all_expected_images]})
        expected = add_pair_columns(expected, "image_name")
        summary = expected.merge(summary, on=["image_name", "pair_label", "replicate"], how="left")
        fill_zero = [
            "particle_count", "total_area_pixels", "total_area_mm2",
            "mean_area_pixels", "mean_area_mm2", "median_area_pixels", "median_area_mm2",
            "std_area_pixels", "std_area_mm2", "min_area_pixels", "min_area_mm2",
            "max_area_pixels", "max_area_mm2",
        ]
        for col in fill_zero:
            if col in summary.columns:
                summary[col] = summary[col].fillna(0)
    return summary


def build_sample_area_stats_summary(
    df_particles: pd.DataFrame,
    image_summary: pd.DataFrame,
    sample_name: str,
    reference_excel: Optional[str | Path] = None,
) -> pd.DataFrame:
    """One-row summary for image-level and particle-level area attributes."""
    img = image_summary.copy()
    if reference_excel is not None and not img.empty:
        ref = load_photo_reference_database(reference_excel)
        img["P"] = img["image_name"].apply(extract_photo_number)
        img = img.dropna(subset=["P"])
        img["P"] = img["P"].astype(int)
        img = img.merge(ref[["C", "S", "G", "x", "y", "SS", "M", "P"]], on="P", how="left")

    row = {
        "sample_name": sample_name,
        "C": _first_or_join(img["C"]) if "C" in img else np.nan,
        "S": _first_or_join(img["S"]) if "S" in img else np.nan,
        "G": _first_or_join(img["G"]) if "G" in img else np.nan,
        "x": _first_or_join(img["x"]) if "x" in img else np.nan,
        "y": _first_or_join(img["y"]) if "y" in img else np.nan,
        "n_photos": int(len(image_summary)),
        "n_subsamples": int(img["SS"].nunique(dropna=True)) if "SS" in img else np.nan,
        "n_particles_total": int(len(df_particles)),
        "photo_numbers": ",".join(map(str, sorted(img["P"].dropna().astype(int).unique().tolist()))) if "P" in img else np.nan,
    }

    image_cols = [
        "particle_count", "total_area_pixels", "total_area_mm2",
        "mean_area_pixels", "mean_area_mm2", "median_area_pixels", "median_area_mm2",
        "std_area_pixels", "std_area_mm2", "min_area_pixels", "min_area_mm2",
        "max_area_pixels", "max_area_mm2",
    ]
    for col in image_cols:
        if col in image_summary:
            values = pd.to_numeric(image_summary[col], errors="coerce")
            row[f"image_{col}_median"] = values.median()
            row[f"image_{col}_mean"] = values.mean()
            row[f"image_{col}_std"] = values.std()
            row[f"image_{col}_min"] = values.min()
            row[f"image_{col}_max"] = values.max()

    for col in ["area_pixels", "area_mm2"]:
        if col in df_particles:
            values = pd.to_numeric(df_particles[col], errors="coerce")
            row[f"particle_{col}_median"] = values.median()
            row[f"particle_{col}_mean"] = values.mean()
            row[f"particle_{col}_std"] = values.std()
            row[f"particle_{col}_min"] = values.min()
            row[f"particle_{col}_max"] = values.max()

    out = pd.DataFrame([row])
    return _round_unity_columns(out, [
        "image_particle_count_median",
        "image_particle_count_mean",
        "particle_area_pixels_median",
        "particle_area_pixels_mean",
    ])


def build_ss_area_summary(
    df_particles: pd.DataFrame,
    image_summary: pd.DataFrame,
    reference_excel: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge area results with reference database and summarize by subsample SS."""
    ref = load_photo_reference_database(reference_excel)
    img = image_summary.copy()
    img["P"] = img["image_name"].apply(extract_photo_number)
    img = img.dropna(subset=["P"])
    img["P"] = img["P"].astype(int)
    img_with_ss = img.merge(ref[["C", "S", "G", "x", "y", "SS", "M", "P"]], on="P", how="inner")

    particles = df_particles.copy()
    if particles.empty:
        particles_with_ss = pd.DataFrame()
    else:
        particles["P"] = particles["image_name"].apply(extract_photo_number)
        particles = particles.dropna(subset=["P"])
        particles["P"] = particles["P"].astype(int)
        particles_with_ss = particles.merge(ref[["C", "S", "G", "x", "y", "SS", "M", "P"]], on="P", how="inner")

    if img_with_ss.empty:
        raise ValueError("No particle-area image rows matched the reference Excel database by P.")

    group_cols = ["C", "S", "G", "x", "y", "SS"]
    image_cols = [
        col for col in [
            "particle_count", "total_area_pixels", "total_area_mm2",
            "mean_area_pixels", "mean_area_mm2", "median_area_pixels", "median_area_mm2",
            "std_area_pixels", "std_area_mm2", "min_area_pixels", "min_area_mm2",
            "max_area_pixels", "max_area_mm2",
        ] if col in img_with_ss.columns
    ]
    image_stats = img_with_ss.groupby(group_cols, as_index=False, dropna=False)[image_cols].agg(
        ["median", "mean", "std", "min", "max"]
    ).reset_index()
    image_stats = _flatten_stat_columns(image_stats)
    photo_info = img_with_ss.groupby(group_cols, as_index=False, dropna=False).agg(
        n_photos=("image_name", "count"),
        photo_numbers=("P", lambda s: ",".join(map(str, sorted(s.astype(int).tolist())))),
    )
    ss_summary = photo_info.merge(image_stats, on=group_cols, how="left")

    if not particles_with_ss.empty:
        particle_cols = [col for col in ["area_pixels", "area_mm2"] if col in particles_with_ss.columns]
        particle_stats = particles_with_ss.groupby(group_cols, as_index=False, dropna=False)[particle_cols].agg(
            ["median", "mean", "std", "min", "max", "count"]
        ).reset_index()
        particle_stats = _flatten_stat_columns(particle_stats)
        rename = {col: f"particle_{col}" for col in particle_stats.columns if col not in group_cols}
        ss_summary = ss_summary.merge(particle_stats.rename(columns=rename), on=group_cols, how="left")

    ss_summary = _round_unity_columns(ss_summary, [
        "particle_count_median", "particle_count_mean",
        "particle_area_pixels_median", "particle_area_pixels_mean",
    ])
    preferred = group_cols + ["n_photos", "photo_numbers"]
    return particles_with_ss, img_with_ss, ss_summary[preferred + [col for col in ss_summary.columns if col not in preferred]]


def metadata_table(campaign_name: str, sample_name: str, cfg: Config) -> pd.DataFrame:
    """Build processing metadata for Excel workbooks."""
    return pd.DataFrame({
        "parameter": [
            "routine", "campaign_name", "sample_name",
            "l_threshold", "a_threshold", "b_threshold",
            "disk_radius", "pixel_area_mm2", "binary_threshold",
            "connectivity", "minimum_area_pixels", "particle_density_definition",
        ],
        "value": [
            "uv_tracer_particle_analysis", campaign_name, sample_name,
            cfg.l_threshold, cfg.a_threshold, cfg.b_threshold,
            cfg.disk_radius, cfg.pixel_area_mm2, cfg.bin_threshold,
            cfg.connectivity, cfg.min_area_px,
            "particle_count / (image_pixels * pixel_area_mm2 * 1e-6)",
        ],
    })


def save_tracer_counts_workbook(
    df_counts: pd.DataFrame,
    excel_path: str | Path,
    campaign_name: str,
    sample_name: str,
    cfg: Config,
    reference_excel: Optional[str | Path] = None,
) -> None:
    """Write tracer count workbook."""
    df_counts_out = drop_empty_columns(add_photo_reference_columns(df_counts, reference_excel))
    sample_mean = drop_empty_columns(build_sample_mean_from_counts(df_counts_out, sample_name=sample_name))
    sample_stats = drop_empty_columns(build_sample_count_stats_summary(df_counts_out, sample_name=sample_name))

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_counts_out.to_excel(writer, sheet_name="Tracer_Counts", index=False)
        sample_mean.to_excel(writer, sheet_name="Sample_Mean_Summary", index=False)
        sample_stats.to_excel(writer, sheet_name="Sample_Stats_Summary", index=False)

        if reference_excel is not None:
            _, ss_summary = build_ss_count_summary(df_counts, reference_excel)
            drop_empty_columns(ss_summary).to_excel(writer, sheet_name="SS_Count_Stats", index=False)

        metadata_table(campaign_name, sample_name, cfg).to_excel(writer, sheet_name="metadata", index=False)


def save_particle_areas_workbook(
    df_particles: pd.DataFrame,
    excel_path: str | Path,
    campaign_name: str,
    sample_name: str,
    cfg: Config,
    all_expected_images=None,
    reference_excel: Optional[str | Path] = None,
) -> None:
    """Write particle area workbook."""
    image_summary = summarize_particle_areas(df_particles, all_expected_images=all_expected_images)
    image_summary_out = drop_empty_columns(add_photo_reference_columns(image_summary, reference_excel))
    df_particles_out = drop_empty_columns(add_photo_reference_columns(df_particles, reference_excel))
    sample_area_stats = drop_empty_columns(
        build_sample_area_stats_summary(df_particles_out, image_summary_out, sample_name=sample_name)
    )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_particles_out.to_excel(writer, sheet_name="Particle_Areas", index=False)
        image_summary_out.to_excel(writer, sheet_name="Image_Summary", index=False)
        sample_area_stats.to_excel(writer, sheet_name="Sample_Area_Stats", index=False)

        if reference_excel is not None:
            _, _, ss_area_summary = build_ss_area_summary(df_particles, image_summary, reference_excel)
            drop_empty_columns(ss_area_summary).to_excel(writer, sheet_name="SS_Area_Stats", index=False)

        metadata_table(campaign_name, sample_name, cfg).to_excel(writer, sheet_name="metadata", index=False)


def run_tracer_count_analysis(
    input_dir: str | Path,
    mask_dir: str | Path,
    outline_dir: str | Path,
    excel_tracer_counts: str | Path,
    campaign_name: str,
    sample_name: str,
    cfg: Config,
    reference_excel: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Segment image files, save masks/outlines, and export count workbook."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise RuntimeError(f"Input folder not found: {input_dir}")

    files = list_image_files(input_dir)
    if not files:
        raise RuntimeError(f"No image files found in folder: {input_dir}")

    Path(mask_dir).mkdir(parents=True, exist_ok=True)
    Path(outline_dir).mkdir(parents=True, exist_ok=True)
    kernel = disk_kernel(cfg.disk_radius)
    rows = []

    print("\n" + "=" * 80)
    print("Running UV tracer-count analysis")
    print("Input folder:", input_dir)

    for image_path in files:
        print("Processing image:", image_path.name)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print("  skipped: image could not be read")
            continue

        height, width = bgr.shape[:2]
        pix_photo = int(height * width)
        bw = segment_tracer_pixels(bgr, kernel, cfg)
        pix_particles = int(np.sum(bw > 0))
        concentration = float(pix_particles / pix_photo)
        n_particles = count_particles_external_contours(bw)

        if cfg.save_masks:
            cv2.imwrite(str(Path(mask_dir) / mask_filename_png(image_path.name)), bw)
        if cfg.save_outlines:
            outlined = outline_image(bgr, bw)
            cv2.imwrite(str(Path(outline_dir) / f"outlined_{image_path.name}"), outlined)

        rows.append({
            "image_name": image_path.name,
            "np": n_particles,
            "concentration": concentration,
            "particles_per_m2": particles_per_m2(n_particles, pix_photo, cfg),
            "pix_particles": pix_particles,
            "pix_photo": pix_photo,
            "image_area_m2": image_area_m2_from_pixels(pix_photo, cfg),
        })
        print(f"  np={n_particles}  concentration={concentration:.6f}")

    df_counts = pd.DataFrame(rows)
    save_tracer_counts_workbook(
        df_counts,
        excel_tracer_counts,
        campaign_name=campaign_name,
        sample_name=sample_name,
        cfg=cfg,
        reference_excel=reference_excel,
    )
    print("Tracer-count workbook saved:", excel_tracer_counts)
    return df_counts


def run_particle_areas_from_masks(
    mask_dir: str | Path,
    excel_particle_areas: str | Path,
    campaign_name: str,
    sample_name: str,
    cfg: Config,
    reference_excel: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Measure connected-component particle areas from generated masks."""
    mask_dir = Path(mask_dir)
    if not mask_dir.is_dir():
        raise RuntimeError(f"Mask folder not found: {mask_dir}")

    mask_files = sorted([p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    if not mask_files:
        raise RuntimeError(f"No mask files found in: {mask_dir}")

    rows = []
    expected_images = []

    print("\nRunning particle-area analysis")
    print("Mask folder:", mask_dir)

    for mask_path in mask_files:
        img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Skipping unreadable mask:", mask_path.name)
            continue

        image_name = clean_image_name_from_mask(mask_path.name)
        expected_images.append(image_name)
        bw01 = (img >= int(cfg.bin_threshold)).astype(np.uint8)
        n_components, _, stats, _ = cv2.connectedComponentsWithStats(bw01, connectivity=int(cfg.connectivity))

        particle_id = 0
        for label in range(1, n_components):
            area_px = int(stats[label, cv2.CC_STAT_AREA])
            if area_px < int(cfg.min_area_px):
                continue
            particle_id += 1
            rows.append({
                "image_name": image_name,
                "particle_id": particle_id,
                "area_pixels": area_px,
                "area_mm2": area_px * float(cfg.pixel_area_mm2),
            })

    df_particles = pd.DataFrame(rows, columns=["image_name", "particle_id", "area_pixels", "area_mm2"])
    save_particle_areas_workbook(
        df_particles,
        excel_particle_areas,
        campaign_name=campaign_name,
        sample_name=sample_name,
        cfg=cfg,
        all_expected_images=expected_images,
        reference_excel=reference_excel,
    )
    print("Particle-area workbook saved:", excel_particle_areas)
    return df_particles


def process_one_folder(
    input_dir: str | Path,
    cfg: Config,
    output_dir: Optional[str | Path] = None,
    reference_excel: Optional[str | Path] = None,
) -> dict:
    """Process one folder containing image files."""
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise RuntimeError(f"Folder not found: {input_dir}")

    campaign_name = detect_campaign_name(input_dir)
    sample_name = input_dir.name
    paths = build_output_paths(input_dir, output_dir, cfg)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    run_tracer_count_analysis(
        input_dir=input_dir,
        mask_dir=paths["mask_dir"],
        outline_dir=paths["outline_dir"],
        excel_tracer_counts=paths["excel_tracer_counts"],
        campaign_name=campaign_name,
        sample_name=sample_name,
        cfg=cfg,
        reference_excel=reference_excel,
    )
    run_particle_areas_from_masks(
        mask_dir=paths["mask_dir"],
        excel_particle_areas=paths["excel_particle_areas"],
        campaign_name=campaign_name,
        sample_name=sample_name,
        cfg=cfg,
        reference_excel=reference_excel,
    )

    print(f"\nFinished successfully: {sample_name} ({campaign_name})")
    return {
        "folder": str(input_dir),
        "sample_name": sample_name,
        "campaign_name": campaign_name,
        "status": "ok",
        "tracer_counts_excel": str(paths["excel_tracer_counts"]),
        "particle_areas_excel": str(paths["excel_particle_areas"]),
    }


def write_batch_log(records: list[dict], parent_dir: str | Path, cfg: Config) -> None:
    """Write batch processing log."""
    if not records:
        return
    parent = Path(parent_dir).resolve()
    out = parent / f"{cfg.output_prefix}BATCH_PROCESSING_LOG.xlsx"
    pd.DataFrame(records).to_excel(out, index=False)
    print("\nBatch log saved:", out)


def process_folder_batch(
    parent_dir: str | Path,
    cfg: Config,
    output_dir: Optional[str | Path] = None,
    reference_excel: Optional[str | Path] = None,
    recursive: bool = False,
) -> list[dict]:
    """Process image-containing subfolders under a parent folder."""
    parent_dir = Path(parent_dir).resolve()
    folders = find_image_folders(parent_dir, recursive=recursive, output_prefix=cfg.output_prefix)
    if not folders:
        raise RuntimeError("No sample subfolders containing image files were found.")

    print("\n" + "=" * 80)
    print(f"Batch mode: found {len(folders)} folder(s) to process")
    for folder in folders:
        print("  -", folder)

    records: list[dict] = []
    for index, folder in enumerate(folders, start=1):
        print("\n" + "#" * 80)
        print(f"Processing folder {index}/{len(folders)}: {folder}")
        try:
            records.append(process_one_folder(folder, cfg=cfg, output_dir=output_dir, reference_excel=reference_excel))
        except Exception as exc:
            print("\nERROR while processing folder:", folder)
            print(type(exc).__name__ + ":", exc)
            records.append({
                "folder": str(folder),
                "sample_name": Path(folder).name,
                "campaign_name": detect_campaign_name(folder),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })

    write_batch_log(records, output_dir or parent_dir, cfg)
    return records


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        description="Identify and quantify fluorescent tracer particles in UV-light photographs."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help=(
            "Input folder. If it directly contains images, it is processed as one sample; "
            "otherwise image-containing subfolders are processed as samples."
        ),
    )
    parser.add_argument("--output-dir", default=None, help="Optional output root folder. Default: each input folder.")
    parser.add_argument("--reference-excel", default=None, help="Optional Excel database with columns C,S,G,x,y,SS,M,P.")
    parser.add_argument("--single-folder", action="store_true", help="Force processing only the selected folder.")
    parser.add_argument("--by-folders", action="store_true", help="Force batch mode over image-containing subfolders.")
    parser.add_argument("--recursive", action="store_true", help="Search nested subfolders in batch mode.")

    parser.add_argument("--l-threshold", type=int, default=35, help="Lab L channel threshold.")
    parser.add_argument("--a-threshold", type=int, default=80, help="Lab a channel threshold.")
    parser.add_argument("--b-threshold", type=int, default=130, help="Lab b channel threshold.")
    parser.add_argument("--disk-radius", type=int, default=2, help="Morphological disk radius in pixels.")
    parser.add_argument("--pixel-area-mm2", type=float, default=4.1698e-03, help="Area represented by one image pixel.")
    parser.add_argument("--bin-threshold", type=int, default=128, help="Binary mask threshold for area measurement.")
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=8, help="Connected-component connectivity.")
    parser.add_argument("--min-area-px", type=int, default=0, help="Minimum connected-component area in pixels.")
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX, help="Prefix for generated output folders/files.")
    parser.add_argument("--no-masks", action="store_true", help="Do not save binary mask images.")
    parser.add_argument("--no-outlines", action="store_true", help="Do not save outlined diagnostic images.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Command-line entry point."""
    args = build_parser().parse_args(argv)
    if args.single_folder and args.by_folders:
        raise SystemExit("Choose either --single-folder or --by-folders, not both.")

    cfg = Config(
        l_threshold=args.l_threshold,
        a_threshold=args.a_threshold,
        b_threshold=args.b_threshold,
        disk_radius=args.disk_radius,
        pixel_area_mm2=args.pixel_area_mm2,
        bin_threshold=args.bin_threshold,
        connectivity=args.connectivity,
        min_area_px=args.min_area_px,
        output_prefix=args.output_prefix,
        save_masks=not args.no_masks,
        save_outlines=not args.no_outlines,
    )

    folder = Path(args.folder).resolve()
    if args.single_folder:
        process_one_folder(folder, cfg=cfg, output_dir=args.output_dir, reference_excel=args.reference_excel)
    elif args.by_folders:
        process_folder_batch(
            folder,
            cfg=cfg,
            output_dir=args.output_dir,
            reference_excel=args.reference_excel,
            recursive=args.recursive,
        )
    elif folder_has_images(folder, output_prefix=cfg.output_prefix):
        process_one_folder(folder, cfg=cfg, output_dir=args.output_dir, reference_excel=args.reference_excel)
    else:
        process_folder_batch(
            folder,
            cfg=cfg,
            output_dir=args.output_dir,
            reference_excel=args.reference_excel,
            recursive=args.recursive,
        )


if __name__ == "__main__":
    main()
