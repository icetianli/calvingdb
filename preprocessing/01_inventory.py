"""
Walk the raw dataset directory and parse each image folder name to extract the glacier
ID, sensor, and acquisition date. The results are written to a flat inventory table
used by the later preprocessing steps.
"""

import re
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

RAW_DIR  = Path("/path/to/calvdb/TrainingDataset")
OUT_DIR  = Path("/path/to/calvdb/TrainingDataset_processed")

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "inventory.parquet"

SENSOR_BANDS = {
    "S2": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B9", "B11", "B12"],
    "LE07": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B10", "B11"],
    "LC08": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B10", "B11"],
    "LC09": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B10", "B11"],
}

BAND_ROLES = {
    "S2": {"blue": "B2", "green": "B3", "red": "B4", "nir": "B8A", "swir1": "B11"},
    "LE07": {"blue": "B1", "green": "B2", "red": "B3", "nir": "B4", "swir1": "B5"},
    "LC08": {"blue": "B2", "green": "B3", "red": "B4", "nir": "B5", "swir1": "B6"},
    "LC09": {"blue": "B2", "green": "B3", "red": "B4", "nir": "B5", "swir1": "B6"},
}

COMMON_TRAINING_ROLES = ["blue", "green", "red", "nir", "swir1"]

STATIC_FILES = {
    "bed_elevation": "Bed_Elevation.tif",
    "strain_rate": "Eff_Strain_Rate.tif",
    "ice_fjord_mask": "Ice_Fjord_Mask.tif",
    "mar_runoff": "MAR_Runoff.tif",
    "ocean_temp": "Ocean_Temp.tif",
    "surface_elevation": "Surface_Elevation.tif",
    "thickness": "Thickness.tif",
    "trace_mask": "Trace_MASK.tif",
    "glacier_mask": "Glacier_MASK.tif",
    "velocity_vx": "Velocity_VX.tif",
    "velocity_vy": "Velocity_VY.tif",
    "velocity": "Velocity.tif",
}

S2_PATTERN = re.compile(r"^(\d{8})T\d{6}_\d{8}T\d{6}_(T[A-Z0-9]{5})$")

LS_PATTERN = re.compile(r"^(LC08|LC09|LE07)_(\d{6})_(\d{8})$")


def parse_image_id(image_id: str) -> dict | None:
    """
    Parse a Sentinel-2 or Landsat image folder name into structured metadata.
    """
    m = S2_PATTERN.match(image_id)
    if m:
        try:
            acq_date = datetime.strptime(m.group(1), "%Y%m%d").date()
        except ValueError:
            return None
        return {
            "sensor": "S2",
            "date": acq_date,
            "tile_or_pathrow": m.group(2),
        }

    m = LS_PATTERN.match(image_id)
    if m:
        sensor = m.group(1)
        path_row = m.group(2)
        try:
            acq_date = datetime.strptime(m.group(3), "%Y%m%d").date()
        except ValueError:
            return None
        return {
            "sensor": sensor,
            "date": acq_date,
            "tile_or_pathrow": path_row,
        }

    return None


BAND_FILENAME_PATTERNS = {
    "S2": "{image_id}_{band}.tif",
    "LE07": "{image_id}.{band}.tif",
    "LC08": "{image_id}.{band}.tif",
    "LC09": "{image_id}.{band}.tif",
}


def get_band_filename(image_id: str, band: str, sensor: str) -> str:
    """
    Construct the expected filename for a given band, respecting the sensor-specific
    naming convention.
    """
    pattern = BAND_FILENAME_PATTERNS.get(sensor)
    if pattern is None:
        raise ValueError(f"Unknown sensor: {sensor}")
    return pattern.format(image_id=image_id, band=band)


def check_bands(image_dir: Path, image_id: str, sensor: str) -> dict:
    """
    Check presence of every expected band file for this sensor.
    """
    expected_bands = SENSOR_BANDS.get(sensor, [])
    roles = BAND_ROLES.get(sensor, {})

    per_band = {}
    for band in expected_bands:
        fname = get_band_filename(image_id, band, sensor)
        per_band[f"band_{band}"] = (image_dir / fname).is_file()

    common_band_names = [roles[role] for role in COMMON_TRAINING_ROLES if role in roles]
    has_common = all(per_band.get(f"band_{b}", False) for b in common_band_names)

    swir1_name = roles.get("swir1")
    has_swir1 = per_band.get(f"band_{swir1_name}", False) if swir1_name else False

    return {
        **per_band,
        "has_common_bands": has_common,
        "has_swir1": has_swir1,
        "n_bands_present": sum(per_band.values()),
        "n_bands_expected": len(expected_bands),
    }


def check_static_files(image_dir: Path) -> dict:
    """
    Check which static geophysical files are present.
    """
    result = {}
    for key, fname in STATIC_FILES.items():
        result[f"has_{key}"] = (image_dir / fname).is_file()

    result["has_trace"] = result["has_glacier_mask"]
    result["has_geophys"] = all(
        [
            result["has_bed_elevation"],
            result["has_surface_elevation"],
            result["has_thickness"],
            result["has_ice_fjord_mask"],
        ]
    )
    result["has_velocity"] = all(
        [
            result["has_velocity_vx"],
            result["has_velocity_vy"],
            result["has_velocity"],
            result["has_strain_rate"],
        ]
    )
    result["has_climate"] = all(
        [
            result["has_mar_runoff"],
            result["has_ocean_temp"],
        ]
    )
    return result


def main():
    records = []
    skipped = []

    glacier_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())
    print(f"Found {len(glacier_dirs)} glacier directories under {RAW_DIR}\n")

    for glacier_dir in glacier_dirs:
        glacier_id = glacier_dir.name
        image_dirs = sorted(d for d in glacier_dir.iterdir() if d.is_dir())

        for image_dir in image_dirs:
            image_id = image_dir.name

            parsed = parse_image_id(image_id)
            if parsed is None:
                skipped.append(
                    {
                        "glacier_id": glacier_id,
                        "image_id": image_id,
                        "reason": "unrecognised image_id format",
                    }
                )
                continue

            sensor = parsed["sensor"]

            band_info = check_bands(image_dir, image_id, sensor)
            static_info = check_static_files(image_dir)

            records.append(
                {
                    "glacier_id": glacier_id,
                    "image_id": image_id,
                    "image_dir": str(image_dir),
                    "date": parsed["date"],
                    "sensor": sensor,
                    "tile_or_pathrow": parsed["tile_or_pathrow"],
                    **band_info,
                    **static_info,
                }
            )

    df = pd.DataFrame(records)

    if df.empty:
        print("WARNING: no valid observations found — check RAW_DIR")
        return

    df = df.sort_values(["glacier_id", "date"]).reset_index(drop=True)

    df["fully_usable"] = df["has_trace"] & df["has_common_bands"]

    print("=" * 60)
    print("INVENTORY SUMMARY")
    print("=" * 60)
    print(f"  Total observations      : {len(df)}")
    print(f"  Glaciers                : {df.glacier_id.nunique()}")
    print(f"  Date range              : {df.date.min()} → {df.date.max()}")
    print()
    print(f"  Sensor breakdown:")
    for sensor, count in df.sensor.value_counts().items():
        print(f"    {sensor:6s}  {count:5d}")
    print()
    print(f"  has calving front trace : {df.has_trace.sum():5d}")
    print(f"  has common bands (BGRN) : {df.has_common_bands.sum():5d}")
    print(f"  has SWIR1 (for NDSI)    : {df.has_swir1.sum():5d}")
    print(f"  has geophys channels    : {df.has_geophys.sum():5d}")
    print(f"  has climate variables   : {df.has_climate.sum():5d}")
    print(f"  has velocity fields     : {df.has_velocity.sum():5d}")
    print(f"  fully usable (trace+BGRN): {df.fully_usable.sum():5d}")
    print()
    print(f"  Skipped (bad image_id)  : {len(skipped)}")

    per_glacier = (
        df.groupby("glacier_id")
        .agg(
            n_obs=("image_id", "count"),
            n_with_trace=("has_trace", "sum"),
            n_with_bands=("has_common_bands", "sum"),
            n_fully_usable=("fully_usable", "sum"),
            n_with_geophys=("has_geophys", "sum"),
            date_min=("date", "min"),
            date_max=("date", "max"),
            sensors=("sensor", lambda x: sorted(x.unique())),
        )
        .reset_index()
    )

    per_glacier["sufficient_obs"] = per_glacier["n_fully_usable"] >= 6
    print(
        f"  Glaciers with ≥6 usable obs : "
        f"{per_glacier.sufficient_obs.sum()} / {len(per_glacier)}"
    )
    print("=" * 60)

    df.to_parquet(OUT_PATH, index=False)
    per_glacier.to_parquet(OUT_DIR / "glacier_summary.parquet", index=False)

    if skipped:
        pd.DataFrame(skipped).to_csv(OUT_DIR / "skipped_image_ids.csv", index=False)

    schema = {
        "sensor_bands": SENSOR_BANDS,
        "band_roles": BAND_ROLES,
        "common_training_roles": COMMON_TRAINING_ROLES,
        "static_files": STATIC_FILES,
        "band_filename_patterns": BAND_FILENAME_PATTERNS,
        "sensors_supported": ["S2", "LE07", "LC08", "LC09"],
    }

    json.dump(schema, open(OUT_DIR / "file_schema.json", "w"), indent=2)

    print(f"\nOutputs written to {OUT_DIR}")
    print(f"  inventory.parquet")
    print(f"  glacier_summary.parquet")
    print(f"  file_schema.json")
    if skipped:
        print(f"  skipped_image_ids.csv")


if __name__ == "__main__":
    main()
