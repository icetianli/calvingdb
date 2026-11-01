"""
Process every observation in the inventory into per-glacier zarr stores.
For each image it computes the signed distance transform (SDT) and stores the
imagery bands, geophysical layers, climate scalars, and calving front trace.
"""

import numpy as np
import pandas as pd
import zarr
import rasterio
from scipy.ndimage import distance_transform_edt
from pathlib import Path
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

INVENTORY_PATH = Path(
    "/path/to/calvdb//TrainingDataset_processed"
    "/inventory.parquet"
)
OUT_DIR = Path(
    "/path/to/calvdb//TrainingDataset_processed"
    "/zarr"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SDT_CLIP_M      = 2000.0
N_IMG_CHANNELS  = 5
D_GEOPHYS       = 6
D_CLIMATE       = 2

SENSOR_CODES    = {"S2": 0, "LE07": 1, "LC08": 2, "LC09": 3}

BAND_ROLES = {
    "S2":   {"blue": "B2",  "green": "B3", "red": "B4", "nir": "B8A", "swir1": "B11"},
    "LE07": {"blue": "B1",  "green": "B2", "red": "B3", "nir": "B4",  "swir1": "B5"},
    "LC08": {"blue": "B2",  "green": "B3", "red": "B4", "nir": "B5",  "swir1": "B6"},
    "LC09": {"blue": "B2",  "green": "B3", "red": "B4", "nir": "B5",  "swir1": "B6"},
}

IMG_CHANNEL_ORDER = ["blue", "green", "red", "nir", "swir1"]

BAND_FILENAME_PATTERNS = {
    "S2":   "{image_id}_{band}.tif",
    "LE07": "{image_id}.{band}.tif",
    "LC08": "{image_id}.{band}.tif",
    "LC09": "{image_id}.{band}.tif",
}

STATIC_FILES = {
    "bed_elevation":     "Bed_Elevation.tif",
    "strain_rate":       "Eff_Strain_Rate.tif",
    "ice_fjord_mask":    "Ice_Fjord_Mask.tif",
    "mar_runoff":        "MAR_Runoff.tif",
    "ocean_temp":        "Ocean_Temp.tif",
    "surface_elevation": "Surface_Elevation.tif",
    "thickness":         "Thickness.tif",
    "trace_mask":        "Trace_MASK.tif",
    "glacier_mask":      "Glacier_MASK.tif",
    "velocity_vx":       "Velocity_VX.tif",
    "velocity_vy":       "Velocity_VY.tif",
    "velocity":          "Velocity.tif",
}

GEOPHYS_CHANNEL_KEYS = [
    "bed_elevation",
    "surface_elevation",
    "thickness",
    "velocity",
    "strain_rate",
    "ice_fjord_mask",
]


def get_band_filename(image_id: str, band: str, sensor: str) -> str:
    return BAND_FILENAME_PATTERNS[sensor].format(image_id=image_id, band=band)


def get_static_path(image_dir: Path, key: str) -> Path:
    return image_dir / STATIC_FILES[key]


def load_grid_meta(image_dir: Path) -> dict | None:
    """
    Read raster geometry from Glacier_MASK.tif.
    """
    mask_path = get_static_path(image_dir, "glacier_mask")
    if not mask_path.exists():
        return None

    try:
        with rasterio.open(mask_path) as src:
            return {
                "height": src.height,
                "width": src.width,
                "resolution_m": float(abs(src.res[0])),
            }
    except Exception as e:
        print(f"    grid metadata error in {image_dir}: {e}")
        return None
    
def load_trace_mask(image_dir: Path, shape: tuple[int, int]) -> np.ndarray | None:
    """
    Load Trace_MASK.tif as a boolean array.
    """
    path = get_static_path(image_dir, "trace_mask")
    if not path.exists():
        return None

    try:
        with rasterio.open(path) as src:
            raw = src.read(1)
        return (raw > 0).astype(bool)
    except Exception as e:
        print(f"    trace_mask error in {image_dir}: {e}")
        return None


def compute_sdt(image_dir: Path, resolution_m: float) -> np.ndarray | None:
    """
    Compute the signed distance transform from Glacier_MASK.tif.

    positive_dist = distance_transform_edt(glacier_mask)
    negative_dist = distance_transform_edt(~glacier_mask)
    SDT = (positive_dist − negative_dist) × resolution_m
    """
    mask_path = get_static_path(image_dir, "glacier_mask")
    if not mask_path.exists():
        return None

    try:
        with rasterio.open(mask_path) as src:
            raw = src.read(1)

        glacier_mask = (raw > 0).astype(np.uint8)

        if glacier_mask.all() or not glacier_mask.any():
            return None

        dist_inside  = distance_transform_edt(glacier_mask).astype(np.float32)
        dist_outside = distance_transform_edt(
            ~glacier_mask.astype(bool)
        ).astype(np.float32)

        sdt = (dist_inside - dist_outside) * resolution_m
        sdt = np.clip(sdt, -SDT_CLIP_M, SDT_CLIP_M).astype(np.float32)
        return sdt

    except Exception as e:
        print(f"    SDT error in {image_dir}: {e}")
        return None


def load_band(
    image_dir: Path, image_id: str, band: str, sensor: str
) -> np.ndarray | None:
    """
    Load one spectral band GeoTIFF scaled to surface reflectance [0, 2].
    """
    fname = get_band_filename(image_id, band, sensor)
    path  = image_dir / fname
    if not path.exists():
        return None

    with rasterio.open(path) as src:
        data   = src.read(1).astype(np.float32)
        nodata = src.nodata

    data /= 10000.0

    if nodata is not None:
        data[data == nodata / 10000.0] = 0.0
    data[(data < 0.0) | (data > 2.0)] = 0.0

    return data


def load_raw_bands(
    image_dir: Path,
    image_id: str,
    sensor: str,
    shape: tuple[int, int],
) -> np.ndarray | None:
    """
    Load the five common training bands (blue, green, red, NIR, SWIR1) in fixed channel
    order.
    """
    roles = BAND_ROLES.get(sensor, {})
    bands = []

    for role in IMG_CHANNEL_ORDER:
        band_name = roles.get(role)
        if band_name is None:
            return None
        band_data = load_band(image_dir, image_id, band_name, sensor)
        if band_data is None:
            return None
        bands.append(band_data)

    return np.stack(bands, axis=0).astype(np.float32)


def load_static_raster(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """
    Load a single-band GeoTIFF as float32.
    """
    H, W = shape
    if not path.exists():
        return np.zeros((H, W), dtype=np.float32)

    with rasterio.open(path) as src:
        data   = src.read(1).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        data[data == nodata] = 0.0

    return data


def compute_geophys(image_dir: Path, shape: tuple[int, int]) -> np.ndarray:
    """
    Load six geophysical channels in the fixed order defined by GEOPHYS_CHANNEL_KEYS.
    """
    channels = [
        load_static_raster(get_static_path(image_dir, key), shape)
        for key in GEOPHYS_CHANNEL_KEYS
    ]
    return np.stack(channels, axis=0).astype(np.float32)


def extract_climate_scalars(image_dir: Path) -> np.ndarray | None:
    """
    Spatially average MAR_Runoff.tif and Ocean_Temp.tif to scalars.
    """
    values    = []
    any_found = False

    for key in ["mar_runoff", "ocean_temp"]:
        path = get_static_path(image_dir, key)
        if path.exists():
            with rasterio.open(path) as src:
                data   = src.read(1).astype(np.float32)
                nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            valid = data[np.isfinite(data)]
            values.append(float(valid.mean()) if len(valid) > 0 else 0.0)
            any_found = True
        else:
            values.append(0.0)

    return np.array(values, dtype=np.float32) if any_found else None


def preallocate_zarr_stores(inventory: pd.DataFrame) -> None:
    """
    Create one zarr store per glacier with all arrays pre-allocated.
    """
    for glacier_id, gdf in inventory.groupby("glacier_id"):
        store_path = OUT_DIR / f"{glacier_id}.zarr"

        required_arrays = {
            "sdt", "imagery", "geophys", "climate",
            "dates", "sensors", "valid_sdt", "valid_img", "valid_clm", "trace", "valid_trace"
        }

        if store_path.exists():
            try:
                existing_store = zarr.open_group(str(store_path), mode="r")
                existing_arrays = set(existing_store.array_keys())
                missing = sorted(required_arrays - existing_arrays)

                chunking_ok = False
                if not missing:
                    chunking_ok = (
                        existing_store["trace"].chunks[0] == 1
                        and existing_store["sdt"].chunks[0] == 1
                        and existing_store["imagery"].chunks[0] == 1
                        and existing_store["geophys"].chunks[0] == 1
                        and existing_store["climate"].chunks[0] == 1
                        and existing_store["dates"].chunks[0] == 1
                        and existing_store["sensors"].chunks[0] == 1
                        and existing_store["valid_trace"].chunks[0] == 1
                        and existing_store["valid_sdt"].chunks[0] == 1
                        and existing_store["valid_img"].chunks[0] == 1
                        and existing_store["valid_clm"].chunks[0] == 1
                    )

                if not missing and chunking_ok:
                    print(f"  {glacier_id}: zarr exists, schema ok, skipping pre-allocation")
                    continue
                print(
                    f"  {glacier_id}: zarr schema/chunk mismatch (missing {missing}) "
                    "— rebuilding"
                )
                shutil.rmtree(store_path)
            except Exception as e:
                print(f"  {glacier_id}: failed to validate existing zarr ({e}) — rebuilding")
                shutil.rmtree(store_path, ignore_errors=True)

        first_row = gdf.sort_values("date").iloc[0]
        grid = load_grid_meta(Path(first_row["image_dir"]))
        if grid is None:
            print(f"  {glacier_id}: no grid metadata — skipping")
            continue

        T = len(gdf)
        H = grid["height"]
        W = grid["width"]

        store = zarr.open_group(str(store_path), mode="w")

        store.create_dataset(
            "trace",
            shape=(T, H, W),
            chunks=(1, H, W),
            dtype=bool,
            fill_value=False,
        )

        store.create_dataset(
            "sdt",
            shape=(T, H, W),
            chunks=(1, H, W),
            dtype="f4",
            fill_value=0.0,
        )
        store.create_dataset(
            "imagery",
            shape=(T, N_IMG_CHANNELS, H, W),
            chunks=(1, N_IMG_CHANNELS, H, W),
            dtype="f4",
            fill_value=0.0,
        )
        store.create_dataset(
            "geophys",
            shape=(T, D_GEOPHYS, H, W),
            chunks=(1, D_GEOPHYS, H, W),
            dtype="f4",
            fill_value=0.0,
        )
        store.create_dataset(
            "climate",
            shape=(T, D_CLIMATE),
            chunks=(1, D_CLIMATE),
            dtype="f4",
            fill_value=0.0,
        )
        store.create_dataset(
            "dates",
            shape=(T,),
            chunks=(1,),
            dtype=str,
            fill_value="",
        )
        store.create_dataset(
            "sensors",
            shape=(T,),
            chunks=(1,),
            dtype="u1",
            fill_value=255,
        )
        store.create_dataset(
            "valid_trace",
            shape=(T,),
            chunks=(1,),
            dtype=bool,
            fill_value=False,
        )
        store.create_dataset(
            "valid_sdt",
            shape=(T,),
            chunks=(1,),
            dtype=bool,
            fill_value=False,
        )
        store.create_dataset(
            "valid_img",
            shape=(T,),
            chunks=(1,),
            dtype=bool,
            fill_value=False,
        )
        store.create_dataset(
            "valid_clm",
            shape=(T,),
            chunks=(1,),
            dtype=bool,
            fill_value=False,
        )

        store.attrs.update({
            "glacier_id":       glacier_id,
            "T":                T,
            "H":                H,
            "W":                W,
            "resolution_m":     grid["resolution_m"],
            "crs":              "EPSG:3995",
            "sdt_clip_m":       SDT_CLIP_M,
            "dates_format":     "YYYY-MM-DD",
            "img_channels":     IMG_CHANNEL_ORDER,
            "geophys_channels": GEOPHYS_CHANNEL_KEYS,
            "climate_channels": ["mar_runoff", "ocean_temp"],
            "trace_description": (
                "Rasterised calving front line from Trace_MASK.tif. "
                "True pixels lie on the calving front only — no fjord boundary. "
                "Used for evaluation metrics only, not as a model input."
            ),
        })

        print(f"  {glacier_id}: pre-allocated  T={T}  H={H}  W={W}")


def process_observation(args: tuple) -> dict:
    """
    Process a single image_id folder and write all arrays directly into the
    pre-allocated zarr store at the correct time index.
    """
    (glacier_id, image_id, image_dir_str,
     time_index, date_val, sensor,
     grid_meta, zarr_path_str) = args

    image_dir    = Path(image_dir_str)
    resolution_m = grid_meta["resolution_m"]
    H            = grid_meta["height"]
    W            = grid_meta["width"]
    shape        = (H, W)

    status = {
        "glacier_id": glacier_id,
        "image_id":   image_id,
        "time_index": time_index,
        "trace_ok":   False,
        "sdt_ok":     False,
        "img_ok":     False,
        "clm_ok":     False,
        "error":      None,
    }

    try:
        store = zarr.open_group(zarr_path_str, mode="r+")

        store["dates"][time_index]   = date_val.isoformat()
        store["sensors"][time_index] = SENSOR_CODES.get(sensor, 255)

        trace = load_trace_mask(image_dir, shape)
        if trace is not None:
            store["trace"][time_index]       = trace
            store["valid_trace"][time_index] = True
            status["trace_ok"]               = True

        sdt = compute_sdt(image_dir, resolution_m)
        if sdt is not None:
            store["sdt"][time_index]       = sdt
            store["valid_sdt"][time_index] = True
            status["sdt_ok"]               = True

        imagery = load_raw_bands(image_dir, image_id, sensor, shape)
        if imagery is not None:
            store["imagery"][time_index]   = imagery
            store["valid_img"][time_index] = True
            status["img_ok"]               = True

        store["geophys"][time_index] = compute_geophys(image_dir, shape)

        climate = extract_climate_scalars(image_dir)
        if climate is not None:
            store["climate"][time_index]   = climate
            store["valid_clm"][time_index] = True
            status["clm_ok"]               = True

    except Exception as e:
        status["error"] = str(e)

    return status


def main():
    inventory = pd.read_parquet(INVENTORY_PATH)

    usable = (
        inventory[inventory["has_glacier_mask"]]
        .sort_values(["glacier_id", "date"])
        .reset_index(drop=True)
    )

    print(f"Observations with Glacier_MASK.tif : {len(usable)}")
    print(f"Glaciers                            : {usable.glacier_id.nunique()}\n")

    print("Phase A — pre-allocating zarr stores...")
    preallocate_zarr_stores(usable)
    print()

    tasks = []
    for glacier_id, gdf in usable.groupby("glacier_id"):
        first_row = gdf.sort_values("date").iloc[0]
        grid = load_grid_meta(Path(first_row["image_dir"]))
        if grid is None:
            continue
        zarr_path = str(OUT_DIR / f"{glacier_id}.zarr")
        for time_index, (_, row) in enumerate(
            gdf.sort_values("date").iterrows()
        ):
            tasks.append((
                glacier_id,
                row["image_id"],
                row["image_dir"],
                time_index,
                row["date"],
                row["sensor"],
                grid,
                zarr_path,
            ))

    print(f"Phase B — processing {len(tasks)} observations...")

    results  = []
    n_workers = 8

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_observation, t): t for t in tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if result["error"]:
                print(
                    f"  ERROR  {result['glacier_id']}  "
                    f"t={result['time_index']:4d}  {result['error']}"
                )
            if i % 500 == 0 or i == len(tasks):
                print(f"  {i:5d} / {len(tasks)}")

    results_df = pd.DataFrame(results)
    n          = len(results_df)
    print(f"\nDone")
    print(f"  Trace ok  : {results_df.trace_ok.sum():5d} / {n}")
    print(f"  SDT ok    : {results_df.sdt_ok.sum():5d} / {n}")
    print(f"  Imagery ok: {results_df.img_ok.sum():5d} / {n}")
    print(f"  Climate ok: {results_df.clm_ok.sum():5d} / {n}")
    print(f"  Errors    : {results_df.error.notna().sum():5d}")

    log_path = Path(str(OUT_DIR).replace("/zarr", "")) / "processing_log.parquet"
    results_df.to_parquet(log_path, index=False)

    errors = results_df[results_df.error.notna()]
    if len(errors):
        err_path = OUT_DIR / "processing_errors.csv"
        errors.to_csv(err_path, index=False)
        print(f"  Error log : {err_path}")


if __name__ == "__main__":
    main()
