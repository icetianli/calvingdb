"""
Create a glacier polygon mask from the glacier trace, fjord mask polygon,
and domain polygon. The glacier mask is used to calculate a Signed Distance
Transform for calving front prediction.
"""

import multiprocessing as mp

import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio import features
from shapely.geometry import LineString, Point
from shapely.ops import split, linemerge, unary_union
import xarray as xr
from tqdm import tqdm

from utils import list_directory, shrink_geometry

CALVING_DIR = "/path/to/Svalbard_ESSD_Calving_Product/Svalbard_Calving_Front_Product.gpkg"
TRAINING_DIR = "/path/to/calvdb/TrainingDataset"
RES = 30

_trace = None
_fjordmask = None
_domain = None
_centreline = None


def _init_worker():
    global _trace, _fjordmask, _domain, _centreline
    _trace = gpd.read_file(CALVING_DIR, layer="traces")
    _fjordmask = gpd.read_file(CALVING_DIR, layer="fjord_masks")
    _domain = gpd.read_file(CALVING_DIR, layer="domains")
    _centreline = gpd.read_file(CALVING_DIR, layer="centerlines")


def _process_image(args):
    glacier_id, image_id = args
    try:
        output_dir = f"{TRAINING_DIR}/{glacier_id}/{image_id}/"

        trace_i = _trace[
            (_trace["Glacier"] == glacier_id) & (_trace["ImageId"] == image_id)
        ]
        fjordmask_i = _fjordmask[_fjordmask["Glacier"] == glacier_id]
        centreline_i = _centreline[_centreline["Glacier"] == glacier_id]
        domain_i = _domain[_domain["Glacier"] == glacier_id].copy()
        domain_i["geometry"] = domain_i.geometry.apply(
            lambda geom: shrink_geometry(geom, shrink_m=300)
        )

        trace_geom = unary_union(trace_i.geometry)
        fjord_geom = unary_union(fjordmask_i.geometry)
        center_geom = unary_union(centreline_i.geometry)
        domain_geom = unary_union(domain_i.geometry)

        if trace_geom.geom_type == "MultiLineString":
            trace_geom = linemerge(trace_geom)
        if center_geom.geom_type == "MultiLineString":
            center_geom = linemerge(center_geom)

        coords = list(trace_geom.coords)
        p0, p1 = np.array(coords[0]), np.array(coords[-1])
        direction = p1 - p0
        unit = direction / np.linalg.norm(direction)

        minx, miny, maxx, maxy = fjord_geom.bounds
        ext = np.hypot(maxx - minx, maxy - miny)
        extended_line = LineString(
            [tuple(p0 - unit * ext)] + coords + [tuple(p1 + unit * ext)]
        )

        parts = split(fjord_geom, extended_line)
        glacier_pt = Point(list(center_geom.coords)[0])
        glacier_poly = min(parts.geoms, key=lambda p: p.distance(glacier_pt))
        glacier_poly = glacier_poly.intersection(domain_geom)

        minx, miny, maxx, maxy = domain_i.total_bounds
        width = int(np.ceil((maxx - minx) / RES))
        height = int(np.ceil((maxy - miny) / RES))
        transform = rio.transform.from_origin(minx, maxy, RES, RES)

        trace_mask = features.rasterize(
            shapes=[(glacier_poly, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
        )
        domain_mask = features.rasterize(
            shapes=[(domain_i.geometry.values[0], 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
        )

        xs = minx + (np.arange(width) + 0.5) * RES
        ys = maxy - (np.arange(height) + 0.5) * RES

        da = xr.DataArray(
            trace_mask * domain_mask,
            coords={"y": ys, "x": xs},
            dims=("y", "x"),
            name="calving_front",
        )
        da.rio.write_crs(domain_i.crs, inplace=True)
        da.rio.to_raster(output_dir + "Glacier_MASK.tif")

    except Exception as e:
        print(f"  [ERROR] {glacier_id}/{image_id}: {e}")


if __name__ == "__main__":
    tasks = []
    for glacier_id in list_directory(TRAINING_DIR):
        images = [
            img
            for img in list_directory(f"{TRAINING_DIR}/{glacier_id}/")
            if img != "Viz"
        ]
        tasks.extend((glacier_id, img) for img in images)

        print(f"Glacier {glacier_id}: {len(images)} images")

    n_workers = mp.cpu_count()
    print(f"Processing {len(tasks)} tasks across {n_workers} workers ...")

    with mp.Pool(n_workers, initializer=_init_worker) as pool:
        for _ in tqdm(pool.imap_unordered(_process_image, tasks), total=len(tasks)):
            pass

    print("Done.")
