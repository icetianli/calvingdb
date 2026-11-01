"""
Shared helper functions for the CalvingDB preprocessing scripts.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import rioxarray
from pyproj import CRS, Transformer
from shapely import wkt
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

import re
from datetime import datetime


def parse_datetime_from_id(sample_id: str) -> datetime:
    """
    Parse a datetime from a sample ID in YYYYMMDD or YYYYMMDDTHHMMSS (Sentinel-2)
    format.
    """
    m = re.search(r"(\d{8})T(\d{6})", sample_id)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")

    m = re.search(r"(\d{8})", sample_id)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d")

    raise ValueError(f"Could not parse datetime from sample_id='{sample_id}'")


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def transform_coord(crs1, crs2, x, y):
    transformer = Transformer.from_crs(
        CRS("EPSG:" + str(crs1)), CRS("EPSG:" + str(crs2)), always_xy=True
    )
    return transformer.transform(x, y)


def convert_csv_to_gdf(csv_dir, crs="EPSG:3413"):
    """
    convert csv files into a geodataframe
    """

    df = pd.read_csv(csv_dir)
    df["geometry"] = df["geometry"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, crs=crs)

    return gdf


def list_directory(folder_path, ext=None):
    """
    Return the sub-folders of folder_path, or the files ending in ext when ext is given.
    """
    if ext is None:
        folders = [
            folder
            for folder in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, folder))
        ]
        return folders
    else:
        files = [file for file in os.listdir(folder_path) if file.endswith(ext)]
        return files


def create_new_dir(newdir):
    """
    create a new directory if does not exist
    """
    os.makedirs(newdir, exist_ok=True)


def split_list(datalist, processes):
    """
    Split datalist into roughly equal chunks for multiprocessing.
    """

    binsize = int(len(datalist) / processes)

    mp_proc_list = []

    for i in range(processes + 1):

        temp_list = []
        temp_list = datalist[i * binsize : (i + 1) * binsize]

        if not temp_list:
            break

        i += 1

        mp_proc_list.append(temp_list)

    return mp_proc_list, i


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def shrink_geometry(geometry, shrink_m=500):
    """
    Shrink a Polygon or MultiPolygon by a given distance (meters).
    """
    if geometry is None or geometry.is_empty:
        return None

    shrunk = geometry.buffer(-shrink_m)

    if shrunk.is_empty:
        return None

    return shrunk


def read_landmask(data_dir, bnds, crs) -> xr.Dataset:
    """Read the land mask and ice mask of each region as an xarray Dataset."""
    src = rioxarray.open_rasterio(data_dir)
    src = src.rio.reproject(crs)
    src_roi = src.sel(x=slice(bnds[0], bnds[1]), y=slice(bnds[2], bnds[3]))
    return src_roi


def compute_strain_rates_from_velocity(ds, vx_name="vx", vy_name="vy"):
    """
    Compute 2D strain rate components (exx, eyy, exy) and effective strain rate from vx,
    vy (m/yr) on a projected grid.
    """
    vx = ds[vx_name]
    vy = ds[vy_name]

    y = vx.coords["y"].values
    x = vx.coords["x"].values

    dvx_dy, dvx_dx = np.gradient(vx.values, y, x, edge_order=2)
    dvy_dy, dvy_dx = np.gradient(vy.values, y, x, edge_order=2)

    exx = dvx_dx
    eyy = dvy_dy
    exy = 0.5 * (dvx_dy + dvy_dx)

    eps_eff = np.sqrt(0.5 * (exx**2 + eyy**2) + exy**2)

    coords = {dim: vx.coords[dim] for dim in vx.dims}
    exx_da = xr.DataArray(exx, coords=coords, dims=vx.dims, name="exx")
    eyy_da = xr.DataArray(eyy, coords=coords, dims=vx.dims, name="eyy")
    exy_da = xr.DataArray(exy, coords=coords, dims=vx.dims, name="exy")
    eps_da = xr.DataArray(eps_eff, coords=coords, dims=vx.dims, name="eps_eff")

    for da in (exx_da, eyy_da, exy_da, eps_da):
        da.attrs["units"] = "1/yr"
        da.attrs["description"] = (
            "strain rate component"
            if da.name != "eps_eff"
            else "effective strain rate (2nd invariant)"
        )

    return xr.Dataset({"exx": exx_da, "eyy": eyy_da, "exy": exy_da, "eps_eff": eps_da})
