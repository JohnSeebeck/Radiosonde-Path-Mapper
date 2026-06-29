"""
This module takes in raw radiosonde sounding data from a CSV file sourced
from https://weather.uwyo.edu/upperair/sounding.shtml, and converts it into a
3D graph showing where the balloon went, with annotations at each point showing
data for that point.
"""

import math
import random

import cartopy.crs as ccrs
import cartopy.feature as cfeat
import matplotlib.pyplot as plt
import mplcursors as mplc
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
)

# Global variables
labels = []


def update_annotation_text(sel):
    """Update annotation text for hovered data point."""
    index = sel.index
    sel.annotation.set_text(labels[index])


def file_import(file_path):
    """
    Read CSV file into a Pandas DataFrame and extract columns as lists.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    tuple
        The extracted columns as lists.
    """
    data_frame = pd.read_csv(file_path)

    x_data = data_frame["longitude"]
    y_data = data_frame["latitude"]
    z_data = data_frame["geopotential height_m"]
    pres = data_frame["pressure_hPa"]
    temp = data_frame["temperature_C"]
    dew = data_frame["dew point temperature_C"]
    humid = data_frame["relative humidity_%"]
    mix = data_frame["mixing ratio_g/kg"]
    wind_dir = data_frame["wind direction_degree"]
    wind_vel = data_frame["wind speed_m/s"]

    return (
        x_data,
        y_data,
        z_data,
        pres,
        temp,
        dew,
        humid,
        mix,
        wind_dir,
        wind_vel,
    )


def limit_extend(x_data, y_data, z_data):
    """
    Determine plot extent limits based on coordinate data.

    Returns a tuple containing:
    (map_z, top, east, west, north, south)
    """
    map_z = z_data[0]
    top = z_data[0]
    east = x_data[0]
    west = x_data[0]
    north = y_data[0]
    south = y_data[0]

    for point in range(len(x_data)):
        if map_z > z_data[point]:
            map_z = z_data[point]
        if top < z_data[point]:
            top = z_data[point]
        if east < x_data[point]:
            east = x_data[point]
        if west > x_data[point]:
            west = x_data[point]
        if north < y_data[point]:
            north = y_data[point]
        if south > y_data[point]:
            south = y_data[point]

    top = math.ceil(top / 1000) * 1000
    east = int(math.ceil(east))
    west = int(math.floor(west))
    north = int(math.ceil(north))
    south = int(math.floor(south))

    return map_z, top, east, west, north, south


def define_space(ax3d, map_z, top, east, west, north, south):
    """Set up 3D axis limits and labels."""
    lons = np.linspace(west, east, 100)
    lats = np.linspace(south, north, 100)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    map_z_arr = np.zeros_like(lon_grid)

    ax3d.plot_surface(
        lon_grid,
        lat_grid,
        map_z_arr,
        alpha=0,
        rcount=10,
        ccount=10,
    )

    ax3d.set_xlim(west, east)
    ax3d.set_ylim(south, north)
    ax3d.set_zlim(map_z, top)

    ax3d.set_xlabel("Longitude (degrees)")
    ax3d.set_ylabel("Latitude (degrees)")
    ax3d.set_zlabel("Elevation (meters)")


def draw_map_features(ax3d, fidelity):
    """Draw cartographic map features onto the 3D plot."""
    features = [
        cfeat.COASTLINE.with_scale(fidelity),
        cfeat.RIVERS.with_scale(fidelity),
        cfeat.STATES.with_scale(fidelity),
        cfeat.BORDERS.with_scale(fidelity),
        cfeat.LAKES.with_scale(fidelity),
    ]

    plate_carree = ccrs.PlateCarree()

    for feature in features:
        for geom in feature.geometries():
            coords_to_transform = []

            if isinstance(geom, LineString):
                coords_to_transform.append(geom.xy)

            elif isinstance(geom, MultiLineString):
                for line in geom.geoms:
                    coords_to_transform.append(line.xy)

            elif isinstance(geom, Polygon):
                coords_to_transform.append(geom.exterior.coords.xy)
                coords_to_transform.extend(
                    [interior.coords.xy for interior in geom.interiors]
                )

            elif isinstance(geom, MultiPolygon):
                for polygon in geom.geoms:
                    coords_to_transform.append(polygon.exterior.coords.xy)
                    coords_to_transform.extend(
                        [interior.coords.xy for interior in polygon.interiors]
                    )

            for x_raw, y_raw in coords_to_transform:
                try:
                    x, y, _ = plate_carree.transform_points(
                        plate_carree,
                        np.array(x_raw),
                        np.array(y_raw),
                    ).T
                except NotImplementedError:
                    continue

                z = np.zeros_like(x)
                ax3d.plot(x, y, z, color="black", linewidth=0.8)


def place_data(
    ax3d,
    x_data,
    y_data,
    z_data,
    pres,
    temp,
    dew,
    humid,
    mix,
    wind_dir,
    wind_vel,
):
    """Plot positional data and attach interactive annotations."""
    global labels

    scatter = ax3d.scatter(x_data, y_data, z_data)
    ax3d.plot(x_data, y_data, z_data)

    cursor = mplc.cursor(scatter, hover=True)

    labels = [
        (
            f"Geopotential Height = {z} meters\n"
            f"Longitude = {x:.2f} degrees\n"
            f"Latitude = {y:.2f} degrees\n"
            f"Pressure = {p:.0f} hPa\n"
            f"Temperature = {t:.0f} °C\n"
            f"Dew Point = {d:.0f} °C\n"
            f"Humidity = {h}%\n"
            f"Mixing Ratio = {m:.2f} g/kg\n"
            f"Wind = {wv:.0f} knots at {wd}°"
        )
        for x, y, z, p, t, d, h, m, wd, wv in zip(
            x_data,
            y_data,
            z_data,
            pres,
            temp,
            dew,
            humid,
            mix,
            wind_dir,
            wind_vel,
        )
    ]

    cursor.connect("add", update_annotation_text)


def generate_data():
    """
    Randomly generate nonsense test data.

    WARNING: The resulting dummy data is extremely derpy.
    Be prepared to laugh at the absurdity.
    """
    height = 0

    x_data = [round(random.uniform(-180.0, 180.0), 4)]
    y_data = [round(random.uniform(-90.0, 90.0), 4)]
    z_data = [0]

    pres, temp, dew = [], [], []
    humid, mix = [], []
    wind_dir, wind_vel = [], []

    while height < 35000:
        if height != 0:
            x_data.append(
                round(
                    random.uniform(x_data[0] - 0.5, x_data[0] + 0.5),
                    4,
                )
            )
            y_data.append(
                round(
                    random.uniform(y_data[0] - 0.5, y_data[0] + 0.5),
                    4,
                )
            )
            z_data.append(height)

        pres.append(round(random.uniform(0.0, 1050.0), 1))
        temp.append(round(random.uniform(-100.0, 60.0), 1))
        dew.append(round(random.uniform(-110.0, 50.0), 1))
        humid.append(random.randint(0, 100))
        mix.append(round(random.uniform(0.0, 30.0), 2))
        wind_dir.append(random.randint(0, 360))
        wind_vel.append(round(random.uniform(0.0, 150.0), 1))

        height += random.randint(5, 10)

    return (
        x_data,
        y_data,
        z_data,
        pres,
        temp,
        dew,
        humid,
        mix,
        wind_dir,
        wind_vel,
    )


def master_function(file_path, fidelity):
    """Master function to create the 3D radiosonde plot."""
    fig = plt.figure()
    ax3d = fig.add_subplot(projection="3d")

    if fidelity == "low":
        fidelity = "110m"
    elif fidelity == "medium":
        fidelity = "50m"
    elif fidelity == "high":
        fidelity = "10m"
    elif fidelity not in ("110m", "10m"):
        fidelity = "50m"

    if file_path == "dummy":
        (
            x_data,
            y_data,
            z_data,
            pres,
            temp,
            dew,
            humid,
            mix,
            wind_dir,
            wind_vel,
        ) = generate_data()
    else:
        (
            x_data,
            y_data,
            z_data,
            pres,
            temp,
            dew,
            humid,
            mix,
            wind_dir,
            wind_vel,
        ) = file_import(file_path)

    map_z, top, east, west, north, south = limit_extend(
        x_data, y_data, z_data
    )

    define_space(ax3d, map_z, top, east, west, north, south)
    draw_map_features(ax3d, fidelity)
    place_data(
        ax3d,
        x_data,
        y_data,
        z_data,
        pres,
        temp,
        dew,
        humid,
        mix,
        wind_dir,
        wind_vel,
    )

    plt.show()


if __name__ == "__main__":
    master_function("2025120512-72694.csv", "medium")
    master_function("dummy", "medium")