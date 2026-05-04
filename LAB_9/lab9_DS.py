from pathlib import Path
from itertools import combinations

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from pyproj import Geod


input_file = "Fire_Station.csv"
count_stations = 38
output_folder = Path("output_gis_fire")
output_folder.mkdir(exist_ok=True)

df = pd.read_csv(input_file)
df = df.head(count_stations)

df["LATITUDE"] = pd.to_numeric(df["LATITUDE"])
df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"])

geometry = [Point(xy) for xy in zip(df["LONGITUDE"], df["LATITUDE"])]

stations = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

geod = Geod(ellps="WGS84")

rows = []

for i, j in combinations(stations.index, 2):
    station_a = stations.loc[i]
    station_b = stations.loc[j]

    lon1 = station_a["LONGITUDE"]
    lat1 = station_a["LATITUDE"]
    lon2 = station_b["LONGITUDE"]
    lat2 = station_b["LATITUDE"]

    azimuth1, azimuth2, distance_m = geod.inv(lon1, lat1, lon2, lat2)

    rows.append({
        "station_a": station_a["NAME"],
        "station_b": station_b["NAME"],
        "distance_m": distance_m,
        "distance_km": distance_m / 1000,
        "distance_mi": distance_m / 1609.344,
        "geometry": LineString([station_a.geometry, station_b.geometry])
    })

distance_grid = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

station_names = list(stations["NAME"])
distance_matrix = pd.DataFrame(0.0, index=station_names, columns=station_names)

for index, row in distance_grid.iterrows():
    distance_matrix.loc[row["station_a"], row["station_b"]] = row["distance_km"]
    distance_matrix.loc[row["station_b"], row["station_a"]] = row["distance_km"]

distance_matrix = distance_matrix.round(3)

km_check = distance_grid["distance_m"] / 1000
mi_check = distance_grid["distance_m"] / 1609.344

max_km_error = abs(km_check - distance_grid["distance_km"]).max()
max_mi_error = abs(mi_check - distance_grid["distance_mi"]).max()

projected_distance = distance_grid.to_crs("EPSG:5070").length
projection_difference = abs(projected_distance - distance_grid["distance_m"]) / distance_grid["distance_m"] * 100
mean_projection_difference = projection_difference.mean()

stations.to_file(output_folder / "stations.geojson", driver="GeoJSON")
distance_grid.to_file(output_folder / "distance_grid.geojson", driver="GeoJSON")
distance_grid.drop(columns="geometry").to_csv(output_folder / "distance_pairs.csv", index=False)
distance_matrix.to_csv(output_folder / "distance_matrix_km.csv")

fig, ax = plt.subplots(figsize=(12, 8))

distance_grid.plot(ax=ax, linewidth=0.7, alpha=0.45)
stations.plot(ax=ax, markersize=45)

for index, row in stations.iterrows():
    ax.annotate(
        row["NAME"],
        xy=(row.geometry.x, row.geometry.y),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=6
    )

ax.set_title("Distance Grid Between Montgomery County Fire Stations")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linewidth=0.3)

plt.tight_layout()
plt.savefig(output_folder / "distance_grid_map.png", dpi=300)
plt.show()

print("=== GIS FIRE DISTANCE BACKEND RESULT ===")
print(f"Number of stations: {len(stations)}")
print(f"Number of distance grid lines: {len(distance_grid)}")
print(f"Average distance: {distance_grid['distance_km'].mean():.3f} km")
print(f"Minimum distance: {distance_grid['distance_km'].min():.3f} km")
print(f"Maximum distance: {distance_grid['distance_km'].max():.3f} km")

print()
print("Unit verification:")
print(f"Max error meters to kilometers: {max_km_error:.12f}")
print(f"Max error meters to miles: {max_mi_error:.12f}")
print(f"Mean geodesic vs EPSG:5070 difference: {mean_projection_difference:.3f}%")

print()
print(f"Files saved to folder: {output_folder.resolve()}")