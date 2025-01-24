import os
import numpy as np
import rasterio
import xml.etree.ElementTree as ET
from scipy.interpolate import griddata

def location_extraction(cropped_image_path, annotation_path):
    # Function to extract geolocation grid from SAFE XML file
    def extract_geolocation(annotation_dir):
        xml_files = [f for f in os.listdir(annotation_dir) if f.endswith('.xml')]
        latitudes, longitudes, pixel_xs, pixel_ys = [], [], [], []

        for xml_file in xml_files:
            tree = ET.parse(os.path.join(annotation_dir, xml_file))
            root = tree.getroot()

            # Extract geolocation grid points
            for geolocation_grid in root.findall('.//geolocationGridPointList/geolocationGridPoint'):
                lat = float(geolocation_grid.find('latitude').text)
                lon = float(geolocation_grid.find('longitude').text)
                pixel_x = int(geolocation_grid.find('pixel').text)
                pixel_y = int(geolocation_grid.find('line').text)

                latitudes.append(lat)
                longitudes.append(lon)
                pixel_xs.append(pixel_x)
                pixel_ys.append(pixel_y)

        return np.array(pixel_xs), np.array(pixel_ys), np.array(latitudes), np.array(longitudes)

    # Function to interpolate geolocation for a given pixel
    def pixel_to_latlon(pixel_x, pixel_y, grid_x, grid_y, lats, lons):
        lat = griddata((grid_x, grid_y), lats, (pixel_x, pixel_y), method='linear')
        lon = griddata((grid_x, grid_y), lons, (pixel_x, pixel_y), method='linear')
        return lat, lon

    # Extract geolocation grid from XML
    grid_x, grid_y, lats, lons = extract_geolocation(annotation_path)

    # Load the cropped image
    with rasterio.open(cropped_image_path) as cropped_image:
        cropped_transform = cropped_image.transform
        width = cropped_image.width
        height = cropped_image.height

    # Compute global pixel coordinates for the image corners
    global_x1, global_y1 = cropped_transform * (0, 0)  # Top-left corner
    global_x2, global_y2 = cropped_transform * (width, height)  # Bottom-right corner

    # Interpolate lat/lon for the corners
    lat1, lon1 = pixel_to_latlon(global_x1, global_y1, grid_x, grid_y, lats, lons)
    lat2, lon2 = pixel_to_latlon(global_x2, global_y2, grid_x, grid_y, lats, lons)

    print("Bounding Box Coordinates (latitude/longitude):")
    print(f"Top-Left: ({lat1}, {lon1})")
    print(f"Bottom-Right: ({lat2}, {lon2})")

    return lat1, lon1, lat2, lon2
