def bbox(vv_image_path,annotation_path,target_lat,target_lon,save_dir_path ):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import rasterio
    import xml.etree.ElementTree as ET
    from scipy.interpolate import griddata
    from skimage import exposure
    from scipy.ndimage import gaussian_filter
    from skimage.filters import median
    from skimage.morphology import disk

    output_image_path = os.path.join(save_dir_path,"output_bbox_image.tif")

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

    # Function to interpolate geolocation for a given lat/lon
    def latlon_to_pixel(lat, lon, grid_x, grid_y, lats, lons):
        pixel_x = griddata((lats, lons), grid_x, (lat, lon), method='linear')
        pixel_y = griddata((lats, lons), grid_y, (lat, lon), method='linear')
        
        # Check if the pixel values are NaN
        if np.isnan(pixel_x) or np.isnan(pixel_y):
            return None, None
        
        return int(pixel_x), int(pixel_y)

    # Load the full VV image
    with rasterio.open(vv_image_path) as vv_image:
        vv_data = vv_image.read(1)
        resolution = vv_image.res[0]  # Assuming square pixels
        transform = vv_image.transform  # Affine transformation

    # Extract geolocation grid from XML
    grid_x, grid_y, lats, lons = extract_geolocation(annotation_path)

    # Input the target latitude and longitude
   
     

    # Convert lat/lon to pixel coordinates
    pixel_x, pixel_y = latlon_to_pixel(target_lat, target_lon, grid_x, grid_y, lats, lons)

    if pixel_x is None or pixel_y is None:
        print("The specified latitude/longitude is outside the image area.")
    else:
        print(f"Corresponding pixel coordinates: ({pixel_x}, {pixel_y})")

        # Calculate the bounding box in pixel coordinates
        km_to_pixel = 500 / resolution
        min_x = max(0, int(pixel_x - km_to_pixel))
        max_x = min(vv_data.shape[1], int(pixel_x + km_to_pixel))
        min_y = max(0, int(pixel_y - km_to_pixel))
        max_y = min(vv_data.shape[0], int(pixel_y + km_to_pixel))

        print(f"BBox pixel coordinates: ({min_x}, {min_y}), ({max_x}, {max_y})")

        # Crop the image
        cropped_image1 = vv_data[min_y:max_y, min_x:max_x]

        cropped_image2 = exposure.equalize_hist(cropped_image1)

        cropped_image3 = exposure.equalize_hist(cropped_image2)

        cropped_image4 = exposure.equalize_hist(cropped_image3)

        cropped_image5 = exposure.equalize_hist(cropped_image4)

        cropped_image = exposure.equalize_hist(cropped_image5)

        # Save the cropped image to a new GeoTIFF file
        with rasterio.open(
            output_image_path,
            'w',
            driver='GTiff',
            height=cropped_image.shape[0],
            width=cropped_image.shape[1],
            count=1,
            dtype=cropped_image.dtype,
            crs=vv_image.crs,
            transform=rasterio.Affine(
                transform.a, transform.b, transform.c + min_x * transform.a,
                transform.d, transform.e, transform.f + min_y * transform.e
            ),
        ) as dst:
            dst.write(cropped_image, 1)

        print(f"Cropped image saved to {output_image_path}")
        return output_image_path
