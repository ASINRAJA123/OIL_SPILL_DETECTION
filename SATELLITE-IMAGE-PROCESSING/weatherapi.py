import os
import cdsapi

def weather_api(year, month, day, lat1, lon1, lat2, lon2,save_dir_path):
    url = "https://cds.climate.copernicus.eu/api"
    key = "786346a8-9ad8-4cd4-b066-b8459ddd0267"

    # Initialize the CDS API client with credentials
    client = cdsapi.Client(url=url, key=key)

    # Define dataset and request parameters
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": "reanalysis",
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
            "mean_wave_direction",
            "sea_surface_temperature",
            "significant_height_of_combined_wind_waves_and_swell",
            "total_precipitation"
        ],
        "year": [year],
        "month": [month],
        "day": [day],
        "time": [
            "00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"
        ],
        "area": [lat1, lon1, lat2, lon2],
        "format": "grib"  # Request data in GRIB format
    }

    # Retrieve and download data
    output_path =  os.path.join(save_dir_path, f"weather_data_{year}-{month}-{day}_history_new_near_final123456_check_NEW.grib")  # File path for saving the image
    client.retrieve(dataset, request).download(output_path)
    # print(output_path)
    return output_path