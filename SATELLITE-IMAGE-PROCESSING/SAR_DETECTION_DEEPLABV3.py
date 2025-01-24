import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import rasterio
from skimage.measure import label, regionprops
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.interpolate import griddata
import xml.etree.ElementTree as ET


# Initialize global variables
area_km2 = []
Compactness = []

def extract_geolocation(annotation_dir):
    xml_files = [f for f in os.listdir(annotation_dir) if f.endswith('.xml')]
    latitudes, longitudes, pixel_xs, pixel_ys = [], [], [], []

    for xml_file in xml_files:
        tree = ET.parse(os.path.join(annotation_dir, xml_file))
        root = tree.getroot()
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

# Function to interpolate pixel coordinates to geographic coordinates
def pixel_to_latlon(pixel_x, pixel_y, grid_x, grid_y, lats, lons):
    lat = griddata((grid_x, grid_y), lats, (pixel_x, pixel_y), method='linear')
    lon = griddata((grid_x, grid_y), lons, (pixel_x, pixel_y), method='linear')
    return lat, lon

# Function to mark lat/lon on image
def mark_latlon_on_image(target_lat, target_lon, vv_image_path, annotation_path):
    with rasterio.open(vv_image_path) as vv_image:
        vv_data = vv_image.read(1)
        resolution = vv_image.res[0]
        transform = vv_image.transform

    grid_x, grid_y, lats, lons = extract_geolocation(annotation_path)

    pixel_x, pixel_y = griddata((lats, lons), grid_x, (target_lat, target_lon), method='linear'), \
                        griddata((lats, lons), grid_y, (target_lat, target_lon), method='linear')

    if pixel_x is None or pixel_y is None:
        print("The specified latitude/longitude is outside the image area.")
        return
    else:
        pixel_x, pixel_y = int(pixel_x), int(pixel_y)
        print(f"Corresponding pixel coordinates: ({pixel_x}, {pixel_y})")

        # Plot the image and mark the target lat/lon
        plt.imshow(vv_data, cmap='gray')
        plt.colorbar(label='Backscatter Intensity')
        plt.scatter(pixel_x, pixel_y, color='red', marker='x', s=100, label='Target Lat/Lon')
        plt.legend()
        plt.title(f"Marking Lat/Lon: ({target_lat}, {target_lon})")
        plt.show()


def detection(image_path12, save_dir_path,target_lat,target_lon,annotation_path):
    model_path = r'F:\SIH_FINAL_AIS_SATE\Final_satellite_intergration - Copy\epoch_15.pth'  # Update this with your actual path
    global area_km2
    global Compactness
    
    # Load your trained model (DeepLabV3 with ResNet50 backbone)
    model = models.segmentation.deeplabv3_resnet50(weights=None)  # Use weights=None instead of pretrained=False
    model.classifier[4] = nn.Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))  # Adjusting output layer for 21 classes

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    model.eval()  # Set the model to evaluation mode

    device = torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    def preprocess_image(image_path12, input_size=(256, 256)):
        with rasterio.open(image_path12) as src:
            sar_data = src.read(1)
            sar_data_rescaled = (sar_data - sar_data.min()) / (sar_data.max() - sar_data.min()) * 255.0
            sar_image = Image.fromarray(sar_data_rescaled.astype(np.uint8))

        sar_image_rgb = np.stack([sar_data_rescaled] * 3, axis=-1).astype(np.uint8)

        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(Image.fromarray(sar_image_rgb)).unsqueeze(0)  
        return image, sar_data

    def extract_patch_features(sar_data, mask, resolution_meters=10):
        labeled_mask = label(mask)
        unique_labels = np.unique(labeled_mask)
        features = []

        global area_km2
        global Compactness

        for region_label in unique_labels[unique_labels > 0]:
            region_mask = labeled_mask == region_label

            area_pixels = np.sum(region_mask)
            area_km2_val = (area_pixels * resolution_meters**2) / 1e6  # Convert to km²
            props = regionprops(region_mask.astype(int))[0]

            perimeter = props.perimeter
            compactness = (perimeter**2) / (4 * np.pi * area_pixels) if area_pixels > 0 else 0
            elongation = props.major_axis_length / props.minor_axis_length if props.minor_axis_length > 0 else 0
            extent = props.extent
            shape_factor_1 = elongation
            shape_factor_2 = compactness

            region_values = sar_data[region_mask]
            mean_intensity = np.mean(region_values)
            std_intensity = np.std(region_values)
            power_to_mean_ratio = np.mean(region_values**2) / (mean_intensity**2 + 1e-6)

            surrounding_mask = np.logical_and(~region_mask, labeled_mask > 0)
            surrounding_values = sar_data[surrounding_mask]
            mean_surrounding = np.mean(surrounding_values) if surrounding_values.size > 0 else 0
            mean_contrast = abs(mean_intensity - mean_surrounding)
            max_contrast = np.max(abs(region_values - mean_surrounding)) if surrounding_values.size > 0 else 0
            local_contrast = std_intensity

            gradient = np.gradient(sar_data.astype(float))
            gradient_magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2)
            region_gradient = gradient_magnitude[region_mask]
            mean_gradient = np.mean(region_gradient)
            std_gradient = np.std(region_gradient)
            max_gradient = np.max(region_gradient)

            complexity = perimeter / np.sqrt(area_pixels) if area_pixels > 0 else 0

            mean_haralick_texture = mean_intensity / (std_intensity + 1e-6)

            perimeter_to_area = perimeter / (area_pixels + 1e-6)

            features.append({
                "Region": region_label,
                "Area (pixels)": area_pixels,
                "Area (km²)": area_km2_val,
                "Perimeter": perimeter,
                "Compactness": compactness,
                "Circularity": (4 * np.pi * area_pixels) / (perimeter**2) if perimeter > 0 else 0,
                "Elongation": elongation,
                "Extent": extent,
                "Shape Factor 1": shape_factor_1,
                "Shape Factor 2": shape_factor_2,
                "Mean Intensity": mean_intensity,
                "Std Intensity": std_intensity,
                "Power to Mean Ratio": power_to_mean_ratio,
                "Mean Contrast": mean_contrast,
                "Max Contrast": max_contrast,
                "Local Contrast": local_contrast,
                "Mean Gradient": mean_gradient,
                "Std Gradient": std_gradient,
                "Max Gradient": max_gradient,
                "Mean Haralick Texture": mean_haralick_texture,
                "Perimeter to Area": perimeter_to_area,
                "Complexity": complexity
            })

            # Update global variables for scatter plot
            area_km2.append(area_km2_val)
            Compactness.append(compactness)

        return features

    preprocessed_image, sar_data = preprocess_image(image_path12)
    preprocessed_image = preprocessed_image.to(device)

    with torch.no_grad():
        output = model(preprocessed_image)['out']

    output = output.squeeze(0)
    output = torch.argmax(output, 0)
    output = output.cpu().numpy()

    output_resized = resize(output, sar_data.shape, order=0, preserve_range=True, anti_aliasing=False).astype(int)

    if np.all(output_resized == 0):  
        print("No valid mask predicted. Skipping...")
        labeled_mask = np.zeros_like(output_resized)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].imshow(sar_data, cmap='gray')
        axs[0].set_title("SAR Backscatter Image")
        axs[0].axis('off')

        axs[1].imshow(labeled_mask, cmap='jet', alpha=0.5)
        axs[1].set_title("Predicted Mask")
        axs[1].axis('off')
        
        output_path = os.path.join(save_dir_path, "predicted_mask.jpg")
        plt.savefig(output_path, format='jpg', dpi=300)
        return None
    
    else:
        patch_features = extract_patch_features(sar_data, output_resized)

        non_mask_region = output_resized == 0
        non_mask_values = sar_data[non_mask_region]

        non_oil_spill_features = {
            "Region": "Non-Oil Spill",
            "Mean Backscatter": np.mean(non_mask_values) if non_mask_values.size > 0 else 0,
            "Std Backscatter": np.std(non_mask_values) if non_mask_values.size > 0 else 0
        }

        patch_features.append(non_oil_spill_features)

        features_df = pd.DataFrame(patch_features)

        output_csv_path = os.path.join(save_dir_path, "extracted_features.csv")
        features_df.to_csv(output_csv_path, index=False)

        print(f"Features saved to {output_csv_path}")

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].imshow(sar_data, cmap='gray')
        axs[0].set_title("SAR Backscatter Image")
        axs[0].axis('off')

        mean_backscatter = non_oil_spill_features["Mean Backscatter"]
        std_backscatter = non_oil_spill_features["Std Backscatter"]
        text = f"Non-Oil Spill:\nMean Backscatter: {mean_backscatter:.2f}\nStd Backscatter: {std_backscatter:.2f}"
        axs[0].text(10, 10, text, color='white', fontsize=10, ha='left', va='top', bbox=dict(facecolor='black', alpha=0.7))

        axs[1].imshow(output_resized, cmap='jet', alpha=0.5)
        axs[1].set_title("Predicted Mask")
        axs[1].axis('off')

        labeled_mask = label(output_resized)
        regions = regionprops(labeled_mask)

        for idx, region in enumerate(regions, start=1):
            y, x = region.centroid
            axs[1].text(x, y, f"Mask {idx}", color='white', fontsize=8, ha='center', va='center')

        plt.tight_layout()
        output_path = os.path.join(save_dir_path, "predicted_mask.jpg")
        plt.savefig(output_path, format='jpg', dpi=300)

        # plt.figure(figsize=(10, 6))
        # plt.scatter(area_km2, Compactness, c='blue', alpha=0.7)
        # for idx, feature in enumerate(patch_features):
        #     plt.text(area_km2[idx], Compactness[idx], f"Mask {feature['Region']}", fontsize=9)
                
        # plt.xlabel("Area (km²)")
        # plt.ylabel("Compactness")
        # plt.title("Scatter Plot of Area vs Compactness")
        # plt.grid(True)
        # plt.show()

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Create 3 columns for 3 images

        # Plot original SAR image
        axs[0].imshow(sar_data, cmap='gray')
        axs[0].set_title("Original SAR Image")
        axs[0].axis('off')

        # Plot image with lat/lon marked
        mark_latlon_on_image(target_lat, target_lon, image_path12, annotation_path)

        # Plot oil spill mask
        axs[2].imshow(output_resized, cmap='jet', alpha=0.5)
        axs[2].set_title("Oil Spill Mask")
        axs[2].axis('off')

        output_path = os.path.join(save_dir_path, "predicted_mask.jpg")
        plt.savefig(output_path, format='jpg', dpi=300)

        plt.tight_layout()
        plt.show()
        
        return output_csv_path
