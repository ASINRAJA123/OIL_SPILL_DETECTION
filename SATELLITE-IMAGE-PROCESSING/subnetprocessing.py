def subnet_processing(vv_path):
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import exposure
    import rasterio

    # Open the file and read as array
    with rasterio.open(vv_path) as vv_image:
        vv_data = vv_image.read(1)
        vv_profile = vv_image.profile  # Get the metadata of the original image

    # Apply histogram equalization to the entire image
    equalized_image = exposure.equalize_hist(vv_data)
    index=vv_path.find("\measurement")
    new_path=vv_path[:index]
    new_path+="\\"
    xy_path=new_path
    new_path+="processed_full_image.tif"

    # Save the processed image as a new .tif file
    output_path = new_path.encode('unicode_escape').decode()
    equalized_image = (equalized_image * 255).astype(np.uint8)  # Convert to uint8 for saving

    # Update metadata for the output image
    output_profile = vv_profile.copy()
    output_profile.update({
        "dtype": "uint8",
        "count": 1,  # single band
        "height": equalized_image.shape[0],
        "width": equalized_image.shape[1]
    })
    try:
        with rasterio.open(output_path, "w", **output_profile) as dest:
            dest.write(equalized_image, 1)
    except IOError:
            from PIL import Image  
            import numpy as np  
            import os  

            # Specify the directory where you want to save the image  
            output_directory = output_path  # Change this to your desired path  
            output_file_path = os.path.join(output_directory, 'dummy_image.tiff')  

            # Ensure the output directory exists  
            os.makedirs(output_directory, exist_ok=True)  

            # Create a dummy image (e.g., a black image)  
            width, height = 100, 100  
            data = np.zeros((height, width, 3), dtype=np.uint8)  # Black image  

            # Create Image object  
            image = Image.fromarray(data)  

            # Save as TIFF file to the specified path  
            image.save(output_file_path)  

            with rasterio.open(output_path, "w", **output_profile) as dest:
                dest.write(equalized_image, 1)

    print(f"Processed full image saved to {output_path}")

    # Load and display the saved image to confirm saving was successful
    with rasterio.open(output_path) as saved_image:
        saved_data = saved_image.read(1)

        # plt.figure(figsize=(10, 10))
        # plt.title("Saved Processed Full Image")
        # plt.imshow(saved_data, cmap='gray')
        # plt.colorbar()
        # plt.show()
    return output_path , xy_path


