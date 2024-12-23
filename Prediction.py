# Load necessary libraries
import numpy as np
import rasterio
import joblib
import os

def load_raster_and_metadata(filepath):
    with rasterio.open(filepath) as src:
        array = src.read()  # Read all bands
        profile = src.profile.copy()  # Get the profile
        descriptions = src.descriptions  # Get band descriptions
    return array, profile, descriptions

def save_classified_image(classified_image, output_path, reference_profile):
    output_profile = reference_profile.copy()
    output_profile.update({
        'count': 1,            # Single band
        'dtype': 'uint8',      # Binary classification (0 and 1)
        'nodata': 255          # NoData value
    })
    
    with rasterio.open(output_path, 'w', **output_profile) as dst:
        dst.write(classified_image.astype('uint8'), 1)

# Path to the saved model
model_path = r"B:\GreenAnt\FloodDetection\xgboost_binary_model.joblib"

# Path to the new imagery
new_imagery_path = r"B:\GreenAnt\FloodImages\Preprcess\Opava_2024.tif"

# Directory to save the classified output images
output_dir = r"B:\GreenAnt\FloodDetection"

# Load the saved model
print("Loading the trained model...")
best_model = joblib.load(model_path)

# Load the new imagery and metadata
print("Loading new imagery and metadata...")
new_imagery, new_profile, band_descriptions = load_raster_and_metadata(new_imagery_path)

# Validate band descriptions
if not all(band_descriptions):
    raise ValueError("Some bands in the TIFF file are missing descriptions. Please verify the input file.")

# Extract unique dates from band descriptions
dates = sorted(set([desc.split('_')[0] for desc in band_descriptions if '_' in desc]))

# Process each date
for date in dates:
    print(f"Processing for date: {date}")
    
    # Find VV and VH bands for the current date
    vv_band_index = band_descriptions.index(f"{date}_vv")
    vh_band_index = band_descriptions.index(f"{date}_vh")
    
    # Extract VV and VH bands
    vv_band = new_imagery[vv_band_index, :, :]
    vh_band = new_imagery[vh_band_index, :, :]
    
    # Combine VV and VH into a single feature array
    combined_bands = np.stack([vv_band, vh_band], axis=0)
    combined_bands_flat = combined_bands.reshape(2, -1).T  # Flatten for prediction

    # Make predictions
    print(f"Predicting flood for {date}...")
    predicted_classes = best_model.predict(combined_bands_flat)

    # Reshape predictions back to the original image shape
    predicted_image = predicted_classes.reshape(vv_band.shape)

    # Save the classified image
    output_path = os.path.join(output_dir, f"{date}_flood.tif")
    print(f"Saving classified flood image for {date}...")
    save_classified_image(predicted_image, output_path, new_profile)

    print(f"Flood image for {date} saved at: {output_path}")

# Verify completion
print("Flood detection completed for all dates.")
