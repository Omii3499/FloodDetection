# Load necessary libraries
import numpy as np
import rasterio
import joblib

def load_raster_as_array(filepath):
    with rasterio.open(filepath) as src:
        array = src.read()
        profile = src.profile.copy()
    return array, profile

def save_classified_image(classified_image, output_path, reference_profile):
    output_profile = reference_profile.copy()
    output_profile.update({
        'count': 1,            # Single band
        'dtype': 'uint8',      # Binary classification (0 and 1)
        'nodata': 255         # NoData value
    })
    
    with rasterio.open(output_path, 'w', **output_profile) as dst:
        dst.write(classified_image.astype('uint8'), 1)

# Path to the saved model
model_path = r"B:\GreenAnt\FloodDetection\xgboost_binary_model.joblib"

# Path to the new imagery
new_imagery_path = r"C:\Users\admin\Downloads\18977ba0-beea-4eee-b4a2-32f5770c0ac8.tiff"

# Path to save the classified output
output_classified_path = r"B:\GreenAnt\FloodDetection\18977ba0-beea-4eee-b4a2-32f5770c0ac8.tif"

# Load the saved model
print("Loading the trained model...")
best_model = joblib.load(model_path)

# Load the new imagery
print("Loading new imagery...")
new_imagery, new_profile = load_raster_as_array(new_imagery_path)

# Preprocess new imagery
print("Preprocessing new imagery...")
new_imagery_flat = new_imagery.reshape(new_imagery.shape[0], -1).T  # Flatten for prediction

# Make predictions on the new imagery
print("Making predictions on new imagery...")
predicted_classes = best_model.predict(new_imagery_flat)

# Reshape predictions back to the original image shape
predicted_image = predicted_classes.reshape(new_imagery.shape[1], new_imagery.shape[2])

# Save the classified image
print("Saving classified image...")
save_classified_image(predicted_image, output_classified_path, new_profile)

print(f"Classified image saved at: {output_classified_path}")

# Verify output values
unique_predicted = np.unique(predicted_image)
print(f"Unique values in classified image: {unique_predicted}")
