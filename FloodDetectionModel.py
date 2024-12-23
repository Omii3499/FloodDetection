import numpy as np
import rasterio
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
import time

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

# Paths to input images
mask_path = r"C:\Users\admin\Downloads\Sentinel1_Classified_2.tif"
sentinel_path = r"C:\Users\admin\Downloads\Sentinel1_2_Preprcoees.tif"

# Load data
print("Loading data...")
mask, mask_profile = load_raster_as_array(mask_path)
sentinel, sentinel_profile = load_raster_as_array(sentinel_path)

# Get the number of bands
num_bands = sentinel.shape[0]
print(f"Input data has {num_bands} bands")

# Flatten data and prepare dataset
mask_flat = mask[0].flatten()  # Take first band if multiple bands exist
valid_mask = mask_flat != 255  # Assuming 255 is NoData value

# Prepare features
sentinel_flat = sentinel.reshape(sentinel.shape[0], -1).T
X = sentinel_flat[valid_mask]
y = mask_flat[valid_mask]

# Verify unique classes
unique_classes = np.unique(y)
print(f"Unique classes in training data: {unique_classes}")

# Split data
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Starting model training...")
start_time = time.time()

# Configure XGBoost for binary classification
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',  # Changed to binary classification
    tree_method='auto',
    random_state=42,
    use_label_encoder=False,     # Avoid label encoding warning
    eval_metric='logloss'        # Appropriate for binary classification
)

# Parameters for binary classification
param_grid = {
    "n_estimators": [100],
    "learning_rate": [0.1],
    "max_depth": [5],
    "scale_pos_weight": [1]  # Adjust if classes are imbalanced
}

grid_search = GridSearchCV(xgb_model, param_grid, scoring="f1", cv=3, verbose=1)
grid_search.fit(X_train, y_train)

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# Save the best model
best_model = grid_search.best_estimator_
model_path = r"B:\GreenAnt\FloodDetection\xgboost_binary_model.joblib"
joblib.dump(best_model, model_path)
print(f"Model saved to: {model_path}")

# Evaluate model
print("\nBinary Classification Metrics:")
y_pred = best_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# Calculate class distribution
class_dist = np.bincount(y_train.astype(int))
print(f"\nClass distribution in training data:")
print(f"Class 0: {class_dist[0]} pixels")
print(f"Class 1: {class_dist[1]} pixels")

# Make predictions
print("\nMaking predictions...")
sentinel_flat_for_prediction = sentinel.reshape(sentinel.shape[0], -1).T
predicted_classes = best_model.predict(sentinel_flat_for_prediction)

# Reshape to original image dimensions
predicted_image = predicted_classes.reshape(sentinel.shape[1], sentinel.shape[2])

# Save the classified image
output_path = r"B:\GreenAnt\FloodDetection\binary_classification.tif"
save_classified_image(predicted_image, output_path, sentinel_profile)
print(f"Binary classification image saved to: {output_path}")

# Verify output values
unique_predicted = np.unique(predicted_image)
print(f"\nUnique values in prediction: {unique_predicted}")


#Binary Classification Metrics:
#Accuracy: 0.9991
#Precision: 0.9934
#Recall: 0.9977
#F1 Score: 0.9956