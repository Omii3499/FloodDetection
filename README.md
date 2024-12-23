# Flood Detection Using Sentinel-1 Imagery with XGBoost Classifier

This project demonstrates the use of Sentinel-1 SAR imagery to perform flood detection using a binary classification model trained with XGBoost. It involves preprocessing imagery, training the model, and predicting flood-affected areas.

## Project Workflow

### 1. Training the Model
- The model is trained using a classified mask and Sentinel-1 imagery. 
- The input Sentinel-1 imagery contains multiple bands (e.g., VV and VH), and a mask file provides labeled data for training.
- The script:
  - Loads the mask and Sentinel-1 data.
  - Prepares training and test datasets.
  - Trains an XGBoost model for binary classification (`flood` vs `non-flood`).
  - Saves the trained model for future use.

### 2. Predicting Flood-Affected Areas
- The trained model is used to predict flood-affected areas on new Sentinel-1 imagery.
- For each date:
  - VV and VH bands are extracted and combined as input features.
  - Predictions are made for the entire image.
  - The binary classification results are saved as GeoTIFF files.

## Requirements

- **Python Libraries**: 
  - `numpy`
  - `rasterio`
  - `joblib`
  - `scikit-learn`
  - `xgboost`

Install the required libraries using:
```bash
pip install numpy rasterio joblib scikit-learn xgboost
Data:
Sentinel-1 imagery with bands VV and VH.
A classified mask file for training.
Usage
1. Training the Model
Run the script to train the model:

python
Copy code
# Paths to input files
mask_path = r"C:\path\to\mask.tif"
sentinel_path = r"C:\path\to\sentinel.tif"

# Output path for the trained model
model_path = r"C:\path\to\save\model.joblib"
2. Flood Prediction on New Imagery
Run the script for flood detection on new Sentinel-1 imagery:

python
Copy code
# Path to the trained model
model_path = r"C:\path\to\trained_model.joblib"

# Path to new Sentinel-1 imagery
new_imagery_path = r"C:\path\to\new\sentinel_image.tif"

# Output directory for results
output_dir = r"C:\path\to\output"
Outputs
Trained Model: Saved as .joblib file.
Flood Maps: Saved as GeoTIFF files with binary classification (0: Non-flood, 1: Flood).
Key Features
Binary Classification Metrics:

Accuracy
Precision
Recall
F1 Score
Class Distribution: Displays the distribution of flood and non-flood pixels in the training data.

Classified Output: Generates a binary flood map for each date in the input imagery.

Example Metrics
Sample performance metrics achieved during training:

yaml
Copy code
Accuracy: 0.9991
Precision: 0.9934
Recall: 0.9977
F1 Score: 0.9956
File Structure
bash
Copy code
FloodDetection/
├── train_model.py             # Script for training the XGBoost model
├── predict_flood.py           # Script for flood prediction
├── data/                      # Directory for input data
├── models/                    # Directory for saved models
├── output/                    # Directory for classified results
└── README.md                  # Proje
