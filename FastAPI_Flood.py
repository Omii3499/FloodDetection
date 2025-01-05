import os
import json
import datetime
from typing import List, Optional
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from sentinelhub import SHConfig, BBox, CRS, SentinelHubCatalog, SentinelHubRequest, DataCollection, bbox_to_dimensions, MimeType
from osgeo import gdal, osr
import rasterio
import joblib

app = FastAPI(
    title="Flood Detection API",
    description="API for processing Sentinel-1 imagery for flood detection",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ProcessingResponse(BaseModel):
    status: str
    output_directory: str
    processed_dates: List[str]
    error_message: Optional[str] = None

# Utility functions
def extract_bbox_from_geojson(geojson_data: dict) -> list:
    """Extract bounding box coordinates from GeoJSON data."""
    try:
        if geojson_data["type"] == "FeatureCollection":
            coordinates = geojson_data["features"][0]["geometry"]["coordinates"][0]
        elif geojson_data["type"] == "Feature":
            coordinates = geojson_data["geometry"]["coordinates"][0]
        else:
            coordinates = geojson_data["coordinates"][0]
        
        lons = [coord[0] for coord in coordinates]
        lats = [coord[1] for coord in coordinates]
        bbox = [min(lons), min(lats), max(lons), max(lats)]
        return bbox
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid GeoJSON format: {str(e)}"
        )

def validate_bbox(bbox: list) -> bool:
    """Validate bbox coordinates."""
    if len(bbox) != 4:
        raise HTTPException(
            status_code=400,
            detail="Bbox must contain exactly 4 coordinates"
        )
        
    lon_min, lat_min, lon_max, lat_max = bbox
    
    if not all(-180 <= lon <= 180 for lon in [lon_min, lon_max]):
        raise HTTPException(
            status_code=400,
            detail="Longitude must be between -180 and 180 degrees"
        )
    if not all(-90 <= lat <= 90 for lat in [lat_min, lat_max]):
        raise HTTPException(
            status_code=400,
            detail="Latitude must be between -90 and 90 degrees"
        )
        
    if abs(lon_max - lon_min) > 5 or abs(lat_max - lat_min) > 5:
        raise HTTPException(
            status_code=400,
            detail="Bounding box too large (max 5 degrees)"
        )
    return True

def validate_dates(start_date: str, end_date: str):
    """Validate input dates."""
    try:
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        
        if end < start:
            raise HTTPException(
                status_code=400,
                detail="End date must be after start date"
            )
        
        # Remove 30-day limitation
        # Add warning for very long periods
        days_difference = (end - start).days
        if days_difference > 180:  # Optional: Add a warning for very long periods
            print(f"Warning: Processing a long time period of {days_difference} days")
            
        return True
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD"
        )

def validate_output_dir(output_dir: str):
    """Validate and create output directory."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Test write permissions
        test_file = os.path.join(output_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid output directory: {str(e)}"
        )

def process_sentinel_data(date: str, bbox_coords: list, config: SHConfig, 
                         flood_model, output_dir: str, resolution: int = 10):
    """Process Sentinel data for a given date."""
    # Calculate tile size
    center_lat = (bbox_coords[1] + bbox_coords[3]) / 2
    earth_radius = 6371000
    degrees_lat = (resolution / earth_radius) * (180 / np.pi)
    degrees_lon = degrees_lat / np.cos(center_lat * np.pi / 180)
    
    aoi_bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
    size = bbox_to_dimensions(aoi_bbox, resolution=resolution)
    
    # Ensure size doesn't exceed limits
    size = [min(2500, s) for s in size]
    
    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: [{ bands: ["VV", "VH"] }],
            output: { bands: 2, sampleType: "FLOAT32" }
        };
    }

    function evaluatePixel(sample) {
        var decibelVV = sample.VV > 0 ? 10 * Math.log10(sample.VV) : -30;
        var decibelVH = sample.VH > 0 ? 10 * Math.log10(sample.VH) : -30;
        return [decibelVV, decibelVH];
    }
    """
    
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL1.define_from(
                    name="s1grd",
                    service_url="https://sh.dataspace.copernicus.eu"
                ),
                time_interval=(date, date),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=aoi_bbox,
        size=size,
        config=config,
    )
    
    data = request.get_data()
    
    if not data or len(data) == 0:
        raise Exception(f"No data retrieved for {date}")
        
    raw_data = data[0].astype(np.float32)
    
    vv_band = raw_data[:, :, 0]
    vh_band = raw_data[:, :, 1]
    
    features = np.stack([vv_band.flatten(), vh_band.flatten()], axis=1)
    predictions = flood_model.predict(features)
    predicted_image = predictions.reshape(vv_band.shape)
    
    return vv_band, vh_band, predicted_image

@app.post("/process_flood_detection", response_model=ProcessingResponse)
async def process_flood_detection(
    geojson: UploadFile = File(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    output_dir: str = Form(...)
):
    try:
        # Read and validate GeoJSON
        geojson_content = await geojson.read()
        geojson_data = json.loads(geojson_content)
        bbox_coords = extract_bbox_from_geojson(geojson_data)
        validate_bbox(bbox_coords)
        
        # Validate dates
        validate_dates(start_date, end_date)
        
        # Validate output directory
        validate_output_dir(output_dir)
        
        # Initialize Sentinel Hub
        config = SHConfig()
        config.sh_client_id = "sh-298f8633-2679-497b-9354-7cd820eecd13"
        config.sh_client_secret = "4I2WwDGB3kIQw9skZvmtyITr3YuEZxFx"
        config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        config.sh_base_url = "https://sh.dataspace.copernicus.eu"
        config.save()
        
        # Set up AOI
        aoi_bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
        
        # Search for available data
        catalog = SentinelHubCatalog(config=config)
        search_iterator = catalog.search(
            DataCollection.SENTINEL1,
            bbox=aoi_bbox,
            time=(start_date, end_date),
            fields={"include": ["id", "properties.datetime"], "exclude": []},
        )
        results = list(search_iterator)
        available_dates = [result["properties"]["datetime"][:10] for result in results]

        if not available_dates:
            return ProcessingResponse(
                status="completed",
                output_directory=output_dir,
                processed_dates=[],
                error_message="No images found for the specified date range"
            )

        # Load flood detection model
        model_path = r"B:\GreenAnt\FloodDetection\xgboost_binary_model.joblib"
        flood_model = joblib.load(model_path)
        
        processed_dates = []
        total_dates = len(available_dates)

        
        
        # Process each date
        for index, date in enumerate(available_dates, 1):
            print(f"Processing date {index}/{total_dates}: {date}")
            try:
                vv_band, vh_band, predicted_image = process_sentinel_data(
                    date=date,
                    bbox_coords=bbox_coords,
                    config=config,
                    flood_model=flood_model,
                    output_dir=output_dir
                )
                
                # Save Sentinel data
                sentinel_output = os.path.join(output_dir, f"sentinel1_{date}.tiff")
                driver = gdal.GetDriverByName('GTiff')
                out_ds = driver.Create(sentinel_output,
                                     vv_band.shape[1],
                                     vv_band.shape[0],
                                     2,
                                     gdal.GDT_Float32,
                                     options=['COMPRESS=LZW', 'TILED=YES'])
                
                geotransform = [
                    bbox_coords[0],
                    (bbox_coords[2] - bbox_coords[0]) / vv_band.shape[1],
                    0,
                    bbox_coords[3],
                    0,
                    -(bbox_coords[3] - bbox_coords[1]) / vv_band.shape[0]
                ]
                out_ds.SetGeoTransform(geotransform)
                
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(4326)
                out_ds.SetProjection(srs.ExportToWkt())
                
                for idx, (band_data, band_type) in enumerate([(vv_band, 'VV'), (vh_band, 'VH')], 1):
                    band = out_ds.GetRasterBand(idx)
                    band.WriteArray(band_data)
                    band.SetDescription(f"{date}_{band_type}")
                    band.SetNoDataValue(np.nan)
                
                out_ds = None
                
                # Save flood prediction
                flood_output = os.path.join(output_dir, f"flood_prediction_{date}.tiff")
                with rasterio.open(sentinel_output) as src:
                    profile = src.profile.copy()
                    profile.update(count=1, dtype='uint8', nodata=255)
                    with rasterio.open(flood_output, 'w', **profile) as dst:
                        dst.write(predicted_image.astype('uint8'), 1)
                
                # Create visualization
                plt.figure(figsize=(18, 6))
                
                plt.subplot(131)
                plt.imshow(vv_band, cmap='gray')
                plt.title(f'VV Band - {date}')
                plt.axis('off')
                
                plt.subplot(132)
                plt.imshow(vh_band, cmap='gray')
                plt.title(f'VH Band - {date}')
                plt.axis('off')
                
                plt.subplot(133)
                plt.imshow(predicted_image, cmap='RdYlBu')
                plt.title(f'Flood Prediction - {date}')
                plt.axis('off')
                
                plt.savefig(os.path.join(output_dir, f"visualization_{date}.png"))
                plt.close()
                
                processed_dates.append(date)
                print(f"Completed {index}/{total_dates} dates ({(index/total_dates*100):.1f}%)")
                
            except Exception as e:
                print(f"Error processing date {date}: {str(e)}")
                continue
        
        return ProcessingResponse(
            status="completed",
            output_directory=output_dir,
            processed_dates=processed_dates,
            error_message=None if processed_dates else "No dates were successfully processed"
        )
        
    except Exception as e:
        return ProcessingResponse(
            status="error",
            output_directory=output_dir,
            processed_dates=[],
            error_message=str(e)
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)