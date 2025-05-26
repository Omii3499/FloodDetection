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
        coordinates = []
        
        # Handle different GeoJSON types
        if geojson_data["type"] == "FeatureCollection":
            geometry = geojson_data["features"][0]["geometry"]
            if geometry["type"] == "MultiPolygon":
                # For MultiPolygon, concatenate all polygon coordinates
                for polygon in geometry["coordinates"]:
                    coordinates.extend(polygon[0])  # First ring of each polygon
            else:  # Regular Polygon
                coordinates = geometry["coordinates"][0]
        elif geojson_data["type"] == "Feature":
            geometry = geojson_data["geometry"]
            if geometry["type"] == "MultiPolygon":
                for polygon in geometry["coordinates"]:
                    coordinates.extend(polygon[0])
            else:
                coordinates = geometry["coordinates"][0]
        else:  # Raw geometry
            if geojson_data["type"] == "MultiPolygon":
                for polygon in geojson_data["coordinates"]:
                    coordinates.extend(polygon[0])
            else:
                coordinates = geojson_data["coordinates"][0]
        
        if not coordinates:
            raise ValueError("No coordinates found in GeoJSON")
            
        # Extract all x and y coordinates
        x_coords = [float(coord[0]) for coord in coordinates]
        y_coords = [float(coord[1]) for coord in coordinates]
        
        # Calculate bounding box
        bbox = [
            min(x_coords),  # min longitude
            min(y_coords),  # min latitude
            max(x_coords),  # max longitude
            max(y_coords)   # max latitude
        ]
        
        print(f"Extracted bbox: {bbox}")  # Debug print
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
        
    try:
        # Convert all coordinates to float and unpack
        lon_min, lat_min, lon_max, lat_max = [float(coord) for coord in bbox]
        
        # Validate longitude values
        if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
            raise HTTPException(
                status_code=400,
                detail="Longitude must be between -180 and 180 degrees"
            )
            
        # Validate latitude values    
        if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
            raise HTTPException(
                status_code=400,
                detail="Latitude must be between -90 and 90 degrees"
            )
            
        # Validate bounding box size
        if abs(lon_max - lon_min) > 5 or abs(lat_max - lat_min) > 5:
            raise HTTPException(
                status_code=400,
                detail="Bounding box too large (max 5 degrees)"
            )
            
        return True
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid coordinate values: {str(e)}"
        )

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
# Update this part in the main processing loop
def create_binary_flood_map(predicted_image, vv_band, vh_band, date, output_dir):
    """Create a binary flood map with custom colors and legend, including no-data areas."""
    plt.figure(figsize=(12, 8))
    
    # Create a mask for no-data areas
    no_data_mask = np.isnan(vv_band)  # Basic no-data from NaN
    sentinel_no_data = (vv_band == -30) & (vh_band == -30)  # Sentinel-1 no-data condition
    combined_no_data = no_data_mask | sentinel_no_data  # Combine both conditions
    
    # Create a modified classification image
    classification = np.copy(predicted_image)
    classification[combined_no_data] = 2  # Use 2 for no-data
    
    # Create custom colormap (Red for Non-Flooded, Blue for Flooded, Gray for No-Data)
    colors = ['red', 'blue', 'lightgray']
    custom_cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    # Create the main plot
    img = plt.imshow(classification, cmap=custom_cmap, interpolation='nearest')
    plt.title(f'Flood Classification Map - {date}', fontsize=14, pad=20)
    
    # Create custom legend patches
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='red', label='Non-Flooded'),
        plt.Rectangle((0,0),1,1, facecolor='blue', label='Flooded'),
        plt.Rectangle((0,0),1,1, facecolor='lightgray', label='No Data')
    ]
    plt.legend(handles=legend_elements, loc='upper right', 
              bbox_to_anchor=(1.2, 1), fontsize=12)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f"binary_classified_{date}.jpg"), 
                bbox_inches='tight', dpi=900)
    plt.close()

def calculate_statistics(vv_band: np.ndarray, vh_band: np.ndarray, 
                       predicted_image: np.ndarray, bbox_coords: list, 
                       date: str, region_name: str = "AOI"):
    """Calculate statistics for the JSON output."""
    # Create combined no-data mask
    no_data_mask = np.isnan(vv_band)  # Basic no-data from NaN
    sentinel_no_data = (vv_band == -30) & (vh_band == -30)  # Sentinel-1 no-data condition
    combined_no_data = no_data_mask | sentinel_no_data  # Combine both conditions
    
    # Calculate pixel counts excluding no-data areas
    valid_pixels = ~combined_no_data
    total_pixels = np.sum(valid_pixels)
    flood_pixels = np.sum((predicted_image == 1) & valid_pixels)
    non_flood_pixels = np.sum((predicted_image == 0) & valid_pixels)
    no_data_pixels = np.sum(combined_no_data)
    
    # Calculate areas (assuming 10m resolution)
    pixel_area_km2 = 0.0001  # 10m x 10m = 100m2 = 0.0001 km2
    total_area_km2 = total_pixels * pixel_area_km2
    flood_area_km2 = flood_pixels * pixel_area_km2
    
    # Calculate band statistics
    vv_stats = {
        "mean": float(np.nanmean(vv_band)),
        "std": float(np.nanstd(vv_band))
    }
    
    vh_stats = {
        "mean": float(np.nanmean(vh_band)),
        "std": float(np.nanstd(vh_band))
    }
    
    # Calculate simple confidence metrics (example implementation)
    model_confidence = 0.85  # This should be replaced with actual model confidence
    spatial_consistency = 1 - (np.sum(np.isnan(predicted_image)) / total_pixels)
    
    return {
        "metadata": {
            "timestamp": date,
            "source": "Sentinel-1",
            "region": region_name,
            "spatial_extent": {
                "bbox": bbox_coords,
                "crs": "EPSG:4326"
            }
        },
        "flood_analysis": {
            "affected_areas": {
                "total_area_km2": float(total_area_km2),
                "flood_area_km2": float(flood_area_km2),
                "percentage": float(flood_area_km2 / total_area_km2 * 100)
            },
            "confidence_metrics": {
                "model_confidence": float(model_confidence),
                "spatial_consistency": float(spatial_consistency)
            }
        },
        "statistics": {
            "pixel_counts": {
                "total": int(total_pixels),
                "flood": int(flood_pixels),
                "non_flood": int(non_flood_pixels),
                "no_data": int(no_data_pixels)
            },
            "band_statistics": {
                "vv": vv_stats,
                "vh": vh_stats
            }
        }
    }

def validate_and_clean_output_dir(output_dir: str) -> str:
    """Clean and validate the output directory path."""
    try:
        # Remove any quotes and prefixes
        cleaned_path = output_dir.strip("'").strip('"')
        if cleaned_path.startswith('r'):
            cleaned_path = cleaned_path[1:].strip("'").strip('"')
        
        # Replace backslashes with forward slashes
        cleaned_path = cleaned_path.replace('\\', '/')
        
        # Remove any double forward slashes
        while '//' in cleaned_path:
            cleaned_path = cleaned_path.replace('//', '/')
            
        # Create directory if it doesn't exist
        os.makedirs(cleaned_path, exist_ok=True)
        
        # Test write permissions
        test_file = os.path.join(cleaned_path, "test.txt")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot write to output directory: {str(e)}"
            )
            
        return cleaned_path
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid output directory: {str(e)}"
        )

def process_sentinel_data(date: str, bbox_coords: list, config: SHConfig, 
                         flood_model, output_dir: str, region_name: str = "AOI",
                         resolution: int = 10):
    """Process Sentinel data for a given date."""
    # Calculate tile size
    center_lat = (bbox_coords[1] + bbox_coords[3]) / 2
    earth_radius = 6371000
    degrees_lat = (resolution / earth_radius) * (180 / np.pi)
    degrees_lon = degrees_lat / np.cos(center_lat * np.pi / 180)
    
    aoi_bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
    size = bbox_to_dimensions(aoi_bbox, resolution=resolution)
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
    
    # Calculate statistics here, passing the date parameter
    stats = calculate_statistics(
        vv_band=vv_band,
        vh_band=vh_band,
        predicted_image=predicted_image,
        bbox_coords=bbox_coords,
        date=date,
        region_name=region_name
    )
    
    return vv_band, vh_band, predicted_image, stats

@app.post("/process_flood_detection", response_model=ProcessingResponse)
async def process_flood_detection(
    geojson: UploadFile = File(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    output_dir: str = Form(...)
):
    try:

        geojson_filename = geojson.filename
        region_name = os.path.splitext(geojson_filename)[0] 

        # Clean and validate the output directory
        output_dir = validate_and_clean_output_dir(output_dir)
        
        # Read and validate GeoJSON
        geojson_content = await geojson.read()
        geojson_data = json.loads(geojson_content)
        bbox_coords = extract_bbox_from_geojson(geojson_data)
        validate_bbox(bbox_coords)
        
        # Validate dates
        validate_dates(start_date, end_date)
        
        # Initialize Sentinel Hub
        config = SHConfig()
        config.sh_client_id " "
        config.sh_client_secret =  "
        config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        config.sh_base_url = "https://sh.dataspace.copernicus.eu"
        config.save()
        
        # Search for available data
        catalog = SentinelHubCatalog(config=config)
        search_iterator = catalog.search(
            DataCollection.SENTINEL1,
            bbox=BBox(bbox=bbox_coords, crs=CRS.WGS84),
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
        model_path = r"B:\GreenAnt\FloodDetection\Flood_Model.joblib"
        flood_model = joblib.load(model_path)
        
        processed_dates = []
        total_dates = len(available_dates)

        # Process each date
        for index, date in enumerate(available_dates, 1):
            print(f"Processing date {index}/{total_dates}: {date}")
            try:
                # Process the data
                vv_band, vh_band, predicted_image, stats = process_sentinel_data(
                    date=date,
                    bbox_coords=bbox_coords,
                    config=config,
                    flood_model=flood_model,
                    output_dir=output_dir,
                    region_name=region_name
                )
                
                # Save JSON output
                json_output = os.path.join(output_dir, f"flood_analysis_{date}.json")
                with open(json_output, 'w') as f:
                    json.dump(stats, f, indent=2)
                
                # # Save classified image as JPG with legend
                # plt.figure(figsize=(10, 8))
                # classified_img = plt.imshow(predicted_image, cmap='RdYlBu')
                # plt.colorbar(classified_img, label='Flood Classification')
                # plt.title(f'Flood Classification - {date}')
                # plt.axis('off')
                # plt.savefig(os.path.join(output_dir, f"classified_{date}.jpg"))
                # plt.close()
 
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
                # Save classified image with binary legend
                create_binary_flood_map(predicted_image,vv_band, vh_band,date, output_dir)

                # For the visualization panel, also update the third subplot
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
                # Create combined no-data mask
                no_data_mask = np.isnan(vv_band)
                sentinel_no_data = (vv_band == -30) & (vh_band == -30)
                combined_no_data = no_data_mask | sentinel_no_data
                
                classification = np.copy(predicted_image)
                classification[combined_no_data] = 2

                colors = ['red', 'blue', 'lightgray']
                custom_cmap = plt.matplotlib.colors.ListedColormap(colors)
                plt.imshow(classification, cmap=custom_cmap)
                plt.title(f'Flood Classification - {date}')
                legend_elements = [
                    plt.Rectangle((0,0),1,1, facecolor='red', label='Non-Flooded'),
                    plt.Rectangle((0,0),1,1, facecolor='blue', label='Flooded'),
                    plt.Rectangle((0,0),1,1, facecolor='lightgray', label='No Data')
                ]
                plt.legend(handles=legend_elements, loc='upper right')
                plt.axis('off')
                
                plt.savefig(os.path.join(output_dir, f"visualization_{date}.png"))
                plt.close()
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
