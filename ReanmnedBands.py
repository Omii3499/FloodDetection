from osgeo import gdal
import re

# Path to your input TIFF file
input_file = r"B:\GreenAnt\FloodImages\Opava_2024.tif"

# Path to save the updated TIFF file
output_file = r"B:\GreenAnt\FloodImages\Preprcess\Opava_2024.tif"
# Open the input file
dataset = gdal.Open(input_file, gdal.GA_Update)

# Check if the file is valid
if dataset is None:
    print("Failed to open file.")
else:
    # Rename the bands
    for band_index in range(dataset.RasterCount):
        band = dataset.GetRasterBand(band_index + 1)
        
        # Get the current band description (filename-based metadata)
        band_name = band.GetDescription()
        
        # Updated regex pattern to extract date and polarization
        pattern = r"(\d{4}-\d{2}-\d{2})_(VV|VH)$"
        match = re.search(pattern, band_name)
        if match:
            date = match.group(1)  # Extract the date
            polarization = match.group(2).lower()  # Extract VV/VH and convert to lowercase
            new_name = f"{date}_{polarization}"  # Create the new name
            
            # Set the new description for the band
            band.SetDescription(new_name)
            print(f"Band {band_index + 1} renamed to: {new_name}")
        else:
            print(f"Band {band_index + 1} does not match the pattern and was not renamed.")

    # Save changes to a new file
    dataset.FlushCache()
    dataset = None

    # Copy the updated dataset to a new file
    gdal.Translate(output_file, input_file)
    print(f"Updated file saved at: {output_file}")