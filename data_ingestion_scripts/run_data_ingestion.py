import sys
import zipfile
from pathlib import Path
from sql_db.pipeline import main as sql_main
from vector_db.pipeline import main as vector_main
from config.settings import RAW_DATA_DIR, TMDB_ZIP_PATH, TMDB_ZIP_FILE_NAME, MOVIES_CSV_FILE_NAME, CREDITS_CSV_FILE_NAME

# Required CSV files in the zip
REQUIRED_FILES = [CREDITS_CSV_FILE_NAME, MOVIES_CSV_FILE_NAME]


def extract_csv_files() -> bool:
    """
    Extract required CSV files from zip archive.
    
    Returns:
        True if extraction successful, False otherwise
    """
    zip_path = TMDB_ZIP_PATH
    
    # Check if zip file exists
    if not zip_path.exists():
        print(f"Error: Zip file not found at {zip_path}")
        print("Please download the correct zip file from:")
        print("https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
        return False
    
    # Validate it's a valid zip file
    if not zipfile.is_zipfile(zip_path):
        print(f"Error: {zip_path} is not a valid zip file")
        print("Please download the correct zip file from:")
        print("https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
        return False
    
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Get all files in the zip
            zip_contents = zf.namelist()
            
            # Check if required files exist in zip
            missing_files = []
            for required_file in REQUIRED_FILES:
                # Check if file exists (could be in root or subdirectory)
                matching_files = [f for f in zip_contents if f.endswith(required_file)]
                if not matching_files:
                    missing_files.append(required_file)
            
            if missing_files:
                print(f"Error: Required files missing from zip: {missing_files}")
                print("Files found in zip:", zip_contents)
                print("\nPlease extract correct CSV files from:")
                print("https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
                return False
            
            # Extract only the required CSV files
            print(f"Extracting CSV files from {TMDB_ZIP_FILE_NAME}")
            for required_file in REQUIRED_FILES:
                # Find the file in zip (handle subdirectories)
                matching_files = [f for f in zip_contents if f.endswith(required_file)]
                
                for file_in_zip in matching_files:
                    # Extract to RAW_DATA_DIR with just the filename (no subdirs)
                    source = zf.open(file_in_zip)
                    target_path = RAW_DATA_DIR / required_file
                    
                    with open(target_path, "wb") as target:
                        target.write(source.read())
                    
                    # Get file size
                    size_mb = target_path.stat().st_size / (1024 * 1024)
                    print(f"  ✓ Extracted {required_file} ({size_mb:.2f} MB)")
            
            print(f"Successfully extracted all CSV files to {RAW_DATA_DIR}")
            return True
            
    except Exception as e:
        print(f"❌ Error extracting files: {e}")
        print("Please extract correct CSV files from:")
        print("https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
        return False

if __name__ == "__main__":
    # Extract CSV files from zip
    extraction_success = extract_csv_files()
    
    if not extraction_success:
        print("\nExtraction failed. Data ingestion execution aborted.")
        sys.exit(1)
    
    # Only proceed if extraction was successful
    print("\nStarting Data ingestion...")
    try:
        sql_main()
        vector_main()
        print("\nData ingestion completed successfully!")
    except Exception as e:
        print(f"\nData ingestion execution failed: {e}")
        sys.exit(1)