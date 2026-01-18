import os
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
MAPBOX_ACCESS_TOKEN = "pk.eyJ1Ijoic2FyZGFuYWhpbWVzaCIsImEiOiJjbWswanNsOG8weHV5M2dzYzFucWg5bDA5In0.dLTFxsXeGtxiwUb7ioHbhw"
IMAGE_DIR = "satellite_images"
ZOOM_LEVEL = 18  # Building level detail
IMAGE_SIZE = "600x600"
STYLE_ID = "mapbox/satellite-v9"

def download_image(args):
    """
    Downloads a single image from Mapbox Static API.
    args: (id, lat, lon) tuple
    """
    img_id, lat, lon = args
    filename = os.path.join(IMAGE_DIR, f"{img_id}.jpg")
    
    # Skip if already exists
    if os.path.exists(filename):
        return
    
    url = f"https://api.mapbox.com/styles/v1/{STYLE_ID}/static/{lon},{lat},{ZOOM_LEVEL}/{IMAGE_SIZE}?access_token={MAPBOX_ACCESS_TOKEN}"
    
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
        else:
            print(f"Failed to download {img_id}: Status {response.status_code}")
    except Exception as e:
        print(f"Error downloading {img_id}: {e}")

def main():
    # Create output directory
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    print("Loading datasets...")
    # Load IDs and Coordinates
    # Assuming 'id', 'lat', 'long' are the column names based on standard housing datasets
    try:
        train_df = pd.read_excel('train.xlsx')
        test_df = pd.read_excel('test.xlsx')
    except Exception as e:
        print(f"Error loading Excel files: {e}")
        return

    # Combine to fetch all required images
    # We only need ID and Coordinates
    # Check column names first (sometimes it's 'lat', 'long' or 'latitude', 'longitude')
    lat_col = 'lat' if 'lat' in train_df.columns else 'latitude'
    lon_col = 'long' if 'long' in train_df.columns else 'longitude'
    
    print(f"Using columns: ID='id', Lat='{lat_col}', Lon='{lon_col}'")
    
    # Prepare list of tasks
    tasks = []
    
    for df in [train_df, test_df]:
        for _, row in df.iterrows():
            tasks.append((str(row['id']), row[lat_col], row[lon_col]))
            
    print(f"Found {len(tasks)} images to download.")
    
    # Download with thread pool for speed
    # Be mindful of rate limits, but Mapbox is usually generous with static images.
    # Using 10 workers to speed up ~21k requests
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(download_image, tasks), total=len(tasks), desc="Downloading Images"))

if __name__ == "__main__":
    main()
