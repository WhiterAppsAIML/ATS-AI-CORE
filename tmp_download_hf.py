import os
import requests
import concurrent.futures
from pathlib import Path

# Configuration
BASE_URL = "https://huggingface.co/datasets/netsol/resume-score-details/raw/main/"
DEST_DIR = Path(r"c:\Users\saini\Desktop\ats\data\raw\resume_score_details")
MAX_WORKERS = 10

# File patterns to download based on the API list
FILES_TO_DOWNLOAD = []

# Matched pairs (approx 648 based on search)
for i in range(648):
    FILES_TO_DOWNLOAD.append(f"match_{i}.json")

# Mismatched pairs (approx 201)
for i in range(201):
    FILES_TO_DOWNLOAD.append(f"mismatch_{i}.json")

# Additional info (approx 126)
for i in range(126):
    FILES_TO_DOWNLOAD.append(f"empty_additional_info_{i}.json")

def download_file(filename):
    url = BASE_URL + filename
    dest_path = DEST_DIR / filename
    
    if dest_path.exists():
        return f"Skipped {filename}"
        
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(dest_path, "wb") as f:
                f.write(response.content)
            return f"Downloaded {filename}"
        else:
            return f"Failed {filename} (Status: {response.status_code})"
    except Exception as e:
        return f"Error {filename}: {str(e)}"

def main():
    print(f"Starting download of {len(FILES_TO_DOWNLOAD)} files to {DEST_DIR}...")
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(download_file, FILES_TO_DOWNLOAD))
    
    success_count = len([r for r in results if "Downloaded" in r])
    skipped_count = len([r for r in results if "Skipped" in r])
    failed_count = len(results) - success_count - skipped_count
    
    print(f"Finished! Success: {success_count}, Skipped: {skipped_count}, Failed: {failed_count}")

if __name__ == "__main__":
    main()
