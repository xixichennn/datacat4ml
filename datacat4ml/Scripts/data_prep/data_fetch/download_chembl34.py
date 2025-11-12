import os
import requests
import tarfile

import sys
from datacat4ml.Scripts.const import DATA_DIR

def download_chembl(download_url, download_dir, download_file):
  
  # Create download directory (optional)
  os.makedirs(download_dir, exist_ok=True)  # Create directory if it doesn't exist

  # Download the file
  response = requests.get(download_url, stream=True)

  # Check for successful response
  if response.status_code == 200:
    # Define download path
    download_path = os.path.join(download_dir, download_file)

    # Write the file chunk by chunk
    with open(download_path, "wb") as f:
      for chunk in response.iter_content(1024):
        f.write(chunk)

    print(f"ChEMBL data downloaded to: {download_path}")
  else:
    print(f"Error downloading file: {response.status_code}")

def extract_db(download_dir, download_file):

  download_path = os.path.join(download_dir, download_file)
  with tarfile.open(download_path, "r:gz") as tar:
    tar.extractall(download_dir)

  tar = tarfile.open(os.path.join(download_dir, download_file), "r:gz")
  tar.extractall()
  tar.close()

  print(f"Extracted {download_file} to {download_dir}")

##########################################################################################
# Define download directory and filename
file = "chembl_34_sqlite.tar.gz"
url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_34_sqlite.tar.gz"

extract_db(DATA_DIR, file)