import os
import urllib.request
import zipfile
import ssl

# Attempt to import tqdm for progress bars, guide user if not found
try:
    from tqdm import tqdm
except ImportError:
    print("Tqdm library not found. Please install it for progress bars:")
    print("pip install tqdm")
    # A simple dummy class if tqdm is not available

    class tqdm:
        def __init__(self, total=None, unit=None, unit_scale=None, desc=None):
            self.total = total

        def __enter__(self):
            return self

        def __exit__(self, *args, **kwargs):
            pass

        def update(self, n):
            pass

# --- Configuration ---
DATA_DIR = "data"
# List of files to download: (URL, filename, directory to extract to)
FILES = [
    (
        "http://images.cocodataset.org/zips/train2017.zip",
        "train2017.zip",
        "train2017",
    ),
    (
        "http://images.cocodataset.org/zips/val2017.zip",
        "val2017.zip",
        "val2017"
    ),
    (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "annotations_trainval2017.zip",
        "annotations",
    ),
]

# --- SSL Certificate Workaround ---
# Handles [SSL: CERTIFICATE_VERIFY_FAILED] errors
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- TQDM Progress Bar Helper ---


class TqdmUpTo(tqdm):
    """Provides `update_to(block_num, block_size, total_size)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

# --- Main Script Logic ---


def download_and_extract():
    """Downloads and extracts the COCO dataset."""
    print("--- COCO 2017 Dataset Downloader ---")
    print(
        f"Data will be downloaded and extracted to the '{DATA_DIR}/' directory.")
    print("WARNING: This will download ~20 GB of data and requires >40 GB of free disk space.")

    # Create the main data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    for url, filename, extract_dir_name in FILES:
        zip_path = os.path.join(DATA_DIR, filename)
        extract_path = os.path.join(DATA_DIR, extract_dir_name)

        # --- Step 1: Download the file ---
        if not os.path.exists(zip_path):
            print(f"\nDownloading '{filename}'...")
            try:
                with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
                    urllib.request.urlretrieve(
                        url, filename=zip_path, reporthook=t.update_to)
                print(f"✅ Download complete: '{zip_path}'")
            except Exception as e:
                print(f"❌ Failed to download {filename}. Error: {e}")
                # Clean up partially downloaded file if it exists
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                return  # Stop the script if a download fails
        else:
            print(
                f"\n✅ Zip file '{filename}' already exists. Skipping download.")

        # --- Step 2: Extract the file ---
        # The annotation zip file extracts to its own folder named 'annotations'
        # We handle this by checking for the target directory name
        if extract_dir_name == "annotations":
            final_extract_path = os.path.join(DATA_DIR, 'annotations')
        else:
            final_extract_path = extract_path

        if not os.path.exists(final_extract_path):
            print(f"⏳ Extracting '{filename}'...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(DATA_DIR)
                print(
                    f"✅ Extraction complete. Data is in '{final_extract_path}/'")
            except Exception as e:
                print(
                    f"❌ Failed to extract {filename}. The file might be corrupted. Error: {e}")
                return
        else:
            print(
                f"✅ Directory '{final_extract_path}' already exists. Skipping extraction.")

    print("\n--- All dataset files are downloaded and extracted successfully! ---")
    print("You can now run 'python3 train.py'.")


if __name__ == "__main__":
    try:
        download_and_extract()
    except KeyboardInterrupt:
        print("\nDownload interrupted by user. Cleaning up...")
        # Optional: clean up partially downloaded files if you want
        print("Exiting...")
