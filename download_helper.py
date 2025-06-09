import os
import urllib.request
import ssl

print("--- PyTorch Vision Detection Helper Downloader ---")


# Create an unverified SSL context to bypass certificate verification.
# This is a workaround for [SSL: CERTIFICATE_VERIFY_FAILED] errors.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Python 2.7.9+ and 3.4+ have this attribute.
    # For older versions, this will fail, but they are less likely to have this issue.
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# The base URL for the raw files on GitHub
BASE_URL = "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/"

# List of files to download
FILES_TO_DOWNLOAD = [
    "engine.py",
    "utils.py",
    "coco_eval.py",
    "coco_utils.py",
    "transforms.py"
]

# --- Script Logic ---
for filename in FILES_TO_DOWNLOAD:
    file_url = BASE_URL + filename

    if os.path.exists(filename):
        print(f"✅ '{filename}' already exists. Skipping.")
        continue

    print(f"⏳ Downloading '{filename}'...")

    try:
        # The download request will now use the unverified context
        urllib.request.urlretrieve(file_url, filename)
        print(f"✅ Successfully downloaded '{filename}'.")
    except Exception as e:
        print(f"❌ Error downloading '{filename}'.")
        print(f"   URL: {file_url}")
        print(f"   Error: {e}")
        print("   Please check your internet connection or the URL.")
        break

print("\n--- Download complete. ---")
