import numpy as np
from pathlib import Path

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create directories if they don't exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---- Inference Parameters ----
# Confidence threshold: only detections with a score above this will be shown.
CONFIDENCE_THRESHOLD = 0.6
# Default image to process, located in the INPUT_DIR
DEFAULT_IMAGE_NAME = "street.jpg"


# ---- Class and Color Definitions ----

# The full list of 91 classes from the COCO dataset the model was trained on.
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Our custom categories and the COCO classes that map to them
CUSTOM_MAPPING = {
    'vehicle': ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'],
    'animal': ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
    'person': ['person']
}

# --- Derived Configurations (no need to change these) ---

# Invert the mapping for quick lookup: { 'car': 'vehicle', 'dog': 'animal', ... }
COCO_TO_CUSTOM_CATEGORY = {
    coco_name: custom_category
    for custom_category, coco_list in CUSTOM_MAPPING.items()
    for coco_name in coco_list
}


# Use a fixed seed for reproducible random colors
# Assign a unique color to each of our custom categories
# CUSTOM_CLASSES = list(CUSTOM_MAPPING.keys())
# np.random.seed(42)
# COLORS = np.random.uniform(0, 255, size=(len(CUSTOM_CLASSES), 3))

# Use a fixed seed for reproducible colors
CATEGORY_TO_COLOR = {
    'vehicle': (0, 255, 0),     # Green (BGR)
    'person':  (0, 0, 255),     # Red (BGR)
    'animal':  (0, 255, 255)    # Yellow (BGR)
}
