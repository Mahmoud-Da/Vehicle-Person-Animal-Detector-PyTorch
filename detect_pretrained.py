import os
import ssl

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

from config import (CATEGORY_TO_COLOR, COCO_INSTANCE_CATEGORY_NAMES,
                    COCO_TO_CUSTOM_CATEGORY, CONFIDENCE_THRESHOLD, OUTPUT_DIR)
from helpers import generate_output_filename, save_image_cv

# This block bypasses SSL certificate verification for this script.
# It is needed (Install Certificates.command) does not work.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


print("--- Vehicle, Person, Animal Detector (Pre-trained COCO Model) ---")


# --- Step 1: Load the pre-trained model ---

# Load the model with the best available pre-trained weights on COCO
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

# Set the model to evaluation mode
model.eval()


# --- Step 2: Create the detection function ---

def detect_objects(image_path, confidence_threshold=CONFIDENCE_THRESHOLD):
    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)  # Add a batch dimension

    # Make a prediction
    with torch.no_grad():
        prediction = model(img_tensor)

    # Process predictions
    boxes = prediction[0]['boxes'].numpy()
    coco_labels = prediction[0]['labels'].numpy()
    scores = prediction[0]['scores'].numpy()

    # Load image with OpenCV for drawing
    cv_image = cv2.imread(image_path)

    # Loop through all detected objects
    for i in range(len(boxes)):
        if scores[i] > confidence_threshold:
            # Get the COCO class name (e.g., 'car', 'dog')
            coco_id = coco_labels[i]
            if coco_id >= len(COCO_INSTANCE_CATEGORY_NAMES):
                continue
            coco_name = COCO_INSTANCE_CATEGORY_NAMES[coco_id]

            # Check if this COCO class is one we care about
            if coco_name in COCO_TO_CUSTOM_CATEGORY:
                # Get our custom category ('vehicle', 'animal') and its color
                custom_category = COCO_TO_CUSTOM_CATEGORY[coco_name]
                color = CATEGORY_TO_COLOR[custom_category]

                # Get the bounding box coordinates
                box = boxes[i]
                x_min, y_min, x_max, y_max = int(box[0]), int(
                    box[1]), int(box[2]), int(box[3])

                # Draw the bounding box
                cv2.rectangle(cv_image, (x_min, y_min),
                              (x_max, y_max), color, 2)

                # Create the label text (e.g., "vehicle: 0.98")
                label_text = f"{custom_category}: {scores[i]:.2f}"
                cv2.putText(cv_image, label_text, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display the final image
    # cv2.imshow('Detections', cv_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Or save it:
    # cv2.imwrite(f'{OUTPUT_DIR}/result.jpg', cv_image)
    output_filename = generate_output_filename(image_path)
    output_save_path = OUTPUT_DIR / output_filename
    save_image_cv(cv_image, output_save_path)


if __name__ == '__main__':
    # set the Path of the Image EX: inputs/test_image.jpg
    image_to_test = 'inputs/street.jpg'
    if not os.path.exists(image_to_test):
        print(f"Error: Test image '{image_to_test}' not found.")
        print("Please download an image and save it with that name to run the detection.")
    else:
        detect_objects(
            image_to_test, confidence_threshold=CONFIDENCE_THRESHOLD)
