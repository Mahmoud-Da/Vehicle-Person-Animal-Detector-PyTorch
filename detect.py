# NOTE: for the "Fine-Tuning" using this script
import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
from train import get_model

# Define our classes
CLASSES = {1: 'person', 2: 'vehicle', 3: 'animal'}
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the trained model
num_classes = 4  # 3 classes + background
model = get_model(num_classes)
model.load_state_dict(torch.load(
    'vehicle_person_animal_detector.pth', map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode


def detect_objects(image_path, confidence_threshold=0.5):
    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image)

    # Add a batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        prediction = model(img_tensor)

    # Process the prediction
    boxes = prediction[0]['boxes'].numpy()
    labels = prediction[0]['labels'].numpy()
    scores = prediction[0]['scores'].numpy()

    # Load image with OpenCV for drawing
    cv_image = cv2.imread(image_path)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    for i in range(len(boxes)):
        if scores[i] > confidence_threshold:
            box = boxes[i]
            label_id = labels[i]
            class_name = CLASSES.get(label_id, 'Unknown')
            color = COLORS[label_id-1]

            # Draw bounding box
            cv2.rectangle(cv_image, (int(box[0]), int(
                box[1])), (int(box[2]), int(box[3])), color, 2)

            # Put label
            label_text = f"{class_name}: {scores[i]:.2f}"
            cv2.putText(cv_image, label_text, (int(box[0]), int(box[1]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the image
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Detections', cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Or save it
    # cv2.imwrite('output.jpg', cv_image)


if __name__ == '__main__':
    # Find a test image online with cars, people, and maybe a dog
    # and save it as 'test_image.jpg' in your project folder.
    detect_objects('test_image.jpg', confidence_threshold=0.6)
