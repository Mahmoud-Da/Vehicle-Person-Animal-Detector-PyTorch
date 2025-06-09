# NOTE: for the "Fine-Tuning" using this script otherwise using "detect_pretrained.py" directly

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataset import CocoDetection
from engine import train_one_epoch, evaluate
import utils


def get_model(num_classes):
    """
    Loads a pre-trained Faster R-CNN model and replaces the
    classifier with a new one for our number of classes.
    """
    # Load a pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT")

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        # For training, add data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Our classes: background, person, vehicle, animal
    num_classes = 4

    # Use our dataset and defined transformations
    dataset = CocoDetection(root='data/train2017',
                            annFile='data/annotations/instances_train2017.json',
                            transform=get_transform(train=True))

    dataset_test = CocoDetection(root='data/val2017',
                                 annFile='data/annotations/instances_val2017.json',
                                 transform=get_transform(train=False))

    # Split dataset into training and validation
    # (Optional: for simplicity, we use the full train/val sets as provided)
    data_loader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn
    )

    data_loader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn
    )

    # Get the model
    model = get_model(num_classes)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 20

    for epoch in range(num_epochs):
        # Train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader,
                        device, epoch, print_freq=10)
        # Update the learning rate
        lr_scheduler.step()
        # Evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
    # Save the trained model
    torch.save(model.state_dict(), 'vehicle_person_animal_detector.pth')


if __name__ == "__main__":
    main()
