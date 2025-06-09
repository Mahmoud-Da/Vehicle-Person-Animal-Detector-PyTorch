import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
from PIL import Image


class CocoDetection(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

        # Our custom class mapping
        # COCO IDs:
        # person: 1, bicycle: 2, car: 3, motorcycle: 4, bus: 6, truck: 8
        # bird: 16, cat: 17, dog: 18, horse: 19, sheep: 20, cow: 21
        self.coco_to_our_id = {
            1: 1,  # person
            2: 2, 3: 2, 4: 2, 6: 2, 8: 2,  # vehicles
            16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 21: 3  # animals
        }
        self.our_classes = {0: '__background__',
                            1: 'person', 2: 'vehicle', 3: 'animal'}

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Load image
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        boxes = []
        labels = []

        for ann in anns:
            # Check if the category is one we care about
            if ann['category_id'] in self.coco_to_our_id:
                # COCO format is [x, y, width, height]
                # PyTorch format is [x_min, y_min, x_max, y_max]
                x_min = ann['bbox'][0]
                y_min = ann['bbox'][1]
                x_max = x_min + ann['bbox'][2]
                y_max = y_min + ann['bbox'][3]

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(self.coco_to_our_id[ann['category_id']])

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.ids)
