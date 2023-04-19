import torch
from os import listdir, path
from PIL import Image
import pandas as pd
import torchvision.transforms.functional
from torch.utils.data import DataLoader


class TRANSFORM(object):
    def __call__(self, image, target):
        image = torchvision.transforms.functional.to_tensor(image)
        image = torchvision.transforms.functional.normalize(image, mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
        return image, target


class RobotHand(torch.utils.data.Dataset):
    def __init__(self, csv_file, root, transforms=TRANSFORM()):
        self.root = root
        self.annotations = pd.read_csv(csv_file)
        self.transforms = transforms
        self.images = list(sorted(listdir(path.join(root))))

    def __getitem__(self, idx):
        # load image
        img_path = path.join(self.root, self.images[idx])
        img = Image.open(img_path).convert("RGB")
        # load bboxes from .csv
        boxes = []
        for i in range(len(self.images)):
            if self.annotations.iloc[i, 0] == self.images[idx]:
                xmin = int(self.annotations.iloc[i, 4])
                xmax = int(self.annotations.iloc[i, 6])
                ymin = int(self.annotations.iloc[i, 5])
                ymax = int(self.annotations.iloc[i, 7])
                boxes.append([xmin, ymin, xmax, ymax])
                break

        area = [(boxes[0][3] - boxes[0][1]) * (boxes[0][2] - boxes[0][0])]
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float)
        # there is only one class
        labels = torch.ones((1,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = torch.as_tensor(area, dtype=torch.float32)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.images)


def collate_fn(batch):
    return tuple(zip(*batch))
