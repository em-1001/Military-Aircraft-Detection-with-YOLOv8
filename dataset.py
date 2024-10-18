# reference : https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/dataset.py

import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_image_names_from_file(filename):
    image_names = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # Remove any leading or trailing whitespace characters (like newline)
            image_name = line.strip()
            # Append the image name to the list
            image_names.append(image_name)
    return image_names

class YOLODataset(Dataset):
    def __init__(
        self,
        dir,
        anchors,
        image_size=config.IMAGE_SIZE,
        S=config.S,
        C=config.NUM_CLASSES,
        transform=None,
    ):
        self.img_name_file = dir + '/image_names.txt'
        self.img_dir = dir +'/'
        self.label_dir = dir + '/label/'
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5
        self.image_names = read_image_names_from_file(self.img_name_file)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_name = self.image_names[index]
        label_path = self.label_dir + img_name + ".txt"
        bboxes = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2).tolist()
        img_path = self.img_dir + img_name + ".jpg"
        image = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w, _ = image.shape
        label_path = self.label_dir + img_name + ".txt"
        bboxes = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2).tolist()

        for i, box in enumerate(bboxes):
            x_min, y_min, x_max, y_max, cid = box

            x_min /= img_w
            x_max /= img_w
            y_min /= img_h
            y_max /= img_h

            x_c = (x_min + x_max) / 2
            y_c = (y_min + y_max) / 2
            w = x_max - x_min
            h = y_max - y_min
            bboxes[i] = [x_c, y_c, w, h, cid]

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S] # 6 -> [objectness, x, y, w, h, class], target => 3x3xSxSx6
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0) # argsort -> https://pytorch.org/docs/stable/generated/torch.argsort.html
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale # scale -> 0, 1, 2
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale # an anchor of particular scale -> 0, 1, 2
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell e.g. x = 0.5, S = 13 --> int(6.5) = 6th cell of x
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1 # objectness = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both are between [0, 1] e.g. 6.5 - 6 = 0.5
                    width_cell, height_cell = (
                        width * S, # e.g. S = 13, width = 0.5, 6.5
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                # box가 현재 scale_idx에 대한 anchor 할당을 iou가 가장 높은거로 받았으나 각 scale당 3개의 anchor가 존재하므로 iou가 가장 높은거 이외의 것들이 여기로 온다.
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction
        return image, tuple(targets)



def test():
    anchors = config.ANCHORS

    transform = config.train_transforms

    dataset = YOLODataset(
        './train',
        S=config.S,
        anchors=anchors,
        transform=transform,
    )
    S = config.S
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    cnt = 0
    for x, y in loader:
        boxes = []
        print(x.shape)

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)
        cnt += 1
        if cnt == 20:
            break


if __name__ == "__main__":
    test()