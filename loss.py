

import random
import torch
import torch.nn as nn

from utils import intersection_over_union


class YoloLoss(nn.Module):

    """reference : https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/loss.py"""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors, box_loss="MSE"):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i


        # No Object Loss
        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )


        # Object Loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])


        # Box Loss
        if box_loss == "MSE":
            predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
            target[..., 3:5] = torch.log(
                (1e-16 + target[..., 3:5] / anchors)
            )  # width, height coordinates
            box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        elif box_loss == "IoU" or box_loss == "GIoU" or box_loss == "DIoU" or box_loss == "CIoU" or box_loss == "SIoU":
            anchors = anchors.reshape(1, 3, 1, 1, 2) # 3(anchor) x 2(h, w), p_w * exp(t_w)를 연산하기 위해 reshape
            box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
            iou_loss = 1 - intersection_over_union(boxes_preds=box_preds[obj], boxes_labels=target[..., 1:5][obj], box_format="midpoint", iou_mode = box_loss)
            # box_loss = sum(iou_loss)/len(iou_loss)
            box_loss = iou_loss.mean()



        # Class Loss
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        #print("__________________________________")
        #print(self.lambda_box * box_loss)
        #print(self.lambda_obj * object_loss)
        #print(self.lambda_noobj * no_object_loss)
        #print(self.lambda_class * class_loss)
        #print("\n")

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
