"""IMPORT"""
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import math
import random

import torch
import torchvision
from torchvision.utils import draw_bounding_boxes, make_grid
from torchvision.transforms import transforms as T
from torch.utils.data import Dataset, DataLoader
from torchmetrics import JaccardIndex
from torchvision.ops import box_iou
from torch import nn


from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

from pycocotools.coco import COCO
from roboflow import Roboflow

import albumentations as A
from albumentations.pytorch import ToTensorV2

from datetime import datetime


if not os.path.exists("maskdetection-3"):
    print("Downloading dataset...")
    rf = Roboflow(api_key="RLpF5qnVG3u4wi0Hgkmg")
    project = rf.workspace("diza-febriyan-hasal").project("maskdetection-tdrvn")
    dataset = project.version(3).download("coco")


def transformation():
    transform = A.Compose([ToTensorV2()], bbox_params=A.BboxParams(format="coco"))
    return transform


class MaskDetection(Dataset):
    def __init__(self, root, split="train", transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.split = split  # train, valid, test
        self.coco = COCO(
            os.path.join(root, split, "_clean_annotations.coco.json")
        )  # annotatiosn stored here
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]
        self.transforms = transform

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        image = cv2.imread(os.path.join(self.root, self.split, path))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        target = copy.deepcopy(self._load_target(id))

        boxes = [
            t["bbox"] + [t["category_id"]] for t in target
        ]  # required annotation format for albumentations

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)

        image = transformed["image"]
        boxes = transformed["bboxes"]

        new_boxes = []  # convert from xywh to xyxy
        for box in boxes:
            xmin = box[0]
            xmax = xmin + box[2]
            ymin = box[1]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(new_boxes, dtype=torch.float32)

        targ = {}  # here is our transformed target
        targ["boxes"] = boxes
        targ["labels"] = torch.tensor(
            [t["category_id"] for t in target], dtype=torch.int64
        )
        targ["image_id"] = torch.tensor([t["image_id"] for t in target])
        targ["area"] = (boxes[:, 3] - boxes[:, 1]) * (
            boxes[:, 2] - boxes[:, 0]
        )  # we have a different area
        targ["iscrowd"] = torch.tensor(
            [t["iscrowd"] for t in target], dtype=torch.int64
        )
        return image.div(255), targ  # scale images

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, batch):
        return tuple(zip(*batch))


class MaskDetectionModule(LightningModule):
    def __init__(self, lr=1e-3, batch_size=4, num_workers=4, usemodel="fasterrcnn"):
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.usemodel = usemodel

        self.train_dataset = MaskDetection(
            root="maskdetection-3", split="train", transform=transformation()
        )
        self.val_dataset = MaskDetection(
            root="maskdetection-3", split="valid", transform=transformation()
        )
        self.test_dataset = MaskDetection(
            root="maskdetection-3", split="test", transform=transformation()
        )

        self.save_hyperparameters(ignore=["model"])

        """LOAD MODEL"""
        if self.usemodel == "fasterrcnn":
            net = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights="DEFAULT"
            )
            n_features = net.roi_heads.box_predictor.cls_score.in_features
            net.roi_heads.box_predictor = (
                torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                    in_features, n_classes
                )
            )
            self.model = net

        elif self.usemodel == "retinanet":
            num_classes = 2
            # Load the network
            net = torchvision.models.detection.retinanet_resnet50_fpn(weights="COCO_V1")
            # Get the params
            in_features = net.head.classification_head.conv[0][0].in_channels
            num_anchors = net.head.classification_head.num_anchors
            # Change the classes
            net.head.classification_head.num_classes = num_classes

            # Change the cls_logits
            cls_logits = nn.Conv2d(
                in_features,
                num_anchors * num_classes,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            nn.init.normal_(cls_logits.weight, std=0.01)
            nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))

            # Assign to the net
            net.head.classification_head.cls_logits = cls_logits
            self.model = net
        """END OF LOAD MODEL"""

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True,
        )
        return opt

    def train_dataloader(self):
        dl = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
        )
        return dl

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: torch.tensor(v) for k, v in t.items()} for t in targets]
        # print(f'Image ID: {targets[0]["image_id"]}')

        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()

        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)

        self.log(
            "train_loss",
            losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return losses

    def val_dataloader(self):
        dl = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn,
        )
        return dl

    def validation_step(self, batch, batch_idx):
        # print(f'Batch {batch_idx}')
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: torch.tensor(v) for k, v in t.items()} for t in targets]
        # print(f'targets: {targets}')
        out = self.model(images)
        # print(f'out: {out}')
        batch_iou_list = []
        for i in range(len(out)):
            pred_bbox = out[i]["boxes"][0].expand(1, 4)
            pred_score = out[i]["scores"][0]
            target_bbox = targets[i]["boxes"]
            # print(f'pred_bbox: {pred_bbox}')
            # print(f'pred_score: {pred_score}')
            # print(f'target_bbox: {target_bbox}')

            iou = box_iou(pred_bbox, target_bbox)
            # print(f'IOU: {iou.item()}')
            batch_iou_list.append(iou.item())
        batch_iou = sum(batch_iou_list) / len(batch_iou_list)
        # print(f'Batch IoU: {batch_iou}')
        self.log(
            "val_iou",
            batch_iou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return batch_iou

    def test_dataloader(self):
        dl = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset.collate_fn,
        )
        return dl

    def test_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: torch.tensor(v) for k, v in t.items()} for t in targets]
        out = self.model(images)
        batch_iou_list = []
        for i in range(len(out)):
            pred_bbox = out[i]["boxes"][0].expand(1, 4)
            pred_score = out[i]["scores"][0]
            target_bbox = targets[i]["boxes"]
            iou = box_iou(pred_bbox, target_bbox)
            batch_iou_list.append(iou.item())
        batch_iou = sum(batch_iou_list) / len(batch_iou_list)
        print(f"Batch IoU: {batch_iou}")
        self.log(
            "test_iou",
            batch_iou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return batch_iou


if __name__ == "__main__":
    """PREPARATION"""
    all_losses = []
    all_losses_dict = []
    coco = COCO(
        os.path.join("maskdetection-3", "train", "_clean_annotations.coco.json")
    )
    categories = coco.cats
    n_classes = len(categories.keys())
    print(f"Number of classes: {n_classes}")

    """NOT IMPORTANT"""
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
    )

    """TRAINING"""
    seed_everything(42)
    model = MaskDetectionModule(
        lr=1e-3,
        batch_size=4,
        num_workers=8,
        usemodel="retinanet"
    )
    wandb_logger = WandbLogger(project="diza-mask-detection-thermal", log_model=False)
    wandb_logger.watch(model)
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=5,
        num_sanity_val_steps=0,
        precision=16,
        callbacks=[
            progress_bar,
        ],
        logger=wandb_logger,
        log_every_n_steps=5,
    )

    trainer.fit(model)  # untuk ngetrain dan validasi

    """Save Model"""
    model_savename = "model/maskdetection-3_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.save(model.model.state_dict(), model_savename + ".pt")

    # trainer.test(model) # <- still have y_max error || untuk testing

    testnet = torchvision.models.resnet18(weights="DEFAULT")