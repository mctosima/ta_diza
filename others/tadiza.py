# Package Dasar Python
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import copy
import math

# Package Pytorch dan Pendukungnya
import torch
import torchvision
from torchvision import datasets, models
from torchvision.utils import draw_bounding_boxes
from torch.utils.data import Dataset, DataLoader

from albumentations.pytorch import ToTensorV2
from torchvision.transforms import transforms as T

# Pytorch Lightning dkk
from pytorch_lightning import LightningModule, Trainer


# Package Utilities
from pycocotools.coco import COCO
from roboflow import Roboflow
import albumentations as A

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

class ModuleFasterRCNN(LightningModule):
    def __init__(self, model, lr=1e-3, batch_size=4, num_workers=4):
        # init untuk mengatur variabel yang akan digunakan
        super().__init__()

        # model architec
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_path = "maskdetection-3"

        # load data
        self.train_dataset = MaskDetection(
            root=self.dataset_path, split="train", transform=transformation()
        )
        self.val_dataset = MaskDetection(
            root=self.dataset_path, split="valid", transform=transformation()
        )
        self.test_dataset = MaskDetection(
            root=self.dataset_path, split="test", transform=transformation()
        )

    def forward(self, x):
        # digunakan untuk mengatur bagaimana model akan melakukan forward pass (selalu sama)
        out = self.model(x)
        return out

    def configure_optimizers(self):
        # digunakan untuk mengatur optimizer dan scheduler
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True,
        )
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1)
        return optimizer
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def train_dataloader(self):
        # memuat dataloader untuk train
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
        )
        return train_loader

    def training_step(self, batch, batch_idx):
        # memuat cara untuk melakukan training
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        # menghitung loss
        # Pada FasterRCNN, model akan mengembalikan loss ketika digunakan dalam mode training
        # Mode Training: memberikan gambar dan target ke model
        # Sementara itu, FasterRCNN akan mengembalikan bounding box apabila digunakan dalam mode
        # Validasi dan testing, yaitu dengan memberikan hanya gambar saja ke model

        loss_dict = self.model(images, targets)
        # print(f"KELUARAN DARI TRAINING ->> {loss_dict}")
        losses = sum(loss for loss in loss_dict.values())

        

        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()

        all_losses.append(loss_value)  # coba dihapus
        all_losses_dict.append(loss_dict_append)  # coba dihapus
        self.log("train_loss", losses, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        # Kalau di collab gak jalan self.log nya, di print manual saja lossesnya
        return losses

    def val_dataloader(self):
        # memuat dataloader untuk validasi
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn,
        )
        return val_loader

    def validation_step(self, batch, batch_idx):
        # memuat cara untuk melakukan validasi
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        output = self.model(images)
        print(f"KELUARAN DARI VALIDASI ->> {output}")

        batch_iou_list = []
        for i in range(len(output)):
            pred_bbox = output[i]["boxes"][0].expand(1, 4)
            pred_score = output[i]["scores"][0]
            target_bbox = targets[i]["boxes"]

            iou = torchvision.ops.box_iou(pred_bbox, target_bbox)

        batch_iou = iou.mean().item()
        self.log("val_iou", batch_iou, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        # Kalau di collab gak jalan self.log nya, di print manual saja lossesnya
        return batch_iou

    def test_dataloader(self):
        # memuat dataloader untuk testing
        pass

    def test_step(self, batch, batch_idx):
        # memuat cara untuk melakukan testing
        pass

if __name__ == "__main__":
    all_losses = []
    all_losses_dict = []

    net = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = net.roi_heads.box_predictor.cls_score.in_features
    net.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
    )

    module = ModuleFasterRCNN(model=net, lr=1e-4, batch_size=8, num_workers=8)
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=1,
        num_sanity_val_steps=0,
        precision=16,
        enable_progress_bar=True,
    )
    trainer.fit(module)