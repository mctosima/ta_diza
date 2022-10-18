from datareader import *
from utils import *
from pytorch_lightning import LightningModule, Trainer
from torchvision.ops import box_iou
import numpy as np


class ModuleMaskDetection(LightningModule):
    def __init__(self, model, opt, sched, lr=1e-4, batch_size=4, num_workers=4):
        super().__init__()

        # model architecture
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_path = "data/"
        self.opt = opt
        self.scheduler = sched

        self.train_dataset = MaskReader(
            root=self.dataset_path, split="train", transform=train_augmentation()
        )
        self.val_dataset = MaskReader(
            root=self.dataset_path, split="valid", transform=val_augmentation()
        )
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        out = self.model(x)
        return out

    def configure_optimizers(self):
        if self.opt == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
        elif self.opt == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.opt == "rmsprop":
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        elif self.opt == "adagrad":
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.lr)
        elif self.opt == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        if self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
        elif self.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        elif self.scheduler == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, 0.001)
        elif self.scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        elif self.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)
        elif self.scheduler == None:
            return optimizer
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
        )
        return train_loader

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        return {"loss": losses}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=self.batch_size,
        )
        # print(f"Epoch : {self.current_epoch} | Loss : {loss}")

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn,
        )
        return val_loader

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        iou_list = []
        iou_batch = []

        output = self.model(images)
        for i in range(len(output)):
            pred_bbox = output[i]["boxes"][0].expand(1, 4)
            target_bbox = targets[i]["boxes"]
            iou = box_iou(pred_bbox, target_bbox)
            iou_list.append(iou)

        iou_batch.append(torch.stack(iou_list))
        return {"val_iou": iou_batch}

    def validation_epoch_end(self, outputs):
        outputs = outputs[0]["val_iou"][0].squeeze(2)
        iou = torch.mean(outputs)
        self.log(
            "val_iou",
            iou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        print(f"Validation on Epoch {self.current_epoch} | Average IOU : {iou:0.3f}")
