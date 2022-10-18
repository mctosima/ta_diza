from roboflow import Roboflow
import os
import albumentations as A
import numpy as np
import random
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

random.seed(2022)
np.random.seed(2022)


def download_data(path):
    if not os.path.exists(path):
        print("Downloading dataset...")
        rf = Roboflow(api_key="RLpF5qnVG3u4wi0Hgkmg")
        project = rf.workspace("diza-febriyan-hasal").project("maskdetection-tdrvn")
        dataset = project.version(5).download("coco", location=path)
        print("Dataset downloaded!")
    else:
        print("Dataset already exists!")


def train_augmentation():
    transform = A.Compose(
        [
            A.RandomScale(scale_limit=(0.8, 1.2), p=0.5),
            A.RandomGamma(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Rotate(limit=(-5, 5), p=0.5),
            A.Resize(80, 60),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco"),
    )
    return transform


def val_augmentation():
    transform = A.Compose(
        [A.Resize(80, 60), ToTensorV2()], bbox_params=A.BboxParams(format="coco")
    )
    return transform


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
