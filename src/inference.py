from matplotlib.lines import Line2D
from pytorch_lightning import LightningModule
from model_list import *
from pycocotools.coco import COCO
import torch
import os
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.ops import box_iou
import argparse
from datetime import datetime

class SingleInference(LightningModule):
    def __init__(self, model="retinanet"):
        super().__init__()
        net = model_selection(model, pretrained=False)
        self.model = net

    def forward(self, x):
        return self.model(x)

def inference_from_dataset():
    coco = COCO(os.path.join("data", "test", "_annotations.coco.json"))
    img_id = list(sorted(coco.imgs.keys()))
    random_id = random.choice(img_id)
    img_path = coco.loadImgs(random_id)[0]["file_name"]
    print(f"Image: {img_path}")
    img = cv2.imread(os.path.join("data", "test", img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([ToTensorV2()])
    img = transform(image=img)["image"]
    img = img.div(255)

    bbox = coco.loadAnns(coco.getAnnIds(random_id))[0]["bbox"]
    bbox = [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]]
    print(f"Bounding Box: {bbox}")

    model.eval()
    preds = model([img])
    preds_bbox = preds[0]["boxes"][0].expand(1, 4)
    print(f"Predicted Bounding Box: {preds_bbox.detach().numpy()}")
    print(f'Predicted Confidence: {preds[0]["scores"][0].detach().numpy()}')
    iou = box_iou(preds_bbox, torch.tensor(bbox))
    print(f"IoU: {iou.squeeze().squeeze().detach().numpy()}")

    """PLOTTER"""
    plt.figure()
    plt.title("Prediction Results vs Ground Truth of image id: " + str(random_id))
    plt.suptitle(f"IOU: {iou.squeeze().squeeze().detach().numpy():.3f}")
    plt.imshow(img.permute(1, 2, 0))
    rect = patches.Rectangle(
        (bbox[0][0], bbox[0][1]),
        bbox[0][2] - bbox[0][0],
        bbox[0][3] - bbox[0][1],
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    plt.gca().add_patch(rect)


    preds_bbox = preds_bbox.detach().numpy()
    rect_pred = patches.Rectangle(
        (preds_bbox[0][0], preds_bbox[0][1]),
        preds_bbox[0][2] - preds_bbox[0][0],
        preds_bbox[0][3] - preds_bbox[0][1],
        linewidth=2,
        edgecolor="g",
        facecolor="none",
    )
    plt.gca().add_patch(rect_pred)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Ground Truth",
            markerfacecolor="r",
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Prediction",
            markerfacecolor="g",
            markersize=15,
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    if args.plotshow:
        plt.show()
    else:
        plt.savefig("out/image_inference.png")

def inference_from_image(img_path):
    start_time = datetime.now()
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([ToTensorV2()])
    img = transform(image=img)["image"]
    img = img.div(255)

    model.eval()
    preds = model([img])
    end_time = datetime.now()
    preds_bbox = preds[0]["boxes"][0].expand(1, 4)
    print(f"Predicted Bounding Box: {preds_bbox.detach().numpy()}")
    print(f'Predicted Confidence: {preds[0]["scores"][0].detach().numpy()}')

    plt.figure()
    plt.title("Prediction Results of Given Image:")
    plt.suptitle(f"Confidence: {preds[0]['scores'][0].detach().numpy():.3f}")
    plt.imshow(img.permute(1, 2, 0))

    preds_bbox = preds_bbox.detach().numpy()
    rect_pred = patches.Rectangle(
        (preds_bbox[0][0], preds_bbox[0][1]),
        preds_bbox[0][2] - preds_bbox[0][0],
        preds_bbox[0][3] - preds_bbox[0][1],
        linewidth=2,
        edgecolor="g",
        facecolor="none",
    )
    plt.gca().add_patch(rect_pred)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Prediction",
            markerfacecolor="g",
            markersize=15,
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    print(f"Inference Time: {end_time - start_time}")

    if args.plotshow:
        plt.show()
    else:
        plt.savefig("out/image_inference_fromfile.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference Mode"
    )
    parser.add_argument(
        "-plotshow", action="store_true", default=False, help="Show plot or save plot to output folder"
    )

    parser.add_argument(
        "-passimage", action="store_true", default=False, help="Use test dataset or define the image path"
    )

    parser.add_argument(
        "-model", type=str, default="retinanet", help="Model to use for inference"
    )

    parser.add_argument(
        "-pth", type=str, default="retinanet_20221018_133502.pth", help="Name of saved pytorch model"
    )

    parser.add_argument(
        "-img", type=str, default="data/train/1-ivy_jpg.rf.07d2bf7eaf34a1205afb75986e488063.jpg", help="Name of image to use for inference"
    )

    args = parser.parse_args()

    model = SingleInference(model=args.model)
    saved_model_path = f"model/{args.pth}"
    model.model.load_state_dict(torch.load(saved_model_path))

    if args.passimage:
        print("Inference from image")
        inference_from_image(args.img)
    else:
        print("Inference from test dataset")
        inference_from_dataset()

    