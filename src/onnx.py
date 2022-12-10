from pytorch_lightning import LightningModule
from model_list import *
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import onnxruntime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from datetime import datetime


def to_onnx(namemodel, model):
    # class onnxmodule(LightningModule):
    #     def __init__(self, model, pretrained=False):
    #         super().__init__()
    #         selected_model = model
    #         pt = pretrained
    #         self.net = model_selection(name=model, pretrained=pt)

    #     def forward(self, x):
    #         return self.net(x)

    # model = onnxmodule(namemodel)
    # model.net.load_state_dict(torch.load("model/ssdlite_20221019_175945.pth"))

    # Create sample input
    SAMPLE_INPUT = "data/test/1-jay_jpg.rf.5f6c42fa8601409a6029d65a872db4e3.jpg"
    img = cv2.imread(SAMPLE_INPUT)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([ToTensorV2()])
    img = transform(image=img)["image"]
    img = img.div(255)
    img = img.detach()
    img = img.unsqueeze(0)
    print(img)
    print(img.shape)
    # save onnx model
    savepath = "model/" + namemodel + ".onnx"
    model.to_onnx(savepath, img, export_params=True)


def try_onnx():
    SAMPLE_INPUT = "data/test/1-jay_jpg.rf.5f6c42fa8601409a6029d65a872db4e3.jpg"
    start_time = datetime.now()
    img = cv2.imread(SAMPLE_INPUT)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([ToTensorV2()])
    img = transform(image=img)["image"]
    img = img.div(255)
    img = img.detach()
    img = img.unsqueeze(0)

    ort_session = onnxruntime.InferenceSession("model/ssdlite_20221027_001508.onnx")
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: img.detach().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    end_time = datetime.now()
    bbox = ort_outs[0][0]

    print(f"Inference Time: {end_time - start_time}")
    print(f"Bounding box: {bbox}")
    print(f"Saving image to out/onnx.png")

    plt.figure()
    plt.title("ONNX Inference")
    plt.imshow(img[0].permute(1, 2, 0))
    rect = patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    plt.gca().add_patch(rect)
    plt.savefig("out/onnx.png")


if __name__ == "__main__":
    # to_onnx()
    try_onnx()
