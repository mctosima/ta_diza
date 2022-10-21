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
import argparse

def to_onnx(select_model, ptrained, pth, savepath):
    class onnxmodule(LightningModule):
        def __init__(self, model, pretrained):
            super().__init__()
            selected_model = model
            pt = pretrained
            self.net = model_selection(name=model, pretrained=pt)
            
        def forward(self, x):
            return self.net(x)

    model = onnxmodule(select_model, ptrained)
    pth_path = "./model/" + pth + ".pth"
    model.net.load_state_dict(torch.load(pth_path))

    # Create sample input
    SAMPLE_INPUT = "data/test/1-jay_jpg.rf.5f6c42fa8601409a6029d65a872db4e3.jpg"
    img = cv2.imread(SAMPLE_INPUT)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([ToTensorV2()])
    img = transform(image=img)["image"]
    img = img.div(255)
    img = img.detach()
    img = img.unsqueeze(0)
    # print(img)
    # print(img.shape)
    # save onnx model
    saveto = savepath + args.pth + ".onnx"
    model.to_onnx(saveto, img, export_params=True)

def try_onnx(onnxpath, imgpath):
    start_time = datetime.now()
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([ToTensorV2()])
    img = transform(image=img)["image"]
    img = img.div(255)
    img = img.detach()
    img = img.unsqueeze(0)

    ort_session = onnxruntime.InferenceSession(onnxpath)
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
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    plt.savefig("out/onnx.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert model to ONNX format"
    )

    parser.add_argument(
        "-mode", type=str, help="Convert to ONNX or try ONNX inference. (onnx, try)"
    )
    
    parser.add_argument(
        "-model", type=str, default="ssdlite", help="Model to convert to ONNX"
    )

    parser.add_argument(
        "-pretrained", action="store_true", default=False, help="Use pretrained weights"
    )

    parser.add_argument(
        "-pth", type=str, default="ssdlite_20221019_175945", help="pth name in model folder. Default: ssdlite_20221019_175945"
    )

    parser.add_argument(
        "-savepath", type=str, default="./model/", help="Path to save ONNX model. Default: ./model" 
    )

    parser.add_argument(
        "-onnxpath", type=str, help="Path to ONNX model"
    )

    parser.add_argument(
        "-imgpath", type=str, default="data/test/1-jay_jpg.rf.5f6c42fa8601409a6029d65a872db4e3.jpg", help="Path to image for ONNX inference"
    )

    args = parser.parse_args()
    # Parse Warning
    if args.pth == "":
        raise ValueError("Please specify path to model weights")
    
    if args.mode == "onnx":
        to_onnx(args.model, args.pretrained, args.pth, args.savepath)
    elif args.mode == "try":
        if args.onnxpath == "":
            raise ValueError("Please specify path to ONNX model")
        try_onnx(args.onnxpath, args.imgpath)
    else:
        raise ValueError("Please specify mode or Mode error")