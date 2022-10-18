import torch.nn as nn
import torchvision
import math


def model_selection(name, pretrained):  # TODO: Add more models
    if name == "fasterrcnn":
        if pretrained:
            net = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights="DEFAULT"
            )
        else:
            net = torchvision.models.detection.fasterrcnn_resnet50_fpn()
        in_features = net.roi_heads.box_predictor.cls_score.in_features
        net.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
        )
        return net

    elif name == "retinanet":
        num_classes = 2
        if pretrained:
            net = torchvision.models.detection.retinanet_resnet50_fpn(weights="COCO_V1")
        else:
            net = torchvision.models.detection.retinanet_resnet50_fpn()
        in_features = net.head.classification_head.conv[0][0].in_channels
        num_anchors = net.head.classification_head.num_anchors

        net.head.classification_head.num_classes = num_classes

        cls_logits = nn.Conv2d(
            in_features,
            num_anchors * num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        nn.init.normal_(cls_logits.weight, std=0.01)
        nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))

        net.head.classification_head.cls_logits = cls_logits
        return net

    else:
        raise ValueError("Model not found")
