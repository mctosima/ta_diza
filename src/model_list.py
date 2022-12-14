import torch.nn as nn
import torchvision
import math
from functools import partial
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead


def model_selection(name, pretrained):  # TODO: Add more models
    if name == "fasterrcnn":
        if pretrained:
            net = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights="DEFAULT", trainable_backbone_layers=0
            )
            in_features = net.roi_heads.box_predictor.cls_score.in_features
            net.roi_heads.box_predictor = (
                torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                    in_features, 2
                )
            )
        else:
            net = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                num_classes=2, trainable_backbone_layers=0
            )

        return net

    elif name == "retinanet":
        num_classes = 2
        if pretrained:
            net = torchvision.models.detection.retinanet_resnet50_fpn(
                weights="COCO_V1", trainable_backbone_layers=0
            )
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

        else:
            net = torchvision.models.detection.retinanet_resnet50_fpn(
                num_classes=2, trainable_backbone_layers=0
            )

        return net

    elif name == "ssdlite":
        num_classes = 2
        if pretrained:
            net = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
                weights="COCO_V1", trainable_backbone_layers=0
            )

            in_channels = det_utils.retrieve_out_channels(net.backbone, (320, 320))
            num_anchors = net.anchor_generator.num_anchors_per_location()
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
            num_classes = 2
            net.head.classification_head = SSDLiteClassificationHead(
                in_channels, num_anchors, num_classes, norm_layer
            )

        else:
            net = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
                num_classes=2, trainable_backbone_layers=0
            )

        return net

    elif name == "ssd":
        net = torchvision.models.detection.ssd300_vgg16(
            num_classes=2, trainable_backbone_layers=0
        )
        return net

    elif name == "fcos":
        net = torchvision.models.detection.fcos_resnet50_fpn(
            num_classes=2, trainable_backbone_layer=0
        )
        return net

    else:
        raise ValueError("Model not found")
