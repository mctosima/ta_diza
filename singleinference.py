from refactor import *
from matplotlib.lines import Line2D


class SingleInference(LightningModule):
    def __init__(self):
        super().__init__()
        net = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = net.roi_heads.box_predictor.cls_score.in_features
        net.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
        )
        self.model = net

    def forward(self, x):
        return self.model(x)


model = SingleInference()
model.model.load_state_dict(torch.load("maskdetection-3.pth"))

coco = COCO(os.path.join("maskdetection-3", "test", "_annotations.coco.json"))
img_id = list(sorted(coco.imgs.keys()))
random_id = random.choice(img_id)

img_path = coco.loadImgs(random_id)[0]["file_name"]
print(f"Image: {img_path}")
img = cv2.imread(os.path.join("maskdetection-3", "test", img_path))
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
print(f'Predicted Score: {preds[0]["scores"][0].detach().numpy()}')
iou = box_iou(preds_bbox, torch.tensor(bbox))

"""PLOTTER"""
plt.figure()
plt.title("Prediction Results vs Ground Truth of image id: " + str(random_id))
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


plt.show()
