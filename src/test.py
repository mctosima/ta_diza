from pytorch_lightning import LightningModule, Trainer
from datareader import *
from utils import *
from torchvision.ops import box_iou
from model_list import *
import pandas as pd
import argparse


class ModuleMaskDetection(LightningModule):
    def __init__(self, model, batch_size=4, num_workers=4):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.iou = 0
        self.test_dataset = MaskReader(
            root="data/", split="test", transform=val_augmentation()
        )

    def forward(self, x):
        out = self.model(x)
        return out

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset.collate_fn,
        )
        return test_loader

    def test_step(self, batch, batch_idx):
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
        return {"test_iou": iou_batch}

    def test_epoch_end(self, outputs):
        outputs = outputs[0]["test_iou"][0].squeeze(2)
        iou = torch.mean(outputs)
        self.log(
            "test_iou",
            iou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.iou = iou.detach().cpu()
        print(f"Test Average IoU: {iou:0.3f}")


def run_testing():
    net = model_selection(args.model, pretrained=False)
    module = ModuleMaskDetection(model=net, batch_size=args.batch_size, num_workers=8)
    pth_path = f"model/{args.pth}"
    module.model.load_state_dict(torch.load(pth_path))
    trainer = Trainer(
        accelerator="gpu", devices=1, enable_progress_bar=True, precision=16
    )
    trainer.test(module)

    if args.exportres:
        print("Exporting Results...")
        # print the test_iou
        data = {
            "MODEL": [args.model],
            "SAVED_MODEL_NAME": [args.pth],
            "TEST_IOU": [module.iou.item()],
            "BatchSize":[args.batch_size],
        }

        df = pd.DataFrame(data)

        df.to_csv(f"out/test_list.csv", mode="a", index=True, header=False)

        print("Results exported to out/test_list.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing Mask Detection Model")

    parser.add_argument(
        "-model", type=str, default="retinanet", help="Model to use for testing"
    )

    parser.add_argument(
        "-pth",
        type=str,
        default="retinanet_20221018_133502.pth",
        help="Name of saved pytorch model (with .pth)",
    )

    parser.add_argument(
        "-batch_size", type=int, default=4, help="Batch size for testing"
    )

    parser.add_argument(
        "-exportres",
        action="store_true",
        default=False,
        help="Export the result to txt file",
    )

    args = parser.parse_args()
    run_testing()
