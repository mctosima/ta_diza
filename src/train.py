from pytorch_lightning import seed_everything
from utils import *
from datareader import *
from lightningmodule import *
from model_list import *
import datetime
import argparse
from pytorch_lightning import seed_everything


def run_training():
    seed_everything(args.seed)
    download_data("data/")

    net = model_selection(name=args.model, pretrained=args.pretrained)

    module = ModuleMaskDetection(
        model=net, lr=args.lr, batch_size=args.batch_size, num_workers=8
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        precision=16,
        enable_progress_bar=True,
        callbacks=[
            progress_bar,
        ],
    )
    train_start_time = datetime.datetime.now()
    trainer.fit(module)
    train_end_time = datetime.datetime.now()
    print("Training time: ", train_end_time - train_start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="To do the training process with some options"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="Select the learning rate (Default:5e-4)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Select the batch size (Default:8)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Select the number of epochs (Default:10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fasterrcnn",
        help="Select the model (fasterrcnn, retinanet)",
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        default=True,
        help="Select if you want to use pretrained weights (Default:True)",
    )
    parser.add_argument(
        "--seed", type=int, default=2022, help="Select the seed (Default:2022)"
    )
    args = parser.parse_args()
    run_training()
