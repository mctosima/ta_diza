from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from utils import *
from datareader import *
from lightningmodule import *
from model_list import *
from datetime import datetime
import argparse
from pytorch_lightning.loggers import WandbLogger


def run_training():
    seed_everything(args.seed)
    download_data("data/")

    # RUN NAME -----------------------------------------------
    run_name = f"{args.model}_{args.runname}"

    # MODEL SELECT -------------------------------------------
    net = model_selection(name=args.model, pretrained=args.pretrained)
    module = ModuleMaskDetection(model=net, lr=args.lr, batch_size=args.batch_size, num_workers=8, opt=args.opt, sched=args.sched)

    # WANDB LOGGING ------------------------------------------
    wandb_logger = WandbLogger(
        name=run_name,
        project="ta_diza",
        log_model=False
    )
    wandb_logger.experiment.config["model_name"] = args.model
    wandb_logger.experiment.config["pretrained"] = args.pretrained

    # LR MONITOR ---------------------------------------------
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # EARLY STOPPING -----------------------------------------
    earlystop = EarlyStopping(monitor="val_iou", patience=args.patience, mode="max")

    # TRAINER ------------------------------------------------
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        precision=16,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=[
            progress_bar,
            lr_monitor,
            earlystop,
        ],
    )
    train_start_time = datetime.now()
    trainer.fit(module)
    train_end_time = datetime.now()
    print("Training time: ", train_end_time - train_start_time)

    # save model
    torch.save(
        module.model.state_dict(),
        f"model/{run_name}.pth",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="To do the training process with some options"
    )
    parser.add_argument(
        "-lr", type=float, default=5e-4, help="Select the learning rate (Default:5e-4)" #0.0005
    )
    parser.add_argument(
        "-batch_size", type=int, default=8, help="Select the batch size (Default:8)"
    )
    parser.add_argument(
        "-epochs",
        type=int,
        default=20,
        help="Select the number of epochs (Default:10)",
    )
    parser.add_argument(
        "-model",
        type=str,
        default="fasterrcnn",
        help="Select the model (fasterrcnn, retinanet, ssdlite)",
    )
    parser.add_argument(
        "-pretrained",
        action="store_true",
        default=False,
        help="Select if you want to use pretrained weights (Default:False)",
    )
    parser.add_argument(
        "-seed", type=int, default=2022, help="Define the seed (Default:2022)"
    )
    parser.add_argument(
        "-runname", type=str, default=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Define the run name"
    )
    parser.add_argument(
        "-patience", type=int, default=2, help="Define the patience for early stopping"
    )

    parser.add_argument(
        "-opt", type=str, default="sgd", help="Define the optimizer (sgd, adam, rmsprop, adagrad, adamw), Default: sgd"
    )
    parser.add_argument(
        "-sched", type=str, default="constant", help="Define the scheduler (cosine, step, linear, exponential, plateau, constant). Default: constant"
    )
    args = parser.parse_args()

    # warning
    if args.model == "ssdlite" and args.pretrained:
        raise Exception("WARNING: SSDLite does not have pretrained weights")

    run_training()
