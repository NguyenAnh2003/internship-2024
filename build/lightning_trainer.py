from typing import Any

import torch

from modules.absa_light_module import ABSALightningModule
from libs.helper_functions import get_configs
from build.model import ABSAModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from modules.printing_callback import PrintingCallback

def test_performance(model, checkpoint_dir, trainer, test_dataloader):
    checkpoint = torch.load(checkpoint_dir)
    state_dict = checkpoint["state_dict"]
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered_state_dict, strict=False)
    trainer.test(model=model, dataloaders=test_dataloader)


if __name__ == "__main__":
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    wandb_logger = WandbLogger(project="absa_")
    # # init Trainer
    trainer = Trainer(default_root_dir="checkpoints",
                      max_epochs=30, logger=wandb_logger,
                      log_every_n_steps=100,
                      precision="32",
                      callbacks=[lr_monitor, PrintingCallback()])

    conf = get_configs("../configs/absa_model.yaml")
    conf["model"]["train"]["lr"] = 0.0005
    conf["model"]["train"]["batch_size"] = 16

    conf["model"]["pretrained"]["name"] = "FacebookAI/xlm-roberta-large"
    conf["model"]["pretrained"]["freeze"] = True
    conf["model"]["train"][
        "train_dir"
    ] = "../data_manipulation/metadata/manifests/train.csv"
    conf["model"]["train"][
        "dev_dir"
    ] = "../data_manipulation/metadata/manifests/val.csv"
    conf["model"]["train"]["test_dir"] = "../data_manipulation/metadata/manifests/test.csv"

    model = ABSAModel(conf)
    wandb_logger.watch(model)
    tokenizer = model.tokenizer
    module_absa = ABSALightningModule(tokenizer=tokenizer, conf=conf, model=model)

    # TRAIN, DEV SET
    train_ds, dev_ds = module_absa.setup_train_dataloader()
    # trainer.fit(module_absa, train_ds, dev_ds)  # train

    # TEST SET
    test_ds = module_absa.setup_test_dataloader()

    checkpoint_dir = "./absa_/svwxj8o1/checkpoints/epoch=29-step=23610.ckpt"
    test_performance(model=module_absa, test_dataloader=test_ds,
                     trainer=trainer, checkpoint_dir=checkpoint_dir)
