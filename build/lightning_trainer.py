from typing import Any
from modules.absa_light_module import ABSALightningModule
from libs.helper_functions import get_configs
from build.model import ABSAModel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="absa_", log_model="all")
    # init Trainer
    trainer = Trainer(default_root_dir="checkpoints", max_epochs=3)

    conf = get_configs("../configs/absa_model.yaml")
    conf["model"]["train"]["lr"] = 2e-4

    conf["model"]["pretrained"]["name"] = "FacebookAI/xlm-roberta-base"
    conf["model"]["pretrained"]["freeze"] = True
    conf["model"]["train"][
        "train_dir"
    ] = "../data_manipulation/metadata/manifests/train.csv"
    conf["model"]["train"][
        "dev_dir"
    ] = "../data_manipulation/metadata/manifests/val.csv"
    conf["model"]["train"][
        "test_dir"
    ] = "../data_manipulation/metadata/manifests/test.csv"

    model = ABSAModel(conf)
    wandb_logger.watch(model)  # watch model grad
    tokenizer = model.tokenizer
    module_absa = ABSALightningModule(tokenizer=tokenizer, conf=conf, model=model)

    # TRAIN, DEV SET
    train_ds, dev_ds = module_absa.setup_train_dataloader()
    # TEST SET
    test_ds = module_absa.setup_test_dataloader()
    trainer.fit(module_absa, train_ds, dev_ds)  # train
