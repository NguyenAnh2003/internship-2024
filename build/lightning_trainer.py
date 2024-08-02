from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from libs.helper_functions import get_configs
from build.model import ABSAModel
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from torch.nn import Module
from omegaconf import OmegaConf, DictConfig
from torch.optim import AdamW
from data_manipulation.dataloader import ABSADataset, DataLoader

class ABSALightningModule(LightningModule):
  def __init__(self, tokenizer, conf: DictConfig = None):
    self.conf = OmegaConf.create(conf)
    self.absa_ds_module = ABSADataset(tokenizer=tokenizer,
                                      conf=self.conf)

  def training_step(self, *args: Any, **kwargs: Any):
    pass

  def validation_step(self, *args: Any, **kwargs: Any):
    pass

  def configure_optimizers(self) -> OptimizerLRScheduler:
    optimizer = AdamW()
    return [optimizer]

  def set_up_dataloader(self):
    train_ds, dev_ds, test_ds = self.absa_ds_module.setup_absa_hf_dataset()
    # train_dataloader = DataLoader(train_ds)
    # dev_dataloader = DataLoader(dev_ds)
    # test_dataloader = DataLoader(test_ds)
    return train_ds, dev_ds, test_ds


if __name__ == "__main__":
  conf = get_configs("../configs/absa_model.yaml")

  conf["model"]["pretrained"]["name"] = "FacebookAI/xlm-roberta-base"
  conf["model"]["pretrained"]["freeze"] = True
  conf["model"]["train"]["train_dir"] = "../data_manipulation/metadata/manifests/train-ds.csv"

  model = ABSAModel(conf)
  tokenizer = model.tokenizer
  module_absa = ABSALightningModule(tokenizer=tokenizer, conf=conf)
  train_ds, dev_ds, test_ds = module_absa.set_up_dataloader()

  for point in enumerate(train_ds):
    print(point)


  # trainer = Trainer(max_epochs=20)

  # trainer.fit(model)
