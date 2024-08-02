from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from libs.helper_functions import get_configs
from build.model import ABSAModel
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule

class ABSALightningModule(LightningModule):
  def __init__(self):
    pass

  def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
    pass

  def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
    pass

  def configure_optimizers(self) -> OptimizerLRScheduler:
    pass



if __name__ == "__main__":
  conf = get_configs("../configs/absa_multi_task.yaml")

  # multi lingual model included Thai and En
  conf["model"]["pretrained"]["name"] = "google-bert/bert-base-multilingual-cased"

  model = ABSAModel(conf)

  sample = "Hello my name is"

  representation = model._embedding_of_input(sample)

  print(f"Parameters: {model.get_model_params()} "
        f"JJJ: {representation.shape}")
