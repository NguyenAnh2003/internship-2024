from typing import Any

from modules.absa_light_module import ABSALightningModule
from libs.helper_functions import get_configs
from build.model import ABSAModel
from pytorch_lightning import Trainer
from torch.nn import CrossEntropyLoss

if __name__ == "__main__":

  # init Trainer
  trainer = Trainer()

  conf = get_configs("../configs/absa_model.yaml")

  conf["model"]["pretrained"]["name"] = "FacebookAI/xlm-roberta-base"
  conf["model"]["pretrained"]["freeze"] = True
  conf["model"]["train"]["train_dir"] = "../data_manipulation/metadata/manifests/train.csv"
  conf["model"]["train"]["dev_dir"] = "../data_manipulation/metadata/manifests/val.csv"
  conf["model"]["train"]["test_dir"] = "../data_manipulation/metadata/manifests/test.csv"

  model = ABSAModel(conf)
  tokenizer = model.tokenizer
  module_absa = ABSALightningModule(tokenizer=tokenizer, conf=conf, model=model)

  # TRAIN, DEV SET
  train_ds, dev_ds = module_absa.setup_train_dataloader()

  loss_fn = CrossEntropyLoss()
  sample = next(iter(train_ds))
  input_ids = sample["input_ids"]
  attention_mask = sample["attention_mask"]
  labels = sample["labels"]
  logits = model(input_ids=input_ids, attention_mask=attention_mask)
  loss = loss_fn(logits, labels)
  print(loss)

  # TEST SET
  # test_ds = module_absa.setup_test_dataloader()
  # trainer.fit(module_absa, train_ds, dev_ds) # train


