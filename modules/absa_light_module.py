from pytorch_lightning import LightningModule
from torch.nn import Module, CrossEntropyLoss
from omegaconf import OmegaConf, DictConfig
from torch.optim import AdamW, lr_scheduler
from data_manipulation.dataloader import ABSADataset, ABSADataloader
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchmetrics.classification import (MulticlassAccuracy, MulticlassPrecision,
                                         MulticlassF1Score, MulticlassRecall)

class ABSALightningModule(LightningModule):
  def __init__(self, tokenizer, conf: DictConfig = None, model: Module = None):
    super().__init__()
    self.conf = OmegaConf.create(conf)
    self.tokenizer = tokenizer
    self.model = model
    self.loss = CrossEntropyLoss()
    self.acc_metric = MulticlassAccuracy(num_classes=10)

  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

    # metric
    loss = self.loss(logits, labels)
    acc = self.acc_metric()

    self.log_dict({'train_loss': loss})
    return loss

  def validation_step(self, batch, batch_idx):
    input_ids, attention_mask, labels = batch
    logits = self.model(input_ids, attention_mask)
    print(logits.shape)

  def configure_optimizers(self) -> OptimizerLRScheduler:
    optimizer = AdamW(self.model.parameters(), lr=self.conf.model.train.lr)
    return [optimizer]

  def setup_train_dataloader(self):
    train_ds = ABSADataset(tokenizer=self.tokenizer, path=self.conf.model.train.train_dir)
    dev_ds = ABSADataset(tokenizer=self.tokenizer, path=self.conf.model.train.dev_dir)
    train_dataloader = ABSADataloader(train_ds, batch_size=16, shuffle=True, tokenizer=self.tokenizer)
    dev_dataloader = ABSADataloader(dev_ds, batch_size=16, shuffle=True, tokenizer=self.tokenizer)
    return train_dataloader, dev_dataloader

  def setup_test_dataloader(self):
    test_ds = ABSADataset(tokenizer=self.tokenizer, path=self.conf.model.train.test_dir)
    test_dataloader = ABSADataloader(test_ds, batch_size=16, shuffle=True, tokenizer=self.tokenizer)
    return test_dataloader