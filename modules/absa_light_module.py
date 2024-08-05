from pytorch_lightning import LightningModule
from torch.nn import Module, CrossEntropyLoss
from omegaconf import OmegaConf, DictConfig
from torch.optim import AdamW, Adam, lr_scheduler
from data_manipulation.dataloader import ABSADataset, ABSADataloader
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchmetrics.classification import (MulticlassAccuracy, MulticlassPrecision,
                                         MulticlassF1Score, MulticlassRecall)
from torchmetrics import Accuracy

class ABSALightningModule(LightningModule):
  def __init__(self, tokenizer, conf: DictConfig = None, model: Module = None):
    super().__init__()
    self.conf = OmegaConf.create(conf)
    self.tokenizer = tokenizer
    self.model = model
    self.loss = CrossEntropyLoss()

    # https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html
    self.acc_metric = Accuracy(task="multiclass", num_classes=10)

  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    # logits
    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
    loss = self.loss(logits, labels)

    # take argmax index from each sample
    # distribution on each label so have to take argmax
    predictions = logits.argmax(dim=-1)
    acc = self.acc_metric(predictions, labels)

    # logging result
    self.log_dict({'train/loss': loss, "train/acc": acc})
    return loss

  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    # logits
    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
    loss = self.loss(logits, labels)

    # take argmax index from each sample
    # distribution on each label so have to take argmax
    predictions = logits.argmax(dim=-1)
    acc = self.acc_metric(predictions, labels)

    # logging result
    self.log_dict({'val/loss': loss, "val/acc": acc})
    return loss

  def configure_optimizers(self):
    optimizer = AdamW(self.model.parameters(),
                      lr=self.conf.model.train.lr)

    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

  def setup_train_dataloader(self):
    train_ds = ABSADataset(tokenizer=self.tokenizer, path=self.conf.model.train.train_dir)
    dev_ds = ABSADataset(tokenizer=self.tokenizer, path=self.conf.model.train.dev_dir)
    train_dataloader = ABSADataloader(train_ds, batch_size=self.conf.model.train.batch_size, shuffle=True, tokenizer=self.tokenizer)
    dev_dataloader = ABSADataloader(dev_ds, batch_size=self.conf.model.train.batch_size, shuffle=True, tokenizer=self.tokenizer)
    return train_dataloader, dev_dataloader

  def setup_test_dataloader(self):
    test_ds = ABSADataset(tokenizer=self.tokenizer, path=self.conf.model.train.test_dir)
    test_dataloader = ABSADataloader(test_ds, batch_size=self.conf.model.train.batch_size, shuffle=True, tokenizer=self.tokenizer)
    return test_dataloader