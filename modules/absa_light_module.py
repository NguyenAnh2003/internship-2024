from typing import Union, IO, Optional, Any
from typing_extensions import Self
from lightning_fabric.utilities.types import _PATH, _MAP_LOCATION_TYPE
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torch.nn import Module, CrossEntropyLoss
from omegaconf import OmegaConf, DictConfig
from torch.optim import AdamW, Adam, lr_scheduler
from data_manipulation.dataloader import ABSADataset, ABSADataloader
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchmetrics.classification import (MulticlassAccuracy, MulticlassPrecision,
                                         MulticlassF1Score, MulticlassRecall)
from torchmetrics import Accuracy
import torch
import numpy as np

class ABSALightningModule(LightningModule):
    def __init__(self, tokenizer, conf: DictConfig = None, model = None):
        super().__init__()
        pl.seed_everything(1234)
        self.conf = OmegaConf.create(conf)
        self.tokenizer = tokenizer
        self.model = model
        self.loss = CrossEntropyLoss()

        # https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html
        self.acc_metric = Accuracy(task="multiclass", num_classes=10)
        self.precision_metric = MulticlassPrecision(num_classes=10, average="macro")
        self.recall_metric = MulticlassRecall(num_classes=10, average="macro")
        self.f1_metric = MulticlassF1Score(num_classes=10, average="macro")

        self.testing_step_outputs = []
        self.training_step_outputs = []
        self.validating_step_outputs = []

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path, *args, **kwargs,
    ):
        model = super().load_from_checkpoint(checkpoint_path, *args, **kwargs)
        return model

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # logits
        logits = self.forward(x1=input_ids, x2=attention_mask)
        loss = self.loss(logits, labels)

        # take argmax index from each sample
        # distribution on each label so have to take argmax
        predictions = logits.argmax(dim=-1)
        acc = self.acc_metric(predictions, labels)
        acc = acc.cpu().item()

        self.training_step_outputs.append({"loss": loss.item(), "acc": acc})

        # logging result
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # logits
        logits = self.forward(x1=input_ids, x2=attention_mask)
        loss = self.loss(logits, labels)

        # take argmax index from each sample
        # distribution on each label so have to take argmax
        predictions = logits.argmax(dim=-1)
        acc = self.acc_metric(predictions, labels)
        acc = acc.cpu().item()

        # append result
        self.validating_step_outputs.append({"loss": loss.item(), "acc": acc})

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # logits
        logits = self.forward(x1=input_ids, x2=attention_mask)

        # take argmax index from each sample
        # distribution on each label so have to take argmax
        predictions = logits.argmax(dim=-1)
        acc = self.acc_metric(predictions, labels)
        acc = acc.cpu().item() # convert from tensor to float : D

        p_score = self.precision_metric(predictions, labels)
        p_score = p_score.cpu().item()

        r_score = self.recall_metric(predictions, labels)
        r_score = r_score.cpu().item()

        f_score = self.f1_metric(predictions, labels)
        f_score = f_score.cpu().item()

        self.testing_step_outputs.append({"acc": acc, "f1": f_score,
                                          "precision": p_score, "recall": r_score})

        print(f"Acc - MulticlassAcc: {acc} Precision: {p_score} Recall: {r_score} F1: {f_score}")
        return acc

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          lr=self.conf.model.train.lr,
                          weight_decay=0.3)

        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def setup_train_dataloader(self):
        train_ds = ABSADataset(tokenizer=self.tokenizer, path=self.conf.model.train.train_dir)
        dev_ds = ABSADataset(tokenizer=self.tokenizer, path=self.conf.model.train.dev_dir)
        train_dataloader = ABSADataloader(train_ds, batch_size=self.conf.model.train.batch_size, shuffle=True,
                                          tokenizer=self.tokenizer)
        dev_dataloader = ABSADataloader(dev_ds, batch_size=self.conf.model.train.batch_size, shuffle=True,
                                        tokenizer=self.tokenizer)
        return train_dataloader, dev_dataloader

    def setup_test_dataloader(self):
        test_ds = ABSADataset(tokenizer=self.tokenizer, path=self.conf.model.train.test_dir)
        test_dataloader = ABSADataloader(test_ds, batch_size=self.conf.model.train.batch_size, shuffle=False,
                                         tokenizer=self.tokenizer)
        return test_dataloader
