from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import numpy as np


class PrintingCallback(Callback):
    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        accuracies = [x["acc"] for x in pl_module.testing_step_outputs]
        f1_scores = [x["f1"] for x in pl_module.testing_step_outputs]
        precisions = [x["precision"] for x in pl_module.testing_step_outputs]
        recall = [x["recall"] for x in pl_module.testing_step_outputs]

        avg_acc = np.mean(accuracies)
        avg_f1 = np.mean(f1_scores)
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recall)

        print(f"Acc: {avg_acc} F1: {avg_f1} Precision: {avg_precision} Recall: {avg_recall}")

        pl_module.testing_step_outputs.clear()
