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

        print(f"Acc: {avg_acc:.4f} F1: {avg_f1:.4f} Precision: {avg_precision:.4f} Recall: {avg_recall:.4f}")

        pl_module.testing_step_outputs.clear()
