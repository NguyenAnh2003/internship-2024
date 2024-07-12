from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    AutoConfig,
)
from omegaconf import OmegaConf, DictConfig
from transformers import Trainer, TrainingArguments
from data_manipulation.dataloader import ABSADataset
import torch.nn.functional as F
from libs.helper_functions import get_configs
import numpy as np
import evaluate


class ABSAFineTuningModel:
    def __init__(self, conf: DictConfig = None) -> None:
        self.conf = OmegaConf.create(conf)

        # get pretrained model
        self.auto_conf = AutoConfig.from_pretrained(self.conf.model.pretrained.name)
        self.auto_conf.id2label = {
            i: label for i, label in enumerate(self.conf.model.label_aspects)
        }
        self.auto_conf.label2id = {
            label: i for i, label in enumerate(self.conf.model.label_aspects)
        }
        self.auto_conf.update({"label2id": self.auto_conf.label2id})
        self.auto_conf.update({"id2label": self.auto_conf.id2label})
        self.auto_conf.num_labels = 11
        self.model, self.tokenizer = self.get_pretrained_model_and_tokenizer()
        self.metric = evaluate.load("accuracy")

    def get_pretrained_model_and_tokenizer(self):
        # using transformers package to get pretrained model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.conf.model.pretrained.name,
            # id2label=self.auto_conf.id2label,
            # label2id=self.auto_conf.label2id,
            config=self.auto_conf,
        )  # get pretrained model name

        tokenizer = AutoTokenizer.from_pretrained(self.conf.model.pretrained.name)
        model.eval()  #

        return model, tokenizer

    def get_model_parameters(self):
        params = sum([p.nelement() for p in self.model.parameters()])
        return params  # sum of parameters

    def setup_dataset(self):
        absa = ABSADataset(
            tokenizer=self.tokenizer,
            csv_path=self.conf.model.train.train_dir,
            conf=self.conf,
        )
        train_set, dev_set, test_set = absa.setup_absa_hf_dataset()

        return train_set, dev_set, test_set

    def prepare_trainer4finetuning(self):
        if self.conf.model.pretrained.freeze == True:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return self.metric.compute(predictions=predictions, references=labels)

        # init dataset
        train_set, dev_set, test_set = self.setup_dataset()

        train_args = TrainingArguments(
            output_dir=self.conf.model.train.out_dir,
            num_train_epochs=self.conf.model.train.epoch,
            per_device_train_batch_size=self.conf.model.train.batch_size,
            per_device_eval_batch_size=self.conf.model.train.batch_size,
            evaluation_strategy="epoch",
            logging_dir=self.conf.model.train.log_dir,
            logging_strategy=self.conf.model.train.log_strategy,
            logging_steps=self.conf.model.train.log_steps,
            learning_rate=self.conf.model.train.lr,
            weight_decay=self.conf.model.train.weight_decay,
            warmup_steps=self.conf.model.train.warmup_step,
            report_to=self.conf.model.train.report_to,  # wandb
            push_to_hub=self.conf.model.train.push_to_hub,  # hub
            hub_strategy=self.conf.model.train.hub_strategy,  # hub
            hub_model_id=self.conf.model.train.hub_model_id,  # hub
            hub_token=self.conf.model.train.hub_token,  # hub
        )

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=train_args,
            train_dataset=train_set,
            eval_dataset=dev_set,
            compute_metrics=compute_metrics,
        )

        return trainer


if __name__ == "__main__":
    conf = get_configs("../../configs/absa_model.yaml")
    conf["model"]["pretrained"]["name"] = "FacebookAI/xlm-roberta-base"
    conf["model"]["train"][
        "train_dir"
    ] = "../../data_manipulation/metadata/manifests/ate/ate-manifest.json"
    conf["model"]["pretrained"]["freeze"] = True
    ab = ABSAFineTuningModel(conf)
    print(f"cac: {ab.get_model_parameters()}")
