from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from omegaconf import OmegaConf, DictConfig
import pandas as pd

class ABSADataset(Dataset):
    def __init__(self, tokenizer=None, conf: DictConfig = None):
        super().__init__()
        self.conf = conf
        self.tokenizer = tokenizer  # tokenizer as transform

        # dataset
        self.dataset = self._create_hf_ds(
            path=self.conf.model.train.train_dir
        )  # create HF dataset
        self.dataset = self.dataset.train_test_split(test_size=0.2)

    def __getitem__(self, index):
        pass

    def setup_absa_hf_dataset(self):

        def _tokenize(batch):
            return self.tokenizer(
                batch["review"], padding=True, truncation=True, max_length=3000
            )

        train_dataset = self.dataset["train"]
        dev_dataset = self.dataset["test"].shard(num_shards=2, index=0)
        test_dataset = self.dataset["test"].shard(num_shards=2, index=0)

        train_dataset = train_dataset.map(lambda x: _tokenize(x))

        dev_dataset = dev_dataset.map(lambda x: _tokenize(x))

        test_dataset = test_dataset.map(lambda x: _tokenize(x))

        train_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )
        dev_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )
        test_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )

        return train_dataset, dev_dataset, test_dataset

    def _process_hf_ds(self, batch):
        # prepare HF dataset
        return batch

    @staticmethod
    def _create_hf_ds(path: str):
        train_csv = pd.read_csv(path)
        ds = HFDataset.from_pandas(train_csv)
        return ds

    def __len__(self):
        return len(self.dataset)