import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset


class ABSADataset(Dataset):
    def __init__(self, tokenizer=None, path: str = ""):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer  # tokenizer as transform

        # dataset
        self.dataset = self._create_hf_ds(path=self.path)  # create HF dataset

    def __getitem__(self, index):
        package = self._transform_sample(self.dataset[index])
        return package

    def _transform_sample(self, batch):
        tokens = self.tokenizer(
            batch["review"],
            padding=True,
            truncation=True,
            max_length=3000,
            return_attention_mask=True,
            return_tensors="pt",
        )

        tokens["labels"] = torch.tensor(batch["label"])

        return tokens

    @staticmethod
    def _create_hf_ds(path: str):
        train_csv = pd.read_csv(path)
        ds = HFDataset.from_pandas(train_csv)
        return ds

    def __len__(self):
        return len(self.dataset)


class ABSADataloader(DataLoader):
    def __init__(
        self,
        dataset: ABSADataset,
        batch_size: int = 16,
        shuffle: bool = True,
        tokenizer: AutoTokenizer = None,
    ):
        super(ABSADataloader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
        self.shuffle = shuffle
        self.collate_fn = self.collate_function
        self.tokenizer = tokenizer

    def collate_function(self, batch):
        input_ids = [item["input_ids"].squeeze(0) for item in batch]
        attention_mask = [item["attention_mask"].squeeze(0) for item in batch]
        labels = [item["labels"].squeeze(0) for item in batch]

        # Pad sequences
        input_ids_padded = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask_padded = pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels_padded = torch.tensor(
            labels
        )  # Assuming labels are already in tensor format and don't need padding

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
        }