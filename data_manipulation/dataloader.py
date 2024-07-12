from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datasets import Dataset as HFDataset
from omegaconf import OmegaConf, DictConfig
from libs.helper_functions import get_configs


class ABSADataset(Dataset):
    def __init__(self, tokenizer=None, conf: DictConfig = None) -> None:
        super().__init__()
        self.conf = OmegaConf.create(conf)
        self.tokenizer = tokenizer  # tokenizer as transform
        self.dataset = self._create_hf_ds(
            csv_path=self.conf.model.train.train_dir
        )  # create HF dataset
        if self.conf.model.style == "ate":
            self.dataset = self.dataset.map(self._process_hf_ds)  # mapping data
        elif self.conf.model.style == "instruction_tuning":
            self.dataset = self.dataset.map(self._processw_instruction_tuning_ds)

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

        train_dataset = train_dataset.map(
            lambda x: _tokenize(x), batched=True, batch_size=8
        )

        dev_dataset = dev_dataset.map(
            lambda x: _tokenize(x), batched=True, batch_size=8
        )

        test_dataset = test_dataset.map(
            lambda x: _tokenize(x), batched=True, batch_size=8
        )

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

    def _processw_instruction_tuning_ds(self, batch):
        # prepare HF dataset
        review = batch["review"]
        aspect = batch["aspect"]  # label2index
        opinion = batch["opinion"]  # tokenize this
        polarity = batch["polarity"]  # label2index
        prompt = ""
        # take review as input and aspect, polarity as output
        if self.conf.model.instruction_style == "simple":
            prompt = f"""###Instruct: Follow the given review and the
            corresponded output try to brainstorm and predict the aspect
            and polarity of that aspect in the review
            ###Input: {review}
            ###Output: aspect: {aspect} polarity: {polarity} opinion: {opinion}

            """
        special_symbol = "\n            "  # do not change "\n            "
        batch["prompt"] = prompt.replace(special_symbol, " ").strip()

        return batch

    def _process_hf_ds(self, batch):
        # prepare HF dataset
        review = batch["review"]
        aspect = batch["aspect"]

        return batch

    @staticmethod
    def _create_hf_ds(csv_path: str):
        train_csv = pd.read_csv(csv_path)
        ds = HFDataset.from_pandas(train_csv)
        return ds

    def __len__(self):
        return len(self.dataset)
