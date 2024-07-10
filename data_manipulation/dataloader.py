from torch.utils.data import Dataset, DataLoader
import pandas as pd

class ABSADataset(Dataset):
    def __init__(self, tokenizer, csv_path) -> None:
        super().__init__()
        self.dataset = pd.read_csv(csv_path) # dataset as csv
        self.tokenizer = tokenizer # tokenizer as transform
        
    def __getitem__(self, index):
        tokens = self.tokenizer.tokenize(self.dataset.iloc[index, 1])
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # attention_mask =
        return {"tokens": tokens, "ids": ids}

w
    def __len__(self):
        return len(self.dataset)

class ABSAloader(DataLoader):
    def __init__(self) -> None:
        super.__init__()