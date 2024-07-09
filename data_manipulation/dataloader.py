from torch.utils.data import Dataset, DataLoader
import pandas as pd

class ABSADataset(Dataset):
    def __init__(self, tokenizer, ds) -> None:
        super().__init__()
        self.dataset = pd.read_csv(ds) # dataset as csv
        self.tokenizer = tokenizer # tokenizer as transform
        
    def __getitem__(self, index):
        
        return super().__getitem__(index)

    def __len__(self):
        return len(self.dataset)

class ABSAloader(DataLoader):
    def __init__(self) -> None:
        super.__init__()
