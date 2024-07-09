from torch.utils.data import Dataset, DataLoader


class ABSADataset(Dataset):
    def __init__(self, tokenizer, ds) -> None:
        super().__init__()
        self.dataset = ds
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        return super().__getitem__(index)

class ABSAloader(DataLoader):
    def __init__(self) -> None:
        super.__init__()
