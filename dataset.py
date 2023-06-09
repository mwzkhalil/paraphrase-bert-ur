import torch
from torch.utils.data import Dataset


class UrduParaphraseDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['sentence1']
        paraphrase = self.data.iloc[index]['sentence2']

        encoded = self.tokenizer.encode_plus(
            text,
            paraphrase,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_length,
            truncation='only_first',
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
