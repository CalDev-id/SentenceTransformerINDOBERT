import json
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query = self.data[idx]['query']
        corpus = self.data[idx]['corpus']


        inputs = self.tokenizer.encode_plus(
            query,
            corpus,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        token_type_ids = inputs['token_type_ids'].squeeze()
        return input_ids, attention_mask, token_type_ids

# Load data
with open('data.json') as f:
    data = json.load(f)

tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
dataset = TextDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
