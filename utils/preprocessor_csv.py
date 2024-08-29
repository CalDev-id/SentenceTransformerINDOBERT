import os
import re
import torch
import emoji
import pandas as pd
import multiprocessing
import pytorch_lightning as pl

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class DataModule(pl.LightningDataModule):

    def __init__(self, tokenizer, max_length=128, batch_size=32, recreate=False, one_hot_label=False) -> None:
        super(DataModule, self).__init__()

        self.seed = 42
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.recreate = recreate
        self.one_hot_label = one_hot_label
        self.train_dataset_path = "datasets/github/github_train.json"
        self.validation_dataset_path = "datasets/github/github_val.json"
        self.test_dataset_path = "datasets/github/github_test.json"
        self.processed_dataset_path = "datasets/github/github_preprocessed.json"

    def load_data(self):
        # Load dataset if exists, else preprocess and save
        if os.path.exists(self.processed_dataset_path) and not self.recreate:
            print('[ Loading Dataset ]')
            dataset = pd.read_csv(self.processed_dataset_path)
            print('[ Load Completed ]\n')
        else:
            print('[ Preprocessing Dataset ]')
            dataset_train = pd.read_csv(self.train_dataset_path)[["query", "label", "corpus"]]
            dataset_valid = pd.read_csv(self.validation_dataset_path)[["query", "label", "corpus"]]
            dataset_test = pd.read_csv(self.test_dataset_path)[["query", "label", "corpus"]]

            dataset_train['step'] = 'train'
            dataset_valid['step'] = 'validation'
            dataset_test['step'] = 'test'

            dataset = pd.concat([dataset_train, dataset_valid, dataset_test], ignore_index=True)

            self.stop_words = StopWordRemoverFactory().get_stop_words()
            tqdm.pandas(desc='Preprocessing')
            dataset["query"] = dataset["query"].progress_apply(lambda x: self.clean_text(x))
            dataset["corpus"] = dataset["corpus"].progress_apply(lambda x: self.clean_text(x))
            dataset.dropna(subset=['query', 'corpus'], inplace=True)
            print('[ Preprocess Completed ]\n')

            print('[ Saving Preprocessed Dataset ]')
            dataset.to_csv(self.processed_dataset_path, index=False)
            print('[ Save Completed ]\n')

        total_size = len(dataset.index)

        print('[ Tokenizing Dataset ]')

        train_x_input_ids, train_x_attention_mask, train_x_token_type_ids, train_y = [], [], [], []
        valid_x_input_ids, valid_x_attention_mask, valid_x_token_type_ids, valid_y = [], [], [], []
        test_x_input_ids, test_x_attention_mask, test_x_token_type_ids, test_y = [], [], [], []

        for (query, label, corpus, step) in tqdm(dataset.values.tolist()):
            if self.one_hot_label:
                default = [0]*2
                default[label] = 1
                label = default 

            encoded_text = self.tokenizer.encode_plus(text=query,
                                          text_pair=corpus,  # text_pair for query-corpus pairing
                                          max_length=self.max_length,
                                          padding="max_length",
                                          truncation=True,
                                          return_token_type_ids=True)  # include token_type_ids
            
            if step == 'train':
                train_x_input_ids.append(encoded_text['input_ids'])
                train_x_attention_mask.append(encoded_text['attention_mask'])
                train_x_token_type_ids.append(encoded_text['token_type_ids'])
                train_y.append(label)
            elif step == 'validation':
                valid_x_input_ids.append(encoded_text['input_ids'])
                valid_x_attention_mask.append(encoded_text['attention_mask'])
                valid_x_token_type_ids.append(encoded_text['token_type_ids'])
                valid_y.append(label)
            elif step == 'test':
                test_x_input_ids.append(encoded_text['input_ids'])
                test_x_attention_mask.append(encoded_text['attention_mask'])
                test_x_token_type_ids.append(encoded_text['token_type_ids'])
                test_y.append(label)

        train_x_input_ids = torch.tensor(train_x_input_ids)
        train_x_attention_mask = torch.tensor(train_x_attention_mask)
        train_x_token_type_ids = torch.tensor(train_x_token_type_ids)
        train_y = torch.tensor(train_y).float()

        valid_x_input_ids = torch.tensor(valid_x_input_ids)
        valid_x_attention_mask = torch.tensor(valid_x_attention_mask)
        valid_x_token_type_ids = torch.tensor(valid_x_token_type_ids)
        valid_y = torch.tensor(valid_y).float()

        test_x_input_ids = torch.tensor(test_x_input_ids)
        test_x_attention_mask = torch.tensor(test_x_attention_mask)
        test_x_token_type_ids = torch.tensor(test_x_token_type_ids)
        test_y = torch.tensor(test_y).float()

        del (dataset)

        train_dataset = TensorDataset(train_x_input_ids, train_x_attention_mask, train_x_token_type_ids, train_y)
        valid_dataset = TensorDataset(valid_x_input_ids, valid_x_attention_mask, valid_x_token_type_ids, valid_y)
        test_dataset = TensorDataset(test_x_input_ids, test_x_attention_mask, test_x_token_type_ids, test_y)
        print('[ Tokenize Completed ]\n')

        return train_dataset, valid_dataset, test_dataset

    def clean_text(self, text):
        result = text.lower()
        result = self.remove_emoji(result)
        result = re.sub(r'\n', ' ', result)
        result = re.sub(r'@\w+', 'user', result)
        result = re.sub(r'http\S+', '', result)
        result = re.sub(r'\d+', '', result)
        result = re.sub(r'[^a-zA-Z ]', '', result)
        result = ' '.join([word for word in result.split() if word not in self.stop_words])
        result = result.strip()

        if result == '':
            result = float('NaN')

        return result

    def remove_emoji(self, text):
        return emoji.replace_emoji(text, replace='')

    def setup(self, stage=None):
        train_data, valid_data, test_data = self.load_data()
        if stage == "fit":
            self.train_data = train_data
            self.valid_data = valid_data
        elif stage == "test":
            self.test_data = test_data

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count()
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )
