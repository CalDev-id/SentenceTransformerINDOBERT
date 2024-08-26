import os
import re
import torch
import json
import emoji
import multiprocessing
import pytorch_lightning as pl

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


class TwitterDataModule(pl.LightningDataModule):

    def __init__(self, tokenizer, max_length=128, batch_size=32, recreate=False, one_hot_label=False) -> None:

        super(TwitterDataModule, self).__init__()

        self.seed = 42
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.recreate = recreate
        self.one_hot_label = one_hot_label
        self.train_dataset_path = "datasets/github/github_sentenced_combined.json"
        self.test_dataset_path = "datasets/turnbackhoax/turnbackhoax_sentenced_test.json"
        self.processed_dataset_path = "datasets/manual_processed.json"

    def load_data(self):
        if os.path.exists(self.processed_dataset_path) and not self.recreate:
            print('[ Loading Dataset ]')
            with open(self.processed_dataset_path, 'r') as f:
                dataset_train, dataset_test = json.load(f)
            print('[ Load Completed ]\n')
        else:
            print('[ Preprocessing Dataset ]')
            with open(self.train_dataset_path, 'r') as f:
                dataset_train = json.load(f)
            with open(self.test_dataset_path, 'r') as f:
                dataset_test = json.load(f)

            self.stop_words = StopWordRemoverFactory().get_stop_words()

            for entry in tqdm(dataset_train, desc='Preprocessing Train'):
                entry["corpus"] = self.clean_tweet(entry["corpus"])
            dataset_train = [entry for entry in dataset_train if entry["corpus"]]

            for entry in tqdm(dataset_test, desc='Preprocessing Test'):
                entry["corpus"] = self.clean_tweet(entry["corpus"])
            dataset_test = [entry for entry in dataset_test if entry["corpus"]]

            print('[ Preprocess Completed ]\n')

            print('[ Saving Preprocessed Dataset ]')
            with open(self.processed_dataset_path, 'w') as f:
                json.dump([dataset_train, dataset_test], f)
            print('[ Save Completed ]\n')

        train_input_ids, train_attention_mask, train_token_type_ids, train_labels = [], [], [], []
        test_input_ids, test_attention_mask, test_token_type_ids, test_labels = [], [], [], []

        for entry in tqdm(dataset_train, desc='Tokenizing Train'):
            label = entry['label']
            if self.one_hot_label:
                default = [0] * 2
                default[label] = 1
                label = default

            encoded_text = self.tokenizer.encode_plus(
                [entry['query'], entry['corpus']],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=True
            )

            train_input_ids.append(encoded_text['input_ids'])
            train_attention_mask.append(encoded_text['attention_mask'])
            train_token_type_ids.append(encoded_text['token_type_ids'])
            train_labels.append(label)

        for entry in tqdm(dataset_test, desc='Tokenizing Test'):
            label = entry['label']
            if self.one_hot_label:
                default = [0] * 2
                default[label] = 1
                label = default

            encoded_text = self.tokenizer.encode_plus(
                [entry['query'], entry['corpus']],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=True
            )

            test_input_ids.append(encoded_text['input_ids'])
            test_attention_mask.append(encoded_text['attention_mask'])
            test_token_type_ids.append(encoded_text['token_type_ids'])
            test_labels.append(label)

        train_input_ids = torch.tensor(train_input_ids)
        train_attention_mask = torch.tensor(train_attention_mask)
        train_token_type_ids = torch.tensor(train_token_type_ids)
        train_labels = torch.tensor(train_labels).float()

        test_input_ids = torch.tensor(test_input_ids)
        test_attention_mask = torch.tensor(test_attention_mask)
        test_token_type_ids = torch.tensor(test_token_type_ids)
        test_labels = torch.tensor(test_labels).float()

        train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_token_type_ids, train_labels)
        test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_token_type_ids, test_labels)

        return train_dataset, test_dataset

    def setup(self, stage=None):
        train_data, test_data = self.load_data()
        if stage == "fit":
            train_size = int(0.8 * len(train_data))
            valid_size = len(train_data) - train_size
            self.train_data, self.valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])
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

    def clean_tweet(self, text):
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

