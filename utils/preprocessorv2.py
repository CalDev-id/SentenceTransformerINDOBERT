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

class DataModule(pl.LightningDataModule):

    def __init__(self, tokenizer, max_length=128, batch_size=32, recreate=False, one_hot_label=False) -> None:
        super(DataModule, self).__init__()

        self.seed = 42
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.recreate = recreate
        self.one_hot_label = one_hot_label
        self.train_dataset_path = "datasets/github/mendaley_tbh_train.json"
        self.validation_dataset_path = "datasets/github/valid_data.json"
        self.test_dataset_path = "datasets/github/github_sentenced_test.json"
        self.processed_dataset_path = "datasets/manual_processed.json"

    def load_data(self):
        # Load dataset if exists, else preprocess and save
        if os.path.exists(self.processed_dataset_path) and not self.recreate:
            print('[ Loading Dataset ]')
            with open(self.processed_dataset_path, 'r') as f:
                dataset = json.load(f)
            print('[ Load Completed ]\n')
        else:
            print('[ Preprocessing Dataset ]')
            # Read train, validation, and test datasets
            with open(self.train_dataset_path, 'r') as f:
                dataset_train = json.load(f)
            with open(self.validation_dataset_path, 'r') as f:
                dataset_valid = json.load(f)
            with open(self.test_dataset_path, 'r') as f:
                dataset_test = json.load(f)

            # Add a 'step' key to identify the source (train, validation, test)
            for item in dataset_train:
                item['step'] = 'train'
            for item in dataset_valid:
                item['step'] = 'validation'
            for item in dataset_test:
                item['step'] = 'test'

            # Concatenate all datasets into one
            dataset = dataset_train + dataset_valid + dataset_test

            # Get stop words for Bahasa Indonesia using Sastrawi library
            self.stop_words = StopWordRemoverFactory().get_stop_words()

            # Clean and preprocess the 'corpus' field
            tqdm.pandas(desc='Preprocessing')
            for item in tqdm(dataset, desc='Preprocessing'):
                item["corpus"] = self.clean_text(item["corpus"])

            # Remove any items with empty 'corpus' after cleaning
            dataset = [item for item in dataset if item['corpus']]

            print('[ Preprocess Completed ]\n')
            print('[ Saving Preprocessed Dataset ]')

            # Save the preprocessed dataset to a JSON file
            with open(self.processed_dataset_path, 'w') as f:
                json.dump(dataset, f)
            print('[ Save Completed ]\n')

        print('[ Tokenizing Dataset ]')

        # Initialize lists for tokenized inputs, attention masks, token type ids, and labels
        train_x_input_ids, train_x_attention_mask, train_x_token_type_ids, train_y = [], [], [], []
        valid_x_input_ids, valid_x_attention_mask, valid_x_token_type_ids, valid_y = [], [], [], []
        test_x_input_ids, test_x_attention_mask, test_x_token_type_ids, test_y = [], [], [], []

        for item in tqdm(dataset, desc='Tokenizing'):
            text = item["query"]
            corpus = item["corpus"]
            label = item["label"]
            step = item["step"]

            # One-hot encode labels if specified
            if self.one_hot_label:
                default = [0] * 2
                default[label] = 1
                label = default

            # Tokenize the text and corpus using the provided tokenizer
            encoded_text = self.tokenizer(text=text,
                                          text_pair=corpus,
                                          max_length=self.max_length,
                                          padding="max_length",
                                          truncation=True,
                                          return_tensors='pt')

            # Append the tokenized input, attention mask, token type ids, and label to the corresponding lists based on the source
            if step == 'train':
                train_x_input_ids.append(encoded_text['input_ids'].squeeze())
                train_x_attention_mask.append(encoded_text['attention_mask'].squeeze())
                train_x_token_type_ids.append(encoded_text['token_type_ids'].squeeze())
                train_y.append(label)
            elif step == 'validation':
                valid_x_input_ids.append(encoded_text['input_ids'].squeeze())
                valid_x_attention_mask.append(encoded_text['attention_mask'].squeeze())
                valid_x_token_type_ids.append(encoded_text['token_type_ids'].squeeze())
                valid_y.append(label)
            elif step == 'test':
                test_x_input_ids.append(encoded_text['input_ids'].squeeze())
                test_x_attention_mask.append(encoded_text['attention_mask'].squeeze())
                test_x_token_type_ids.append(encoded_text['token_type_ids'].squeeze())
                test_y.append(label)

        # Convert lists to PyTorch tensors
        train_x_input_ids = torch.stack(train_x_input_ids)
        train_x_attention_mask = torch.stack(train_x_attention_mask)
        train_x_token_type_ids = torch.stack(train_x_token_type_ids)
        train_y = torch.tensor(train_y).float()

        valid_x_input_ids = torch.stack(valid_x_input_ids)
        valid_x_attention_mask = torch.stack(valid_x_attention_mask)
        valid_x_token_type_ids = torch.stack(valid_x_token_type_ids)
        valid_y = torch.tensor(valid_y).float()

        test_x_input_ids = torch.stack(test_x_input_ids)
        test_x_attention_mask = torch.stack(test_x_attention_mask)
        test_x_token_type_ids = torch.stack(test_x_token_type_ids)
        test_y = torch.tensor(test_y).float()

        del (dataset)  # Release memory by deleting the dataset

        # Create TensorDatasets for train, validation, and test sets
        train_dataset = TensorDataset(train_x_input_ids, train_x_attention_mask, train_x_token_type_ids, train_y)
        valid_dataset = TensorDataset(valid_x_input_ids, valid_x_attention_mask, valid_x_token_type_ids, valid_y)
        test_dataset = TensorDataset(test_x_input_ids, test_x_attention_mask, test_x_token_type_ids, test_y)
        print('[ Tokenize Completed ]\n')

        return train_dataset, valid_dataset, test_dataset

    def clean_text(self, text):
        result = text.lower()
        result = self.remove_emoji(result)  # remove emoji
        result = re.sub(r'\n', ' ', result)  # remove new line
        result = re.sub(r'@\w+', 'user', result)  # remove user mention
        result = re.sub(r'http\S+', '', result)  # remove link
        result = re.sub(r'\d+', '', result)  # remove number
        result = re.sub(r'[^a-zA-Z ]', '', result)  # get only alphabets
        result = ' '.join([word for word in result.split() if word not in self.stop_words])  # remove stopwords
        result = result.strip()

        if result == '':
            result = None

        return result

    def remove_emoji(self, text):
        return emoji.replace_emoji(text, replace='')

    def setup(self, stage=None):
        # Load datasets during the setup phase
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
