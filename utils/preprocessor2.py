import os
import re
import torch
import emoji
import json
import multiprocessing
import pytorch_lightning as pl
import pandas as pd

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


class TwitterDataModule(pl.LightningDataModule):

    def __init__(self, tokenizer, max_length=128, batch_size=32, recreate=False) -> None:
        super(TwitterDataModule, self).__init__()

        self.seed = 42
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.recreate = recreate
        self.train_dataset_path = "datasets/github/github_1.json"
        self.validation_dataset_path = "datasets/github/github_1.json"
        self.test_dataset_path = "datasets/github/github_1.json"
        self.processed_dataset_path = "datasets/twitter_label_manual_processed.csv"

    def load_data(self):
        # Load dataset if exists, else preprocess and save
        if os.path.exists(self.processed_dataset_path) and not self.recreate:
            print('[ Loading Dataset ]')
            dataset = pd.read_csv(self.processed_dataset_path)
            print('[ Load Completed ]\n')
        else:
            print('[ Preprocessing Dataset ]')
            # Read train, validation, and test datasets
            dataset_train = self.load_json(self.train_dataset_path)
            dataset_valid = self.load_json(self.validation_dataset_path)
            dataset_test = self.load_json(self.test_dataset_path)

            # Add a 'step' column to identify the source (train, validation, test)
            dataset_train['step'] = 'train'
            dataset_valid['step'] = 'validation'
            dataset_test['step'] = 'test'

            # Concatenate all datasets into one
            dataset = pd.concat([dataset_train, dataset_valid, dataset_test], ignore_index=True)

            # Get stop words for Bahasa Indonesia using Sastrawi library
            self.stop_words = StopWordRemoverFactory().get_stop_words()

            # Clean and preprocess the 'query' column
            tqdm.pandas(desc='Preprocessing')
            dataset["query"] = dataset["query"].progress_apply(lambda x: self.clean_tweet(x))
            dataset.dropna(subset=['query'], inplace=True)
            dataset["corpus"] = dataset["corpus"].progress_apply(lambda x: self.clean_tweet(x))
            dataset.dropna(subset=['corpus'], inplace=True)
            print('[ Preprocess Completed ]\n')

            print('[ Saving Preprocessed Dataset ]')

            # Save the preprocessed dataset to a CSV file
            dataset.to_csv(self.processed_dataset_path, index=False)
            print('[ Save Completed ]\n')

        total_size = len(dataset.index)

        print('[ Tokenizing Dataset ]')

        # Initialize lists for tokenized inputs, attention masks, and token type ids
        train_x_input_ids, train_x_attention_mask, train_x_token_type_ids = [], [], []
        valid_x_input_ids, valid_x_attention_mask, valid_x_token_type_ids = [], [], []
        test_x_input_ids, test_x_attention_mask, test_x_token_type_ids = [], [], []

        for (query, corpus, step) in tqdm(dataset.values.tolist()):
            # Tokenize the text using the provided tokenizer
            encoded_text = self.tokenizer.encode_plus([query, corpus],
                                          max_length=self.max_length,
                                          padding="max_length",
                                          truncation=True,
                                          return_token_type_ids=True)

            # Append the tokenized input, attention mask, and token type ids to the corresponding lists based on the source
            if step == 'train':
                train_x_input_ids.append(encoded_text['input_ids'])
                train_x_attention_mask.append(encoded_text['attention_mask'])
                train_x_token_type_ids.append(encoded_text['token_type_ids'])
            elif step == 'validation':
                valid_x_input_ids.append(encoded_text['input_ids'])
                valid_x_attention_mask.append(encoded_text['attention_mask'])
                valid_x_token_type_ids.append(encoded_text['token_type_ids'])
            elif step == 'test':
                test_x_input_ids.append(encoded_text['input_ids'])
                test_x_attention_mask.append(encoded_text['attention_mask'])
                test_x_token_type_ids.append(encoded_text['token_type_ids'])

        # Convert lists to PyTorch tensors
        train_x_input_ids = torch.tensor(train_x_input_ids)
        train_x_attention_mask = torch.tensor(train_x_attention_mask)
        train_x_token_type_ids = torch.tensor(train_x_token_type_ids)

        valid_x_input_ids = torch.tensor(valid_x_input_ids)
        valid_x_attention_mask = torch.tensor(valid_x_attention_mask)
        valid_x_token_type_ids = torch.tensor(valid_x_token_type_ids)

        test_x_input_ids = torch.tensor(test_x_input_ids)
        test_x_attention_mask = torch.tensor(test_x_attention_mask)
        test_x_token_type_ids = torch.tensor(test_x_token_type_ids)

        del (dataset)  # Release memory by deleting the dataset DataFrame

        # Create TensorDatasets for train, validation, and test sets
        train_dataset = TensorDataset(train_x_input_ids, train_x_attention_mask, train_x_token_type_ids)
        valid_dataset = TensorDataset(valid_x_input_ids, valid_x_attention_mask, valid_x_token_type_ids)
        test_dataset = TensorDataset(test_x_input_ids, test_x_attention_mask, test_x_token_type_ids)
        print('[ Tokenize Completed ]\n')

        return train_dataset, valid_dataset, test_dataset

    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)

    # Function to clean and preprocess a single tweet text
    def clean_tweet(self, text):
        result = text.lower()
        result = self.remove_emoji(result)  # remove emoji
        result = re.sub(r'\n', ' ', result)  # remove new line
        result = re.sub(r'@\w+', 'user', result)  # remove user mention
        result = re.sub(r'http\S+', '', result)  # remove link
        result = re.sub(r'\d+', '', result)  # remove number
        result = re.sub(r'[^a-zA-Z ]', '', result)  # get only alphabets
        result = ' '.join([word for word in result.split() if word not in self.stop_words])  # remove stopword
        result = result.strip()

        if result == '':
            result = float('NaN')

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

    # Return DataLoader for the training set
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count()
        )

    # Return DataLoader for the validation set
    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )

    # Return DataLoader for the test set
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )
