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
from sklearn.model_selection import train_test_split
import transformers

class DataModule(pl.LightningDataModule):

    def __init__(self, tokenizer, max_length=128, batch_size=32, recreate=False, one_hot_label=False) -> None:
        super(DataModule, self).__init__()

        self.seed = 42
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.recreate = recreate
        self.one_hot_label = one_hot_label

        # Path to the combined dataset
        self.dataset_path = "datasets/datasets.json"
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
            # Load the combined dataset
            with open(self.dataset_path, 'r') as f:
                dataset = json.load(f)

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

        # Initialize lists for tokenized inputs and labels
        x_input_ids, x_attention_mask, x_token_type_ids, y = [], [], [], []

        for entry in tqdm(dataset):
            query = entry['query']
            corpus = entry['corpus']
            label = entry['label']
            transformers.logging.set_verbosity_error()
            if self.one_hot_label:
                default = [0] * 2
                default[label] = 1
                label = default

            # Tokenization
            encoded_text = self.tokenizer.encode_plus(
                text=query,
                text_pair=corpus,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=True
            )

            # Store the tokenized data
            x_input_ids.append(encoded_text['input_ids'].squeeze(0))
            x_attention_mask.append(encoded_text['attention_mask'].squeeze(0))
            x_token_type_ids.append(encoded_text['token_type_ids'].squeeze(0))
            y.append(label)

        # Convert lists to PyTorch tensors
        x_input_ids = torch.stack(x_input_ids)
        x_attention_mask = torch.stack(x_attention_mask)
        x_token_type_ids = torch.stack(x_token_type_ids)
        y = torch.tensor(y).float()

        # Split the dataset into train, validation, and test sets
        x_train, x_temp, y_train, y_temp = train_test_split(
            x_input_ids, y, test_size=0.3, random_state=self.seed
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp, test_size=0.5, random_state=self.seed
        )

        # Create TensorDatasets for train, validation, and test sets
        train_dataset = TensorDataset(x_train, x_attention_mask[:len(x_train)], x_token_type_ids[:len(x_train)], y_train)
        valid_dataset = TensorDataset(x_val, x_attention_mask[len(x_train):len(x_train) + len(x_val)], x_token_type_ids[len(x_train):len(x_train) + len(x_val )], y_val)
        test_dataset = TensorDataset(x_test, x_attention_mask[len(x_train) + len(x_val):], x_token_type_ids[len(x_train) + len(x_val):], y_test)

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