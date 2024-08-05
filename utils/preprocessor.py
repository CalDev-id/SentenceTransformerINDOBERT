import os
import re
import sys
import torch
import emoji
import string
import multiprocessing
import pytorch_lightning as pl
import pandas as pd

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
        self.train_dataset_path = "datasets/train.csv"
        self.validation_dataset_path = "datasets/validation.csv"
        self.test_dataset_path = "datasets/test.csv"
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
            dataset_train = pd.read_csv(self.train_dataset_path)[["text", "label"]]
            dataset_valid = pd.read_csv(self.validation_dataset_path)[["text", "label"]]
            dataset_test = pd.read_csv(self.test_dataset_path)[["text", "label"]]

            # Add a 'step' column to identify the source (train, validation, test)
            dataset_train['step'] = 'train'
            dataset_valid['step'] = 'validation'
            dataset_test['step'] = 'test'

            # Concatenate all datasets into one
            dataset = pd.concat([dataset_train, dataset_valid, dataset_test], ignore_index=True)

            # Get stop words for Bahasa Indonesia using Sastrawi library
            self.stop_words = StopWordRemoverFactory().get_stop_words()

            # Clean and preprocess the 'text' column
            tqdm.pandas(desc='Preprocessing')
            dataset["text"] = dataset["text"].progress_apply(lambda x: self.clean_tweet(x))
            dataset.dropna(subset=['text'], inplace=True)
            print('[ Preprocess Completed ]\n')

            print('[ Saving Preprocessed Dataset ]')

            # Save the preprocessed dataset to a CSV file
            dataset.to_csv(self.processed_dataset_path, index=False)
            print('[ Save Completed ]\n')

        total_size = len(dataset.index)

        print('[ Tokenizing Dataset ]')

        # Initialize lists for tokenized inputs, attention masks, and labels
        train_x_input_ids, train_x_attention_mask, train_y = [], [], []
        valid_x_input_ids, valid_x_attention_mask, valid_y = [], [], []
        test_x_input_ids, test_x_attention_mask, test_y = [], [], []

        for (text, label, step) in tqdm(dataset.values.tolist()):
            # One-hot encode labels if specified
            if self.one_hot_label:
                default = [0]*2
                default[label] = 1
                label = default 

            # Tokenize the text using the provided tokenizer
            encoded_text = self.tokenizer(text=text,
                                          max_length=self.max_length,
                                          padding="max_length",
                                          truncation=True)
            
            # Append the tokenized input, attention mask, and label to the corresponding lists based on the source
            if step == 'train':
                train_x_input_ids.append(encoded_text['input_ids'])
                train_x_attention_mask.append(encoded_text['attention_mask'])
                train_y.append(label)
            elif step == 'validation':
                valid_x_input_ids.append(encoded_text['input_ids'])
                valid_x_attention_mask.append(encoded_text['attention_mask'])
                valid_y.append(label)
            elif step == 'test':
                test_x_input_ids.append(encoded_text['input_ids'])
                test_x_attention_mask.append(encoded_text['attention_mask'])
                test_y.append(label)

        # Convert lists to PyTorch tensors
        train_x_input_ids = torch.tensor(train_x_input_ids)
        train_x_attention_mask = torch.tensor(train_x_attention_mask)
        train_y = torch.tensor(train_y).float()

        valid_x_input_ids = torch.tensor(valid_x_input_ids)
        valid_x_attention_mask = torch.tensor(valid_x_attention_mask)
        valid_y = torch.tensor(valid_y).float()

        test_x_input_ids = torch.tensor(test_x_input_ids)
        test_x_attention_mask = torch.tensor(test_x_attention_mask)
        test_y = torch.tensor(test_y).float()

        del (dataset) # Release memory by deleting the dataset DataFrame

        # Create TensorDatasets for train, validation, and test sets
        train_dataset = TensorDataset(train_x_input_ids, train_x_attention_mask, train_y)
        valid_dataset = TensorDataset(valid_x_input_ids, valid_x_attention_mask, valid_y)
        test_dataset = TensorDataset(test_x_input_ids, test_x_attention_mask, test_y)
        print('[ Tokenize Completed ]\n')

        # print('[ Splitting Dataset ]')
        # train_validation_size = int(0.8 * total_size)
        # train_size = int(0.9 * train_validation_size)
        # validation_size = train_validation_size - train_size
        # test_size = total_size - train_validation_size

        # train_validation_dataset, test_dataset = torch.utils.data.random_split(tensor_dataset, [train_validation_size, test_size], generator=torch.Generator().manual_seed(42))
        # train_dataset, validation_dataset = torch.utils.data.random_split(train_validation_dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(42))
        # print('[ Split Completed ]\n')

        return train_dataset, valid_dataset, test_dataset
    
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

#step
    # data di inisialize dulu apakah ada yg sudah di proses atau belum, jika belum maka akan di concat dan diberi label test vali dan train
    # kemudian di save ke csv
    # kemudian di load lagi
    # kemudian di tokenisasi menjadi input_ids, attention_mask, dan label
    # kemudian di convert menjadi tensor
    # kemudian di buat menjadi tensor dataset
    # kemudian di split menjadi train, validation, dan test
    # kemudian di buat menjadi dataloader
    