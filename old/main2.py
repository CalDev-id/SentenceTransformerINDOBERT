import os
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from utils.preprocessor3 import TwitterDataModule
from models.finetune2 import Finetune
from textwrap import dedent

def main():
    # Set seed for reproducibility
    seed_everything(42)

    # Define model parameters
    model_name = 'indolem/indobert-base-uncased'
    learning_rate = 2e-5
    batch_size = 32
    max_length = 128

    # Print parameter information
    print(dedent(f'''
    -----------------------------------
     Parameter Information        
    -----------------------------------
     Name                | Value       
    -----------------------------------
     Model Name          | {model_name}
     Batch Size          | {batch_size}
     Learning Rate       | {learning_rate}
     Input Max Length    | {max_length} 
    -----------------------------------
    '''))

    # Load pre-trained tokenizer and model
    pretrained_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, return_token_type_ids=True)
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name, 
        output_attentions=False, output_hidden_states=False, num_labels=2)
    model = Finetune(model=pretrained_model, learning_rate=learning_rate)

    # Initialize DataModule
    data_module = TwitterDataModule(tokenizer=pretrained_tokenizer, max_length=max_length, batch_size=batch_size, recreate=True)

    # Initialize callbacks and loggers
    tensor_board_logger = TensorBoardLogger('tensorboard_logs', name=f'{model_name}/{batch_size}_{learning_rate}')
    csv_logger = CSVLogger('csv_logs', name=f'{model_name}/{batch_size}_{learning_rate}')
    checkpoint_callback = ModelCheckpoint(dirpath=f'./checkpoints/{model_name}/{batch_size}_{learning_rate}', 
        monitor='val_loss', mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, check_on_train_epoch_end=1, 
        patience=3, mode='min')
    tqdm_progress_bar = TQDMProgressBar()

    # Initialize Trainer
    trainer = Trainer(
        accelerator='gpu',
        max_epochs=50,
        default_root_dir=f'./checkpoints/{model_name}/{batch_size}_{learning_rate}',
        callbacks=[checkpoint_callback, early_stop_callback, tqdm_progress_bar],
        logger=[tensor_board_logger, csv_logger],
        log_every_n_steps=5,
        deterministic=True  # To ensure reproducible results
    )

    # Train and test model
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path='best')

if __name__ == '__main__':
    main()
