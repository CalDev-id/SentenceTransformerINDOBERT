import torch
import pytorch_lightning as pl

class Finetune(pl.LightningModule):

    def __init__(self, model, learning_rate=2e-5) -> None:
        super(Finetune, self).__init__()
        self.model = model  # Use the initialized model
        self.lr = learning_rate  # Store learning rate

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Forward method for forward pass
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return model_output.logits

    def configure_optimizers(self):
        # Configure optimizers, using Adam optimizer in this case
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Training step method
        input_ids, attention_mask, token_type_ids = batch
        logits = self(input_ids, attention_mask, token_type_ids)
        return logits

    def validation_step(self, batch, batch_idx):
        # Validation step method
        input_ids, attention_mask, token_type_ids = batch
        logits = self(input_ids, attention_mask, token_type_ids)
        return logits

    def test_step(self, batch, batch_idx):
        # Test step method
        input_ids, attention_mask, token_type_ids = batch
        logits = self(input_ids, attention_mask, token_type_ids)
        return logits
