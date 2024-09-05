import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import classification_report

class Finetune(pl.LightningModule):

    def __init__(self, model, learning_rate=2e-5) -> None:
        super(Finetune, self).__init__()
        self.model = model
        self.lr = learning_rate

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        if labels is not None:
            model_output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            return model_output.loss, model_output.logits
        else:
            model_output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            return model_output.logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, targets = batch
        loss, logits = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=targets
        )

        metrics = {}
        metrics['train_loss'] = loss.item()
        self.log_dict(metrics, prog_bar=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, true, pred = self._shared_eval_step(batch, batch_idx)
        return loss, true, pred

    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.Tensor().to(device='cuda')
        true = []
        pred = []

        for output in validation_step_outputs:
            loss = torch.cat((loss, output[0].view(1)), dim=0)
            true += output[1].numpy().tolist()
            pred += output[2].numpy().tolist()

        loss = torch.mean(loss)

        cls_report = classification_report(true, pred, labels=[0, 1], output_dict=True)

        accuracy = cls_report['accuracy']
        f1_score = cls_report['1']['f1-score']
        precision = cls_report['1']['precision']
        recall = cls_report['1']['recall']

        metrics = {}
        metrics['val_loss'] = loss.item()
        metrics['val_accuracy'] = accuracy
        metrics['val_f1_score'] = f1_score
        metrics['val_precision'] = precision
        metrics['val_recall'] = recall

        print()
        print(metrics)

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, true, pred = self._shared_eval_step(batch, batch_idx)
        return loss, true, pred

    def test_epoch_end(self, test_step_outputs):
        loss = torch.Tensor().to(device='cuda')
        true = []
        pred = []

        for output in test_step_outputs:
            loss = torch.cat((loss, output[0].view(1)), dim=0)
            true += output[1].numpy().tolist()
            pred += output[2].numpy().tolist()

        loss = torch.mean(loss)

        cls_report = classification_report(true, pred, labels=[0, 1], output_dict=True)

        accuracy = cls_report['accuracy']
        f1_score = cls_report['1']['f1-score']
        precision = cls_report['1']['precision']
        recall = cls_report['1']['recall']

        metrics = {}
        metrics['test_loss'] = loss.item()
        metrics['test_accuracy'] = accuracy
        metrics['test_f1_score'] = f1_score
        metrics['test_precision'] = precision
        metrics['test_recall'] = recall

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

        return loss

    def _shared_eval_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, targets = batch
        loss, logits = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=targets
        )

        true = torch.argmax(targets, dim=1).to(torch.device("cpu"))
        pred = torch.argmax(logits, dim=1).to(torch.device("cpu"))

        return loss, true, pred

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids = batch
        logits = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pred = torch.argmax(logits, dim=1).to(torch.device("cpu"))

        return pred[0]

#==================================================================================================

import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import classification_report

class FinetuneV2(pl.LightningModule):
    def __init__(self, model, learning_rate=2e-5) -> None:
        super(FinetuneV2, self).__init__()
        self.model = model
        self.lr = learning_rate

        self.linear1 = nn.Linear(768, 32)
        self.linear2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        linear_output = self.linear1(model_output.pooler_output)
        relu_output = self.relu(linear_output)
        linear_output = self.linear2(relu_output)
        return linear_output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, targets = batch
        outputs = torch.squeeze(self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids), dim=1)

        loss = self.criterion(outputs, targets)

        metrics = {}
        metrics['train_loss'] = loss.item()

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, true, pred = self._shared_eval_step(batch, batch_idx)
        return loss, true, pred

    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.Tensor().to(device='cuda')
        true = []
        pred = []

        for output in validation_step_outputs:
            loss = torch.cat((loss, output[0].view(1)), dim=0)
            true += output[1].numpy().tolist()
            pred += output[2].numpy().tolist()

        loss = torch.mean(loss)

        cls_report = classification_report(true, pred, labels=[0, 1], output_dict=True, zero_division=0)

        accuracy = cls_report['accuracy']
        f1_score = cls_report['1']['f1-score']
        precision = cls_report['1']['precision']
        recall = cls_report['1']['recall']

        metrics = {}
        metrics['val_loss'] = loss.item()
        metrics['val_accuracy'] = accuracy
        metrics['val_f1_score'] = f1_score
        metrics['val_precision'] = precision
        metrics['val_recall'] = recall

        print()
        print(metrics)

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, true, pred = self._shared_eval_step(batch, batch_idx)
        return loss, true, pred

    def test_epoch_end(self, test_step_outputs):
        loss = torch.Tensor().to(device='cuda')
        true = []
        pred = []

        for output in test_step_outputs:
            loss = torch.cat((loss, output[0].view(1)), dim=0)
            true += output[1].numpy().tolist()
            pred += output[2].numpy().tolist()

        loss = torch.mean(loss)

        cls_report = classification_report(true, pred, labels=[0, 1], output_dict=True, zero_division=0)

        accuracy = cls_report['accuracy']
        f1_score = cls_report['1']['f1-score']
        precision = cls_report['1']['precision']
        recall = cls_report['1']['recall']

        metrics = {}
        metrics['test_loss'] = loss.item()
        metrics['test_accuracy'] = accuracy
        metrics['test_f1_score'] = f1_score
        metrics['test_precision'] = precision
        metrics['test_recall'] = recall

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

        return loss

    def _shared_eval_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, targets = batch
        outputs = torch.squeeze(self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids), dim=1)

        loss = self.criterion(outputs, targets)

        true = targets.to(torch.device("cpu"))
        pred = (torch.sigmoid(outputs) >= 0.5).int().to(torch.device("cpu"))

        return loss, true, pred

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids = batch
        outputs = torch.squeeze(self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids), dim=1)

        pred = (torch.sigmoid(outputs) >= 0.5).int().to(torch.device("cpu"))

        return pred[0]

#==================================================================================================

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value, attention_mask=None):
        attention_output, _ = self.attention(query, key, value, attn_mask=attention_mask)
        attention_output = self.dropout(attention_output)
        return self.norm(attention_output + query)
    
class FinetuneV3WithCrossAttention(pl.LightningModule):
    def __init__(self, model, learning_rate=2e-5) -> None:
        super(FinetuneV3WithCrossAttention, self).__init__()
        self.model = model
        self.lr = learning_rate

        # Adding cross-attention layers
        self.cross_attention = CrossAttentionLayer(embed_dim=768, num_heads=8)
        
        self.linear1 = nn.Linear(768, 32)
        self.linear2 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Apply cross-attention on the hidden states (last_hidden_state)
        cross_attn_output = self.cross_attention(model_output.last_hidden_state, model_output.last_hidden_state, model_output.last_hidden_state)

        pooled_output = cross_attn_output[:, 0]  # Pooling only the [CLS] token output
        linear_output = self.linear1(pooled_output)
        relu_output = self.relu(linear_output)
        output = self.linear2(relu_output)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    # def training_step(self, batch, batch_idx):
    #     input_ids, attention_mask, token_type_ids, targets = batch
    #     outputs = torch.squeeze(self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids), dim=1)

    #     loss = self.criterion(outputs, targets)

    #     metrics = {}
    #     metrics['train_loss'] = loss.item()

    #     self.log_dict(metrics, prog_bar=False, on_epoch=True)

    #     return loss
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, targets = batch
        outputs = torch.squeeze(self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids), dim=1)

        # Pastikan targets memiliki tipe long
        targets = targets.long()

        loss = self.criterion(outputs, targets)

        metrics = {}
        metrics['train_loss'] = loss.item()

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

        return loss


    def validation_step(self, batch, batch_idx):
        loss, true, pred = self._shared_eval_step(batch, batch_idx)
        return loss, true, pred

    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.Tensor().to(device='cuda')
        true = []
        pred = []

        for output in validation_step_outputs:
            loss = torch.cat((loss, output[0].view(1)), dim=0)
            true += output[1].numpy().tolist()
            pred += output[2].numpy().tolist()

        loss = torch.mean(loss)

        cls_report = classification_report(true, pred, labels=[0, 1], output_dict=True, zero_division=0)

        accuracy = cls_report['accuracy']
        f1_score = cls_report['1']['f1-score']
        precision = cls_report['1']['precision']
        recall = cls_report['1']['recall']

        metrics = {}
        metrics['val_loss'] = loss.item()
        metrics['val_accuracy'] = accuracy
        metrics['val_f1_score'] = f1_score
        metrics['val_precision'] = precision
        metrics['val_recall'] = recall

        print()
        print(metrics)

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, true, pred = self._shared_eval_step(batch, batch_idx)
        return loss, true, pred

    def test_epoch_end(self, test_step_outputs):
        loss = torch.Tensor().to(device='cuda')
        true = []
        pred = []

        for output in test_step_outputs:
            loss = torch.cat((loss, output[0].view(1)), dim=0)
            true += output[1].numpy().tolist()
            pred += output[2].numpy().tolist()

        loss = torch.mean(loss)

        cls_report = classification_report(true, pred, labels=[0, 1], output_dict=True, zero_division=0)

        accuracy = cls_report['accuracy']
        f1_score = cls_report['1']['f1-score']
        precision = cls_report['1']['precision']
        recall = cls_report['1']['recall']

        metrics = {}
        metrics['test_loss'] = loss.item()
        metrics['test_accuracy'] = accuracy
        metrics['test_f1_score'] = f1_score
        metrics['test_precision'] = precision
        metrics['test_recall'] = recall

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

        return loss

    def _shared_eval_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, targets = batch
        outputs = torch.squeeze(self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids), dim=1)

        loss = self.criterion(outputs, targets)

        true = targets.to(torch.device("cpu"))
        pred = (torch.sigmoid(outputs) >= 0.5).int().to(torch.device("cpu"))

        return loss, true, pred

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids = batch
        outputs = torch.squeeze(self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids), dim=1)

        pred = (torch.sigmoid(outputs) >= 0.5).int().to(torch.device("cpu"))

        return pred[0]