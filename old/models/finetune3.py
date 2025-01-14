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
