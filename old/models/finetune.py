import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from sklearn.metrics import classification_report

class Finetune(pl.LightningModule):

    def __init__(self, model, learning_rate=2e-5) -> None:
        # Inisialisasi kelas Finetune
        super(Finetune, self).__init__()
        self.model = model # Menggunakan model yang telah diinisialisasi
        self.lr = learning_rate # Menyimpan learning rate

    def forward(self, input_ids, attention_mask, labels=None):
        # Metode forward untuk melakukan propagasi maju (forward pass)
        if labels is not None:
            # Jika terdapat label, gunakan loss dan logits dari model
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return model_output.loss, model_output.logits
        else:
            # Jika tidak ada label, gunakan hanya logits dari model
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return model_output.logits

    def configure_optimizers(self):
        # Konfigurasi optimizers, dalam hal ini menggunakan Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    # Metode untuk langkah pelatihan
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch
        loss, logits = self(input_ids=input_ids, attention_mask=attention_mask, labels=targets)

        metrics = {}
        metrics['train_loss'] = loss.item() # Menyimpan loss pelatihan

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

        return loss

    # Metode untuk langkah validasi
    def validation_step(self, batch, batch_idx):
        loss, true, pred = self._shared_eval_step(batch, batch_idx)
        return loss, true, pred

    # Metode untuk menyelesaikan epoch validasi
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

    # Metode untuk langkah pengujian
    def test_step(self, batch, batch_idx):
        loss, true, pred = self._shared_eval_step(batch, batch_idx)
        return loss, true, pred

    # Metode untuk menyelesaikan epoch pengujian
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

    # Metode untuk langkah evaluasi bersama
    def _shared_eval_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch
        loss, logits = self(input_ids=input_ids, attention_mask=attention_mask, labels=targets)

        true = torch.argmax(targets, dim=1).to(torch.device("cpu"))
        pred = torch.argmax(logits, dim=1).to(torch.device("cpu"))

        return loss, true, pred

    # Metode untuk langkah prediksi
    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        logits = self(input_ids=input_ids, attention_mask=attention_mask)

        pred = torch.argmax(logits, dim=1).to(torch.device("cpu"))

        return pred[0]

#step
    # pertama tama init model dan learning rate
    # kemudian forward pass untuk melakukan propagasi maju (forward pass)
    # kemudian configure optimizers, dalam hal ini menggunakan Adam optimizer
    # kemudian training step untuk langkah pelatihan
    # kemudian validation step untuk langkah validasi
    # kemudian validation epoch end untuk menyelesaikan epoch validasi
    # kemudian test step untuk langkah pengujian
    # kemudian test epoch end untuk menyelesaikan epoch pengujian
    # kemudian shared eval step untuk langkah evaluasi bersama
    # kemudian predict step untuk langkah prediksi



#sebuah agent yang bertugas untuk membuat constraint dan memeriksa apakah input user ke llm sudah aman