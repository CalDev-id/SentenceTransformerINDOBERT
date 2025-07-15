# sudah bagus

# import torch
# from transformers import AutoTokenizer, AutoModel
# from models.finetune import FinetuneV2  # Ganti sesuai versi saat training

# tokenizer = AutoTokenizer.from_pretrained('indolem/indobert-base-uncased', return_token_type_ids=True, use_fast=False)
# pretrained_model = AutoModel.from_pretrained('indolem/indobert-base-uncased')

# checkpoint_path = '/kaggle/input/model-fake-news/epoch6-step3759.ckpt'

# model = FinetuneV2.load_from_checkpoint(
#     checkpoint_path,
#     model=pretrained_model
# )
# model.eval()

# text = "polda nusa tenggara barat mengklarifkasi kejadian perkosaan turis perancis gili trawangan lombok"

# encoding = tokenizer(
#     text,
#     truncation=True,
#     padding='max_length',
#     max_length=128,
#     return_tensors='pt'
# )

# with torch.no_grad():
#     logits = model(
#         input_ids=encoding['input_ids'],
#         attention_mask=encoding['attention_mask'],
#         token_type_ids=encoding.get('token_type_ids', None)
#     )
#     if isinstance(logits, tuple):
#         logits = logits[1]
#     prediction = torch.argmax(logits, dim=1).item()

# label_map = {0: "HOAX", 1: "BENAR"}
# print(f"Prediction: {label_map[prediction]}")




# import torch
# from transformers import AutoTokenizer, AutoModel
# from models.finetune import FinetuneV2  # Ganti sesuai versi modelmu
# from sklearn.metrics import accuracy_score, classification_report

# # Load tokenizer dan pretrained backbone
# tokenizer = AutoTokenizer.from_pretrained('indolem/indobert-base-uncased', return_token_type_ids=True, use_fast=False)
# pretrained_model = AutoModel.from_pretrained('indolem/indobert-base-uncased')

# # Load checkpoint yang sudah di-finetune
# checkpoint_path = '/kaggle/input/model-fake-news/epoch6-step3759.ckpt'
# model = FinetuneV2.load_from_checkpoint(checkpoint_path, model=pretrained_model)
# model.eval()

# # Data uji (texts) dan label ground-truth (labels)
# texts = [
#     "kemenhub klarifikasi kabar pegawai ditahan kebakaran gedung kemenhub",  # BENAR
#     "kemenkeu klarifikasi sri mulyani pakai topi kalimat tauhid",  # BENAR
#     "klarifikasi camat kemalang terkait isu pungli perekrutan perdes tangkil",  # BENAR
#     "penangkapan penyusup membawa bom acara pernikahan putri presiden jokowi",  #HOAX
#     "polisi israel mencekik anak palestina mati sabtu demonstrasi kedutaan amerika pindah yerusalem",  #HOAX
# ]
# labels = [1, 1, 1, 0, 0]  # 0: HOAX, 1: BENAR

# # Prediksi model
# predictions = []
# for text in texts:
#     encoding = tokenizer(
#         text,
#         truncation=True,
#         padding='max_length',
#         max_length=128,
#         return_tensors='pt'
#     )
#     with torch.no_grad():
#         logits = model(
#             input_ids=encoding['input_ids'],
#             attention_mask=encoding['attention_mask'],
#             token_type_ids=encoding.get('token_type_ids', None)
#         )
#         if isinstance(logits, tuple):
#             logits = logits[1]
#         pred = torch.argmax(logits, dim=1).item()
#         predictions.append(pred)

# # Hitung metrik evaluasi
# accuracy = accuracy_score(labels, predictions)
# report = classification_report(labels, predictions, target_names=["HOAX", "BENAR"])

# # Output hasil
# label_map = {0: "HOAX", 1: "BENAR"}
# print("=== Prediction Results ===")
# for i, text in enumerate(texts):
#     print(f"[{label_map[labels[i]]}] {text}")
#     print(f"  --> Predicted: {label_map[predictions[i]]}")
# print("\n=== Evaluation Metrics ===")
# print(f"Akurasi: {accuracy:.2f}")
# print(report)

#test pake json

import torch
import json
from transformers import AutoTokenizer, AutoModel
from models.finetune import FinetuneV2
from sklearn.metrics import accuracy_score, classification_report

# Load tokenizer dan backbone IndoBERT
tokenizer = AutoTokenizer.from_pretrained('indolem/indobert-base-uncased', return_token_type_ids=True, use_fast=False)
pretrained_model = AutoModel.from_pretrained('indolem/indobert-base-uncased')

# Load checkpoint
checkpoint_path = '/kaggle/input/model-fake-news/epoch6-step3759.ckpt'
model = FinetuneV2.load_from_checkpoint(checkpoint_path, model=pretrained_model)
model.eval()

# Load data uji dari mendaley_sentenced_test.json
test_file_path = 'datasets/mendaley/mendaley_sentenced_test.json'
with open(test_file_path, 'r') as f:
    test_data = json.load(f)

# Siapkan list untuk teks dan label
texts = [item['query'] for item in test_data]
labels = [item['label'] for item in test_data]

# Prediksi
predictions = []
for text in texts:
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    with torch.no_grad():
        logits = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            token_type_ids=encoding.get('token_type_ids', None)
        )
        if isinstance(logits, tuple):
            logits = logits[1]
        pred = torch.argmax(logits, dim=1).item()
        predictions.append(pred)

# Evaluasi
accuracy = accuracy_score(labels, predictions)
report = classification_report(labels, predictions, target_names=["HOAX", "BENAR"])

# Tampilkan hasil prediksi dan evaluasi
label_map = {0: "HOAX", 1: "BENAR"}
print("=== Prediction Results (first 10 samples) ===")
for i in range(min(10, len(texts))):
    print(f"[{label_map[labels[i]]}] {texts[i]}")
    print(f"  --> Predicted: {label_map[predictions[i]]}")
print("\n=== Evaluation Metrics ===")
print(f"Akurasi: {accuracy:.2f}")
print(report)
