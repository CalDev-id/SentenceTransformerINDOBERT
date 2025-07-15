# import torch
# from transformers import AutoTokenizer, AutoModel
# from models.finetune import Finetune  # Import class Finetune kamu
# from utils.preprocessorv2 import DataModule  # Kalau kamu mau proses data

# # Load tokenizer dan model backbone
# tokenizer = AutoTokenizer.from_pretrained('indolem/indobert-base-uncased', return_token_type_ids=True, use_fast=False)
# pretrained_model = AutoModel.from_pretrained('indolem/indobert-base-uncased')

# # Load model checkpoint
# checkpoint_path = '/kaggle/input/model-fake-news/epoch6-step3759.ckpt'
# model = Finetune.load_from_checkpoint(checkpoint_path, model=pretrained_model)
# model.eval()

# text = "Pemerintah membagikan uang tunai kepada seluruh warga negara Indonesia."

# # Tokenisasi input
# encoding = tokenizer(
#     text,
#     truncation=True,
#     padding='max_length',
#     max_length=128,
#     return_tensors='pt'
# )

# # Inference
# with torch.no_grad():
#     logits = model(
#         input_ids=encoding['input_ids'],
#         attention_mask=encoding['attention_mask'],
#         token_type_ids=encoding.get('token_type_ids', None)
#     )
#     if isinstance(logits, tuple):  # Kalau output adalah (loss, logits)
#         logits = logits[1]
#     prediction = torch.argmax(logits, dim=1).item()

# label_map = {0: "Real", 1: "Fake"}
# print(f"Prediction: {label_map[prediction]}")

import torch
from transformers import AutoTokenizer, AutoModel
from models.finetune import Finetune  # Pastikan import sesuai struktur proyekmu

# 1. Load tokenizer dan pretrained backbone model
pretrained_model_name = 'indolem/indobert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name,
    return_token_type_ids=True,
    use_fast=False
)

pretrained_model = AutoModel.from_pretrained(pretrained_model_name)

# 2. Path ke checkpoint hasil training
checkpoint_path = '/kaggle/input/model-fake-news/epoch6-step3759.ckpt'

# 3. Load LightningModule dari checkpoint, dengan model backbone dikirim sebagai argumen
model = Finetune.load_from_checkpoint(
    checkpoint_path,
    model=pretrained_model  # wajib karena Finetune butuh ini
)

# 4. Set ke mode evaluasi
model.eval()

# 5. Siapkan input test (satu kalimat)
text = "Pemerintah membagikan uang tunai kepada seluruh warga negara Indonesia."

# 6. Tokenisasi
encoding = tokenizer(
    text,
    truncation=True,
    padding='max_length',
    max_length=128,
    return_tensors='pt'
)

# 7. Inference
with torch.no_grad():
    logits = model(
        input_ids=encoding['input_ids'],
        attention_mask=encoding['attention_mask'],
        token_type_ids=encoding.get('token_type_ids')
    )
    if isinstance(logits, tuple):  # jika output adalah (loss, logits)
        logits = logits[1]
    prediction = torch.argmax(logits, dim=1).item()

# 8. Label mapping dan output
label_map = {0: "Real", 1: "Fake"}
print(f"Prediction: {label_map[prediction]}")
