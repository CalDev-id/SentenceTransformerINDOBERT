import json
import random
from sklearn.model_selection import train_test_split

# Fungsi untuk membagi dataset
def split_dataset(json_file, validation_split=0.2, seed=42):
    # Set seed untuk hasil yang dapat direproduksi
    random.seed(seed)
    
    # Baca data dari file JSON
    with open(json_file, 'r') as f:
        dataset = json.load(f)
    
    # Bagi dataset menjadi training dan validation
    train_data, valid_data = train_test_split(dataset, test_size=validation_split, random_state=seed)
    
    # Simpan data yang telah dibagi ke file JSON baru
    # with open('train_data.json', 'w') as f:
    #     json.dump(train_data, f)
    
    with open('turnbackhoax/valid_data.json', 'w') as f:
        json.dump(valid_data, f)

    print(f"Train data: {len(train_data)} samples")
    print(f"Validation data: {len(valid_data)} samples")

# Panggil fungsi dengan file JSON Anda
split_dataset('turnbackhoax/github_mendaley_train.json', validation_split=0.2)
