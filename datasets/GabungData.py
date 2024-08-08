import json

# List of file names to be combined
file_names = ['github/github_1.json', 'github/github_2.json', 'github/github_3.json', 'github/github_4.json','github/github_5.json']

combined_data = []

# Membaca dan menggabungkan data dari setiap file
for file_name in file_names:
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
        combined_data.extend(data)

# Menyimpan data gabungan ke file JSON baru
with open('github/Github_combined.json', 'w', encoding='utf-8') as file:
    json.dump(combined_data, file, ensure_ascii=False, indent=4)

print("Penggabungan selesai. Hasil disimpan di 'combined_dataset.json'")
