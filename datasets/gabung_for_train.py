import json

# Memuat file JSON pertama
with open('github/github_sentenced_combined.json', 'r', encoding='utf-8') as file1:
    data1 = json.load(file1)

# Memuat file JSON kedua
with open('mendaley/mendaley_sentenced_combined.json', 'r', encoding='utf-8') as file2:
    data2 = json.load(file2)

# Menggabungkan kedua list
merged_data = data1 + data2

# Menyimpan hasil gabungan ke dalam file JSON baru
with open('turnbackhoax/github_mendaley_train.json', 'w', encoding='utf-8') as outfile:
    json.dump(merged_data, outfile, ensure_ascii=False, indent=4)

print("Penggabungan berhasil dan disimpan ke 'merged_file.json'")
