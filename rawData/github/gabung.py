import json

# List of file names to be combined
file_names = ['Githuboutput_1.json', 'Githuboutput_2.json', 'Githuboutput_3.json', 'Githuboutput_4.json','Githuboutput_5.json']

combined_data = []

# Membaca dan menggabungkan data dari setiap file
for file_name in file_names:
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
        combined_data.extend(data)

# Menyimpan data gabungan ke file JSON baru
with open('Github_combined.json', 'w', encoding='utf-8') as file:
    json.dump(combined_data, file, ensure_ascii=False, indent=4)

print("Penggabungan selesai. Hasil disimpan di 'combined_dataset.json'")
