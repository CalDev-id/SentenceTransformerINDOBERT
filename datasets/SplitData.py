# import json

# # Membaca dataset dari file JSON
# with open('turnbackhoax/turnbackhoax_output_v1.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)

# # Split the dataset
# split_data = []

# for item in data:
#     query = item["query"]
#     label = item["label"]
#     for result in item["results"]:
#         split_data.append({
#             "query": query,
#             "label": label,
#             "corpus": result["corpus"]
#         })

# # Menyimpan hasil transformasi ke file JSON baru
# with open('turnbackhoax/turnbackhoax_output.json', 'w', encoding='utf-8') as file:
#     json.dump(split_data, file, ensure_ascii=False, indent=4)

# print("Transformasi selesai. Hasil disimpan di 'split_dataset.json'")

# import json

# # Membaca dataset dari file JSON
# with open('../rawdata/github/updated_json_file.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)

# # Mapping label ke integer
# label_mapping = {
#     "label1": 0,
#     "label2": 1,
#     # Tambahkan mapping untuk label lain sesuai kebutuhan
# }

# # Split the dataset
# split_data = []

# for item in data:
#     query = item["query"]
#     label = label_mapping.get(item["label"], -1)  # Mengubah label menjadi integer
#     for result in item["results"]:
#         split_data.append({
#             "query": query,
#             "label": label,
#             "corpus": result["corpus"]
#         })

# # Menyimpan hasil transformasi ke file JSON baru
# with open('github/github_output_v1.json', 'w', encoding='utf-8') as file:
#     json.dump(split_data, file, ensure_ascii=False, indent=4)

# print("Transformasi selesai. Hasil disimpan di 'turnbackhoax_output.json'")

#=======================================================================================================
import json

# Membaca dataset dari file JSON
with open('../rawdata/github/updated_json_file.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Split the dataset
split_data = []

for item in data:
    query = item["query"]
    # Mengubah label menjadi integer
    label = int(item["label"])
    for result in item["results"]:
        split_data.append({
            "query": query,
            "label": label,
            "corpus": result["corpus"]
        })

# Menyimpan hasil transformasi ke file JSON baru
with open('github/github_output_v1.json', 'w', encoding='utf-8') as file:
    json.dump(split_data, file, ensure_ascii=False, indent=4)

print("Transformasi selesai. Hasil disimpan di 'turnbackhoax/turnbackhoax_output.json'")
