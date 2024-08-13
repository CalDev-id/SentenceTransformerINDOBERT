import json

# Membaca dataset dari file JSON
with open('turnbackhoax/turnbackhoax_output_v1.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Split the dataset
split_data = []

for item in data:
    query = item["query"]
    label = item["label"]
    for result in item["results"]:
        split_data.append({
            "query": query,
            "label": label,
            "corpus": result["corpus"]
        })

# Menyimpan hasil transformasi ke file JSON baru
with open('turnbackhoax/turnbackhoax_output.json', 'w', encoding='utf-8') as file:
    json.dump(split_data, file, ensure_ascii=False, indent=4)

print("Transformasi selesai. Hasil disimpan di 'split_dataset.json'")
