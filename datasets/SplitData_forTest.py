import json

# Membaca dataset dari file JSON
with open('turnbackhoax/turnbackhoax_output_v1.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Split the dataset
split_data = []

for item in data:
    query = item["query"]
    label = item["label"]
    if item["results"]:  # Pastikan ada hasil
        first_result = item["results"][0]
        split_data.append({
            "query": query,
            "label": label,
            "corpus": first_result["corpus"]
        })

# Menyimpan hasil transformasi ke file JSON baru
with open('turnbackhoax/turnbackhoax_output_v1.json', 'w', encoding='utf-8') as file:
    json.dump(split_data, file, ensure_ascii=False, indent=4)

print("Transformasi selesai. Hasil disimpan di 'turnbackhoax_output.json'")
