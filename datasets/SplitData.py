import json

# Membaca dataset dari file JSON
with open('../rawData/github/Githuboutput_5.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Split the dataset
split_data = []

for item in data:
    query = item["query"]
    for result in item["results"]:
        split_data.append({
            "query": query,
            "corpus": result["corpus"]
        })

# Menyimpan hasil transformasi ke file JSON baru
with open('github/github_5.json', 'w', encoding='utf-8') as file:
    json.dump(split_data, file, ensure_ascii=False, indent=4)

print("Transformasi selesai. Hasil disimpan di 'split_dataset.json'")
