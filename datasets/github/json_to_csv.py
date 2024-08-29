import json
import pandas as pd

# Baca file JSON
with open('valid_data.json', 'r') as file:
    data = json.load(file)

# Konversi JSON ke DataFrame pandas
df = pd.DataFrame(data)

# Simpan DataFrame sebagai file CSV
df.to_csv('github_val.csv', index=False)
