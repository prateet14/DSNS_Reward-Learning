import os
import re
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

input_folder = r"C:\Users\Dell\Desktop\Data Science Applications in Neuroscience\Dataset\Features Table"
output_folder = r"C:\Users\Dell\Desktop\Data Science Applications in Neuroscience\Dataset\PCA_2D"
os.makedirs(output_folder, exist_ok=True)

features = [
    'Control', 'Bin1', 'Bin2', 'Bin3', 'Bin4', 'Bin5',
    'Bin6', 'Bin7', 'Bin8', 'Bin9', 'Bin10', 'Prediction', 'Reward'
]

pca_day_dfs = {}
pca_list = []

for file in sorted(os.listdir(input_folder)):
    if not file.endswith(".csv"):
        continue

    base = os.path.splitext(file)[0]
    m = re.search(r'(\d)', base)
    day_label = f"Day{m.group(1)}" if m else base

    df = pd.read_csv(os.path.join(input_folder, file))

    if "Animal" not in df.columns:
        raise ValueError(f"'Animal' column not found in {file}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['Animal'] = df['Animal'].values
    pca_df['Day'] = day_label

    out_name = f"{day_label}_PCA.csv"
    out_path = os.path.join(output_folder, out_name)
    pca_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    pca_day_dfs[day_label] = pca_df
    pca_list.append(pca_df)