import pandas as pd
from sklearn.cluster import KMeans

def cluster_day_pair(day_a, day_b, pca_days, k=3):
    """
    Clusters animals based on their PCA trajectory between two days.
    
    Parameters:
        day_a (int): First day number (key in pca_days)
        day_b (int): Second day number (key in pca_days)
        pca_days (dict): Dictionary mapping day number to PCA DataFrame
        k (int): Number of clusters for KMeans
        
    Returns:
        pd.DataFrame: Merged DataFrame with delta_PC1, delta_PC2, and cluster labels
    """
    df_a = pca_days[day_a]
    df_b = pca_days[day_b]

    merged = pd.merge(
        df_a, df_b,
        on='Animal',
        suffixes=(f'_day{day_a}', f'_day{day_b}')
    )

    merged['delta_PC1'] = merged[f'PC1_day{day_b}'] - merged[f'PC1_day{day_a}']
    merged['delta_PC2'] = merged[f'PC2_day{day_b}'] - merged[f'PC2_day{day_a}']

    X_delta = merged[['delta_PC1', 'delta_PC2']]
    kmeans = KMeans(n_clusters=k, random_state=42)
    merged['Cluster'] = kmeans.fit_predict(X_delta)

    return merged