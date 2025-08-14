import pandas as pd
import numpy as np

def compute_learning_trajectory(df_day_a, df_day_b, day_a, day_b):
    """
    Computes Euclidean distances in PCA space between two days for each animal.

    Parameters:
        df_day_a (pd.DataFrame): PCA DataFrame for day_a (must contain 'Animal', 'PC1', 'PC2')
        df_day_b (pd.DataFrame): PCA DataFrame for day_b (must contain 'Animal', 'PC1', 'PC2')
        day_a (int): Day number for df_day_a
        day_b (int): Day number for df_day_b

    Returns:
        pd.DataFrame: DataFrame with Animal IDs and their Euclidean distances, sorted descending.
    """
    merged = pd.merge(
        df_day_a, df_day_b,
        on='Animal',
        suffixes=(f'_day{day_a}', f'_day{day_b}')
    )

    merged['Euclidean_Distance'] = np.sqrt(
        (merged[f'PC1_day{day_b}'] - merged[f'PC1_day{day_a}'])**2 +
        (merged[f'PC2_day{day_b}'] - merged[f'PC2_day{day_a}'])**2
    )

    merged = merged.sort_values(by='Euclidean_Distance', ascending=False)
    return merged[['Animal', 'Euclidean_Distance']]