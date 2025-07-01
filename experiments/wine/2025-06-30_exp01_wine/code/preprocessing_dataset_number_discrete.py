import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import json
import os
# 事前処理でドメインを定める．


def equal_width_binning(series: pd.Series, bins: int = 5) -> pd.Series:
    """等幅ビニングを行い、各ビンの中央値に変換する。"""
    cut_bins = pd.cut(series, bins)
    return cut_bins.apply(lambda x: (x.left + x.right) / 2)

def preprocess_wine_quality_data(bins: int = 5):
    """
    wine_qualityデータを等幅ビニングで前処理して保存する。
    
    Parameters
    bins : int
        数値列を等幅で分割するビンの数。
    """
    raw_path = "../data/raw/wine_quality.csv"
    output_path = "../data/processed"
    os.makedirs(output_path,exist_ok=True)
    df = pd.read_csv(raw_path, index_col=0)
    NUM_COLUMNS=df.select_dtypes('number').columns
    df_discrete = df.copy()
    df_discrete[NUM_COLUMNS] = df_discrete[NUM_COLUMNS].apply(
        lambda col: equal_width_binning(col, bins)
    )
    domain_dict={}
    for column in NUM_COLUMNS:
        domain_dict[column]=np.array([(x.left+x.right)/2 for x in pd.cut(df[column],bins).cat.categories]).tolist()
    os.makedirs('../data/processed',exist_ok=True)
    with open('../data/processed/domain_dict.json', 'w', encoding='utf-8') as f:
        json.dump(domain_dict, f, ensure_ascii=False, indent=2)
    df_discrete.to_csv(f"{output_path}/discrete_{bins}_wine_quality.csv",index=False)
    print(f"Saved to: {output_path}/discrete_{bins}_wine_quality.csv")

if __name__ == "__main__":
    preprocess_wine_quality_data(bins=30)  # rawから処理


