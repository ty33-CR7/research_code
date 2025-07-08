import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import json
# 事前処理でドメインを定める．

# 定数
NUM_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CAT_COLUMNS = [
    "workclass", "education", "occupation", "race",
    "sex", "native_country", "relationship", "marital_status"
]

def equal_width_binning(series: pd.Series, bins: int = 5) -> pd.Series:
    """等幅ビニングを行い、各ビンの中央値に変換する。"""
    cut_bins = pd.cut(series, bins)
    return cut_bins.apply(lambda x: (x.left + x.right) / 2)

def preprocess_adult_data(labeled: bool = False, bins: int = 5):
    """
    adultデータを等幅ビニングで前処理して保存する。
    
    Parameters
    ----------
    labeled : bool
        Trueの場合、ラベル済みデータ (label_adult.csv) を読み込む。
        Falseの場合、rawデータ (adult.data) を読み込む。
    bins : int
        数値列を等幅で分割するビンの数。
    """
    if labeled:
        raw_path = "../data/processed/label_adult.csv"
        output_path = f"../data/processed/discrete_{bins}_label_adult.csv"
        df = pd.read_csv(raw_path, index_col=0)
        df = df.drop(columns=["fnlwgt"], errors="ignore")
    else:
        raw_path = "../data/raw/adult.data"
        output_path = f"../data/processed/discrete_{bins}_adult.csv"    
        df=pd.read_csv(raw_path,header=None)
        df.columns=[
                    "age",
                    "workclass",
                    "fnlwgt",
                    "education",
                    "education_num",
                    "marital_status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "capital_gain",
                    "capital_loss",
                    "hours_per_week",
                    "native_country",
                    "income",
                ]
        df = df.drop(columns=["fnlwgt"], errors="ignore")
    df_discrete = df.copy()
    df_discrete[NUM_COLUMNS] = df_discrete[NUM_COLUMNS].apply(
        lambda col: equal_width_binning(col, bins)
    )
    domain_dict={}
    for column in NUM_COLUMNS:
        domain_dict[column]=np.array([(x.left+x.right)/2 for x in pd.cut(df[column],bins).cat.categories]).astype(str).tolist()
    for column in CAT_COLUMNS:
        domain_dict[column]=df_discrete[column].unique().astype(str).tolist()
    with open('../data/processed/domain_dict.json', 'w', encoding='utf-8') as f:
        json.dump(domain_dict, f, ensure_ascii=False, indent=2)
    df_discrete.to_csv(output_path,index=False)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    preprocess_adult_data(labeled=False,bins=30)  # rawから処理
    #preprocess_adult_data(labeled=True)   # label済みから処理


