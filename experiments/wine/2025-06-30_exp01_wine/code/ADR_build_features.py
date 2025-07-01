import numpy as np
import pandas as pd
import argparse
from typing import Literal
from merge_algorithm import (
    cross_chimerge,
    low_freq_merge_numbers,
    low_freq_merge_categories,
    category_attribute_class_order
)
from dict_json import dict_domain_to_json
import os
def Attribute_Domain_Reconstruction(
        cross_table: pd.DataFrame,
        low_threshold: float,
        max_interval_len: int,
        column_type: Literal['numerical', 'categorical']
) -> list[set]:
    """
    クロス集計表に基づくドメイン再構築。
    """
    if column_type not in ('numerical', 'categorical'):
        raise ValueError(f"column_type は 'numerical' または 'categorical' で指定してください。: {column_type}")

    if column_type == 'numerical':
        low_freq_domain = low_freq_merge_numbers(cross_table, low_threshold)[0]
        chi_domain = cross_chimerge(cross_table, max_interval_len, low_freq_domain)
        return chi_domain
    else:
        low_freq_domain = low_freq_merge_categories(cross_table, low_threshold)[0]
        low_freq_order_domain = category_attribute_class_order(cross_table, low_freq_domain)
        chi_domain = cross_chimerge(cross_table, max_interval_len, low_freq_order_domain)
        return chi_domain

if __name__ == "__main__":
    # 引数処理
    # parser = argparse.ArgumentParser(description="属性ごとのドメイン再構築処理")
    # parser.add_argument("--low_threshold", type=float, required=True, help="低頻度とみなす割合 (例: 0.01)")
    # parser.add_argument("--max_interval_len", type=int, required=True, help="最大区間数 (例: 5)")
    # args = parser.parse_args()
    low_threshold=500
    max_interval_len=5
    epsilon=10

    # 定数
    adult_column = [
        "age", "workclass",  "education", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
        "hours_per_week", "native_country"
    ]
    NUM_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    CAT_COLUMNS = [
        "workclass", "education", "occupation", "race", "sex",
        "native_country", "relationship", "marital_status"
    ]
    file_path=f"../data/external/domain/epsilon{epsilon:.2f}/ADR_domain_T_{low_threshold}_L_{max_interval_len}.csv"
    # 各属性に対して処理を実行
    dict_domain={}
    for column in adult_column:
        if column in NUM_COLUMNS:
            column_type = "numerical"
        elif column in CAT_COLUMNS:
            column_type = "categorical"
        else:
            raise ValueError(f"column は 'numerical' または 'categorical' の型で指定してください: {column}")

        cross_table = pd.read_csv(f"../data/external/dist/epsilon{epsilon:.2f}/{column}_OUE_estimation.csv",index_col=0)
        result = Attribute_Domain_Reconstruction(
            cross_table,
            low_threshold,
            max_interval_len,
            column_type
        )
        dict_domain[column]=result
       # print(f"{column} → {result}")
    os.makedirs(f"../data/external/domain/epsilon{epsilon:.2f}", exist_ok=True)
    dict_domain_to_json(dict_domain,file_path)
    

