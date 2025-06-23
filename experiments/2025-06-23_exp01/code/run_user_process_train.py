import numpy as np
import pandas as pd 
import os 
from noise_mechanism import RRNoiseTransformer
from sklearn.preprocessing import LabelEncoder



def selected_attibute(df,selected_columns):
    """
    指定された特徴のみを使用する
    """
    return df.loc[:,selected_columns]


def Attribute_domain_reconstruction(df,domain_dict):
    """データをdomain_dictに格納されたドメインに合わせて、変換する"""

    df_copy=df.copy()
    for column in df_copy.columns:
        value_to_bin = {}
        for i, bin_values in enumerate(domain_dict[column]):
            for val in bin_values:
                value_to_bin[round(val, 5)] = i  # 小数誤差を防ぐため丸める
        
        df_copy[column] = df[column].round(5).map(value_to_bin).fillna(-1).astype(int)
    
    return df_copy




def user_process_train(x_train, y_train, epsilon, domain_dict, selected_columns):
    # 特徴量選択
    x_selected = selected_attibute(x_train, selected_columns)

    # ドメインに従った整数化
    x_train_ADR = Attribute_domain_reconstruction(x_selected, domain_dict)

    if epsilon != "no_noise":
        # RRノイズ適用（特徴量）
        X_RR_noise_Trans = RRNoiseTransformer(epsilon / (len(selected_columns) + 1), domain_dict)
        x_train_ADR_RR = X_RR_noise_Trans.fit_transform(x_train_ADR)

        # ラベルのエンコード + RR
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)

        y_RR_noise_Trans = RRNoiseTransformer(epsilon / (len(selected_columns) + 1))
        y_train_RR = y_RR_noise_Trans.fit_transform(y_train_encoded)

        return x_train_ADR_RR, y_train_RR

    # ノイズなし処理
    return x_train_ADR, y_train

