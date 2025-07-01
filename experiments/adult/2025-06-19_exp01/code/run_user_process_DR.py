import numpy as np
import pandas as pd
import os
from noise_mechanism import get_oue_frequency,attribute_class_set,make_cross_tab_from_domain,get_pair_explanatory_and_response
import json

# ランダムにK個の属性を選択、それ以外はnull
def randomly_nullify(df, k, random_state=42):
    np.random.seed(random_state)
    total_attribute = df.shape[1]
    # 各行に対して、ランダムに NaN を設定
    def nullify_row(row):
        cols_to_null = np.random.choice(df.columns, total_attribute - k, replace=False)
        row[cols_to_null] = np.nan
        return row
    return df.apply(nullify_row, axis=1)



def user_process_DR(X, domain_dict, y, epsilon, d,out_dir):
    """
    X: ビン化済みのユーザデータ
    domain_dict: 各属性に対する可能な値のリスト（属性×クラスのドメイン）
    y: ラベル列
    epsilon: 全体のプライバシーバジェット
    d: 次元数（選択する属性数）

    戻り値: 常に 0
    """
    X_selected = randomly_nullify(X, d)
    df_pair = get_pair_explanatory_and_response(X_selected, y)



    for col in df_pair.columns:
        domain = domain_dict.get(col, [])
        pairs = attribute_class_set(domain, y)
        
        freq, _ = get_oue_frequency(
            df_pair[col], pairs, epsilon / d
        )
        cross_tab = make_cross_tab_from_domain(freq,domain,col,y)
        # 出力ファイルパス
        out_path = os.path.join(out_dir, f"{col}_OUE_estimation.csv")
        cross_tab.to_csv(out_path)
        
    return 0





if __name__ =="__main__":
    df=pd.read_csv("../data/processed/discrete_30_adult.csv")
    d=7
    epsilon=10
    X=df.drop(["income"],axis=1)
    y=df.loc[:,"income"]  
    with open('../data/processed/domain_dict.json', 'r', encoding='utf-8') as f:
        domain_dict = json.load(f)
        # 保存ディレクトリ準備
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "external", "dist", f"epsilon{epsilon:.2f}"))
    os.makedirs(out_dir, exist_ok=True)
    user_process_DR(X,domain_dict,y,epsilon,d,out_dir)