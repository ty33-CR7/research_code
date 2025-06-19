import pandas as pd
import json
import os
from preprocessing_dataset_number_discrete import preprocess_adult_data
from run_user_process_DR import user_process_DR
from ADR_build_features import Attribute_Domain_Reconstruction
from dict_json import dict_domain_to_json

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

#初期設定数値
L=30
d=7
epsilon=10
low_threshold=500
max_interval_len=5


#前処理フェーズ
preprocess_adult_data(labeled=False,bins=L)  # rawから処理


#繰り返し回数
N=10

for i in N:

    #ユーザ側の処理
    df=pd.read_csv("../data/processed/discrete_30_adult.csv")
    X=df.drop(["income"],axis=1)
    y=df.loc[:,"income"]  
    with open('../data/processed/domain_dict.json', 'r', encoding='utf-8') as f:
        domain_dict = json.load(f)
            # 保存ディレクトリ準備
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "external", "dist", f"epsilon{epsilon:.2f}",f"{N}"))
    os.makedirs(out_dir, exist_ok=True)
    user_process_DR(X,domain_dict,y,epsilon,d,out_dir)


    #収集者側の処理
    file_path=f"../data/external/domain/epsilon{epsilon:.2f}/{N}/ADR_domain_T_{low_threshold}_L_{max_interval_len}.csv"
    # 各属性に対して処理を実行
    dict_domain={}
    for column in adult_column:
        if column in NUM_COLUMNS:
            column_type = "numerical"
        elif column in CAT_COLUMNS:
            column_type = "categorical"
        else:
            raise ValueError(f"column は 'numerical' または 'categorical' の型で指定してください: {column}")

        cross_table = pd.read_csv(f"../data/external/dist/epsilon{epsilon:.2f}/{N}/{column}_OUE_estimation.csv",index_col=0)
        result = Attribute_Domain_Reconstruction(
            cross_table,
            low_threshold,
            max_interval_len,
            column_type
        )
        dict_domain[column]=result
    os.makedirs(f"../data/external/domain/epsilon{epsilon:.2f}/{N}", exist_ok=True)
    dict_domain_to_json(dict_domain,file_path)