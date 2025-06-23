import pandas as pd
import json
import os
from preprocessing_dataset_number_discrete import preprocess_adult_data
from run_user_process_DR import user_process_DR
from ADR_build_features import Attribute_Domain_Reconstruction
from dict_json import dict_domain_to_json
from Cramer_feature_selection import calculate_cramer_chi_statistic
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from run_user_process_train import user_process_train
from sklearn.ensemble import RandomForestClassifier
import joblib
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
dimension_number=7
epsilon=10
low_threshold=500
max_interval_len=5


#前処理フェーズ
preprocess_adult_data(labeled=False,bins=L)  # rawから処理


#繰り返し回数
N=5
split_portion=0.9

for epsilon in [5,7,10,12,15,17,20,30,40,50]: 
    df=pd.read_csv("../data/processed/discrete_30_adult.csv")
    X=df.drop(["income"],axis=1)
    y=df.loc[:,"income"]  
    skf = StratifiedKFold(n_splits=10,random_state=42)
    for fold_index,train_index,test_index in enumerate(skf.split(X,y),start=1):
        x_train, y_train, x_test, y_test = X.loc[train_index],y.loc[train_index],X.loc[test_index],y.loc[test_index]
        x_train, y_train, x_test, y_test = x_train.reset_index(drop=True),y_train.reset_index(drop=True),x_test.reset_index(drop=True),y_test.reset_index(drop=True)
        x_train, x_feature, y_train, y_feature = train_test_split( x_train, y_train, train_size=split_portion,random_state=42)
        x_train, y_train, x_feature, y_feature = x_train.reset_index(drop=True),y_train.reset_index(drop=True),x_feature.reset_index(drop=True),y_feature.reset_index(drop=True)
        for noise_num in range(N):

            #ユーザ側の処理        
            with open('../data/processed/domain_dict.json', 'r', encoding='utf-8') as f:
                domain_dict = json.load(f)
                    # 保存ディレクトリ準備
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            out_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "external", "dist", f"epsilon{epsilon:.2f}",f"Fold{fold_index}",f"{noise_num}"))
            os.makedirs(out_dir, exist_ok=True)
            user_process_DR(x_feature,domain_dict,y_feature,epsilon,dimension_number,out_dir)


            #収集者側の処理
            file_path=f"../data/external/domain/epsilon{epsilon:.2f}/Fold{fold_index}/{noise_num}/ADR_domain_T_{low_threshold}_L_{max_interval_len}.csv"
            # 各属性に対して処理を実行
            dict_domain={}
            #各属性のカイ2乗係数の計算結果を格納
            dict_chi_square={}
            for column in adult_column:
                if column in NUM_COLUMNS:
                    column_type = "numerical"
                elif column in CAT_COLUMNS:
                    column_type = "categorical"
                else:
                    raise ValueError(f"column は 'numerical' または 'categorical' の型で指定してください: {column}")

                cross_table = pd.read_csv(f"../data/external/dist/epsilon{epsilon:.2f}/Fold{fold_index}/{noise_num}/{column}_OUE_estimation.csv",index_col=0)
                result = Attribute_Domain_Reconstruction(
                    cross_table,
                    low_threshold,
                    max_interval_len,
                    column_type
                )
                dict_chi_square[column]=calculate_cramer_chi_statistic(cross_table,result)
                dict_domain[column]=result

            os.makedirs(f"../data/external/domain/epsilon{epsilon:.2f}/Fold{fold_index}/{noise_num}", exist_ok=True)
            dict_domain_to_json(dict_domain,file_path)


            #特徴量の評価フェーズ（次元削減）
            series_chi_square=pd.Series(dict_chi_square)
            dscending_sort_column_by_chi=series_chi_square.sort_values(ascending=False).index
            selected_columns=dscending_sort_column_by_chi[0:dimension_number]          


            #学習フェーズ
            # ユーザの側の処理
            # ：選択属性に基づいて特徴量を選択し、RRノイズを付与して送信  
            x_train_ADR_RR,y_train_RR=user_process_train(x_train,y_train,epsilon,domain_dict,selected_columns)

            #収集者側の処理
            #ユーザから取得したデータを用いて，モデルを作成
            classfier = RandomForestClassifier().fit(x_train_ADR_RR,y_train_RR)
            model_out_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "models", f"epsilon{epsilon:.2f}",f"Fold{fold_index}",f"{noise_num}"))
            os.makedirs(model_out_dir, exist_ok=True)
            model_name = "random_forest_model"
            model_path = os.path.join(model_out_dir, f"{model_name}.joblib")
            joblib.dump(classfier, model_path)

