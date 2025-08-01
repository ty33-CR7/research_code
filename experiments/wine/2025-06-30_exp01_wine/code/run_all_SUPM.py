import pandas as pd
import json
import os
from preprocessing_dataset_number_discrete import preprocess_wine_quality_data
from run_user_process_DR import user_process_DR
from ADR_build_features import Attribute_Domain_Reconstruction
from dict_json import dict_domain_to_json
from Cramer_feature_selection import calculate_cramer_chi_statistic
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from run_user_process_train import user_process_train
from sklearn.ensemble import RandomForestClassifier
import joblib
from SUPM_build_features import SUPM_Domain

#初期設定数値
L_init=30
dimension_number=7
L=5

#前処理フェーズ
preprocess_wine_quality_data(bins=L_init)  # rawから処理


#繰り返し回数
N=5
split_portion=0.9

for epsilon in [5,7,10,12,15,17,20]: 
    df=pd.read_csv("../data/processed/discrete_30_wine_quality.csv")
    X=df.drop(["color"],axis=1)
    y=df.loc[:,"color"]  
    skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
    for fold_index,(train_index,test_index) in enumerate(skf.split(X,y),start=1):
        x_train, y_train, x_test, y_test = X.loc[train_index],y.loc[train_index],X.loc[test_index],y.loc[test_index]
        x_train, y_train, x_test, y_test = x_train.reset_index(drop=True),y_train.reset_index(drop=True),x_test.reset_index(drop=True),y_test.reset_index(drop=True)
        x_train, x_feature, y_train, y_feature = train_test_split( x_train, y_train, train_size=split_portion,random_state=42)
        x_train, y_train, x_feature, y_feature = x_train.reset_index(drop=True),y_train.reset_index(drop=True),x_feature.reset_index(drop=True),y_feature.reset_index(drop=True)
        for noise_num in range(N):

            #ユーザ側の処理        
            with open('../data/processed/domain_dict.json', 'r', encoding='utf-8') as f:
                original_domain_dict = json.load(f)
                    # 保存ディレクトリ準備
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            out_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "external", "dist", f"epsilon{epsilon:.2f}",f"Fold{fold_index}",f"{noise_num}"))
            os.makedirs(out_dir, exist_ok=True)
            user_process_DR(x_feature,original_domain_dict,y_feature,epsilon,dimension_number,out_dir)


            #収集者側の処理
            # 保存パス（ドメイン情報）
            file_path = os.path.join(
                BASE_DIR, "..", "data", "external", "domain",
                 f"Fold{fold_index}", f"{noise_num}",
                f"SUPM_domain_L_{L}.json"
            )
            # 各属性に対して処理を実行
            SUPM_dict_domain={}
            #各属性のカイ2乗係数の計算結果を格納
            dict_chi_square={}
            column_type = "numerical"
            for column in X.columns:

                cross_table_path = os.path.join(
                    BASE_DIR, "..", "data", "external", "dist",
                    f"epsilon{epsilon:.2f}", f"Fold{fold_index}", f"{noise_num}",
                    f"{column}_OUE_estimation.csv"
                )

                cross_table = pd.read_csv(cross_table_path,index_col=0)
                dict_chi_square[column]=calculate_cramer_chi_statistic(cross_table)
                SUPM_dict_domain[column]=SUPM_Domain(original_domain_dict[column],column_type,L)

            domain_out_dir = os.path.dirname(file_path)
            os.makedirs(domain_out_dir, exist_ok=True)
            dict_domain_to_json(SUPM_dict_domain,file_path)


            #特徴量の評価フェーズ（次元削減）
            series_chi_square=pd.Series(dict_chi_square)
            dscending_sort_column_by_chi=series_chi_square.sort_values(ascending=False).index
            selected_columns=dscending_sort_column_by_chi[0:dimension_number]         


            #学習フェーズ
            # ユーザの側の処理
            # ：選択属性に基づいて特徴量を選択し、RRノイズを付与して送信 
            x_train_ADR_RR,y_train_RR=user_process_train(x_train,y_train,epsilon,SUPM_dict_domain,selected_columns)

            #収集者側の処理
            #ユーザから取得したデータを用いて，モデルを作成
            classfier = RandomForestClassifier().fit(x_train_ADR_RR,y_train_RR)
            model_out_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "models", f"epsilon{epsilon:.2f}",f"Fold{fold_index}",f"{noise_num}"))
            os.makedirs(model_out_dir, exist_ok=True)
            # 選択された属性の保存
            selected_columns_path = os.path.join(model_out_dir, "selected_columns.csv")
            pd.Series(selected_columns, name="selected_column").to_csv(selected_columns_path, index=False)
            model_name = "random_forest_model"
            model_path = os.path.join(model_out_dir, f"{model_name}.joblib")
            joblib.dump(classfier, model_path)