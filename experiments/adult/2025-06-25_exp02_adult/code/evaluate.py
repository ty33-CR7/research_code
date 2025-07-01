import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from run_user_process_train import user_process_train
import json
import os
# 評価（例）
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

# データ読み込み
df = pd.read_csv("../data/processed/discrete_30_adult.csv")
X = df.drop(["income"], axis=1)
y = df["income"]

#繰り返し回数
N=5
split_portion=0.9
target_fold_index=[1,2,3,4,5,6,7,8,9,10]

#初期設定数値
L=30
dimension_number=6
epsilon=15
low_threshold=100
max_interval_len=5
# Foldとsplit再現
skf = StratifiedKFold(n_splits=10, shuffle=True,random_state=42)
for fold_index, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    x_train_full = X.loc[train_idx].reset_index(drop=True)
    y_train_full = y.loc[train_idx].reset_index(drop=True)
    x_test = X.loc[test_idx].reset_index(drop=True)
    y_test = y.loc[test_idx].reset_index(drop=True)

    x_train, x_feature, y_train, y_feature = train_test_split(
        x_train_full, y_train_full, train_size=split_portion, random_state=42
    )
    for noise_num in range(N):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # モデル読み込み
        model_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "models", f"epsilon{epsilon:.2f}",f"Fold{fold_index}",f"{noise_num}"))
        
        # 選択された属性の保存
        selected_columns_path = os.path.join(model_dir, "selected_columns.csv")
        #modelの保存
        model_name = "random_forest_model"
        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        
        
        classfier = joblib.load(model_path)
        selected_columns=pd.read_csv(selected_columns_path).squeeze().values
        file_path = os.path.join(
                BASE_DIR, "..", "data", "external", "domain",
                f"epsilon{epsilon:.2f}", f"Fold{fold_index}", f"{noise_num}",
                f"ADR_domain_T_{low_threshold}_L_{max_interval_len}.json"
            )
        with open(file_path, 'r', encoding='utf-8') as f:
                decided_domain_dict = json.load(f)
        # テストデータをノイズ無しでユーザ側処理
        x_test_ADR_RR, y_test_RR = user_process_train(x_test, y_test, "no_noise", decided_domain_dict , selected_columns)

        # 予測
        y_pred = classfier.predict(x_test_ADR_RR)

        print("Fold",fold_index,"number",noise_num,"Test accuracy:", balanced_accuracy_score(y_test_RR, y_pred))