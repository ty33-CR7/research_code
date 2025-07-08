import math
import numpy as np
from typing import List, Tuple, Dict, Optional,Set,Any,Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin



class RRNoiseTransformer(BaseEstimator, TransformerMixin):
    """
    一般化ランダム化応答（Randomized Response, RR）ノイズ変換器。

    各カテゴリ値を確率 p で保持し、それ以外の確率で他のカテゴリに一様に置換します。
    大規模データに対応できるように NumPy を用いてベクトル化しています。
    """

    def __init__(
        self,
        epsilon: float,
        dict_domain: Optional[Dict[str, List[Set[Any]]]]= None,
        rng=None
    ):
        """
        Parameters
        ----------
        epsilon : float
            プライバシー予算 ε。大きい場合はノイズをスキップします。
        dict_domain : dict[str, list[set]]
        保存対象の辞書。属性名をキー、ビン集合を要素とするリストを値とする。
        """
        self.epsilon = epsilon
        domain_sizes={key:len(interval_set)for key,interval_set in dict_domain.items()}
        self.domain_sizes = domain_sizes or {}
        self.rng = rng if rng else np.random.default_rng()

    def fit(self, X: pd.DataFrame, y=None) -> 'RRNoiseTransformer':
        """
        DataFrame からドメインサイズを推定し、各カテゴリの保持確率 p を事前計算します。

        Parameters
        ----------
        X : pd.DataFrame
            整数コード化されたカテゴリカル DataFrame。
        y : None
            未使用。

        Returns
        -------
        self
        """
        for col in X.columns:
            if col not in self.domain_sizes:
                self.domain_sizes[col] = int(X[col].nunique())
        self.feature_names_in_ = list(X.columns)
        # カラムごとに p を計算してキャッシュ
        self._p_map: Dict[str, float] = {}
        for col, L in self.domain_sizes.items():
            if self.epsilon != "no_noise":
                exp_eps = math.exp(self.epsilon)
                self._p_map[col] = exp_eps / (L + exp_eps - 1)
            else:
                self._p_map[col] = 1.0
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y=None,
        skip_noise: bool = False
    ) -> pd.DataFrame:
        """
        一般化 RR ノイズを DataFrame に適用します。

        Parameters
        ----------
        X : pd.DataFrame or pd.Series
            整数コード化されたカテゴリカルデータ (0～L-1) の DataFrame または Series。
        skip_noise : bool
            True の場合、ε が大きい場合と同様にノイズを付与せずにそのまま返します。

        Returns
        -------
        pd.DataFrame
            ノイズが適用された DataFrame。
        """
        X_df = X.to_frame() if isinstance(X, pd.Series) else X.copy()
        if skip_noise or self.epsilon >= 50:
            return X_df.astype(int)

        for col in X_df.columns:
            arr = X_df[col].astype(int).to_numpy()
            n = arr.shape[0]
            L = self.domain_sizes.get(col, int(np.max(arr) + 1))
            p = self._p_map.get(col, math.exp(self.epsilon) / (L + math.exp(self.epsilon) - 1))

            # ノイズ適用の要否を判定するマスク
            rnd = self.rng.random(n)
            keep = rnd <= p

            # 異なる値への置換を保証
            new_vals = arr.copy()
            replace_indices = np.where(~keep)[0]
            for i in replace_indices:
                choices = list(range(L))
                choices.remove(arr[i])
                new_vals[i] = self.rng.choice(choices)
            # 置換
            X_df[col] = new_vals
        return X_df

    def get_feature_names_out(
        self,
        input_features: Optional[List[str]] = None
    ) -> List[str]:
        """
        変換後の特徴量名を返します。

        Parameters
        ----------
        input_features : Optional[List[str]]
            外部から指定された特徴量名リスト。

        Returns
        -------
        List[str]
            フィット時に保存した特徴量名、もしくは input_features。
        """
        return getattr(self, 'feature_names_in_', input_features or [])


class OUENoiseTransformer(BaseEstimator, TransformerMixin):
    """
    最適化一元符号化（Optimized Unary Encoding, OUE）ノイズ変換器。

    バイナリ DataFrame (0/1) に対して、1 を確率 p で保持し、0 に変換、
    また 0 を確率 q で 1 に変換する二元ランダム化応答を行います。
    """

    def __init__(self, epsilon: float,rng=None):
        """
        Parameters
        ----------
        epsilon : float
            プライバシー予算 ε。
        """
        self.epsilon = epsilon
        self.p = 0.5
        self.q = 1.0 / (math.exp(epsilon) + 1.0)
        self.rng = rng if rng else np.random.default_rng()

    def fit(self, X: pd.DataFrame, y=None) -> 'OUENoiseTransformer':
        """
        特徴量名を保存します。
        """
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y=None,
        skip_noise: bool = False
    ) -> pd.DataFrame:
        """
        OUE によるノイズを DataFrame にベクトル化して適用します。

        Parameters
        ----------
        X : pd.DataFrame
            バイナリ (0/1) DataFrame。
        skip_noise : bool
            True の場合、ノイズを付与せずにそのまま返します。

        Returns
        -------
        pd.DataFrame
            ノイズ適用後の DataFrame。
        """
        X_copy = X.astype(int)
        if skip_noise or self.epsilon >= 1e6:
            return X_copy.copy()

        arr = X_copy.values
        rnd = self.rng.random(arr.shape)
        noisy = np.empty_like(arr)
        mask1 = arr == 1
        mask0 = ~mask1
        noisy[mask1] = (rnd[mask1] <= self.p).astype(int)
        noisy[mask0] = (rnd[mask0] <= self.q).astype(int)

        return pd.DataFrame(noisy, index=X.index, columns=X.columns)

    def get_feature_names_out(
        self,
        input_features: Optional[List[str]] = None
    ) -> List[str]:
        """
        変換後の特徴量名を返します。
        """
        return getattr(self, 'feature_names_in_', input_features or [])


def get_oue_frequency(
    series: pd.Series,
    categories: List[str],
    epsilon: float,
    calculate_mse: bool = False
) -> Tuple[pd.Series, Optional[float]]:
    """
    OUE を用いてカテゴリカルシリーズの頻度分布を推定します。

    Parameters
    ----------
    series : pd.Series
        推定対象のカテゴリカルシリーズ。
    categories : List[str]
        カテゴリの全ドメイン。
    epsilon : float
        プライバシー予算 ε。
    calculate_mse : bool
        True の場合、真のカウントとの MSE も返します。

    Returns
    -------
    Tuple[pd.Series, Optional[float]]
        推定頻度シリーズと MSE（calculate_mse=True の場合）。
    """
    cat = pd.Categorical(series, categories=categories)
    df_onehot = pd.get_dummies(cat, prefix=series.name, prefix_sep='%', dtype=int).fillna(0)

    if epsilon != "no_noise":
        transformer = OUENoiseTransformer(epsilon)
        noisy = transformer.fit_transform(df_onehot)
        n = noisy.shape[0]
        est = (noisy.sum(axis=0) - n * transformer.q) / (transformer.p - transformer.q)
    else:
        est = df_onehot.sum(axis=0)

    if calculate_mse:
        true_counts = df_onehot.sum(axis=0)
        est_clip = est.clip(lower=0)
        mse = ((est_clip - true_counts) ** 2).mean()
        return est, mse

    return est, None


def get_pair_explanatory_and_response(
    df_x: pd.DataFrame,
    df_y: Union[pd.Series, pd.DataFrame]
) -> pd.DataFrame:
    """
    説明変数 df_x と目的変数 df_y の値を
    '属性*クラス' 形式で結合した DataFrame を返します。
    """
    y = df_y.iloc[:, 0] if isinstance(df_y, pd.DataFrame) else df_y
    df_pair = df_x.copy().astype(str)
    for col in df_pair.columns:
        df_pair[col] = df_pair[col] + '*' + y.astype(str)
    return df_pair

def attribute_class_set(
    domain: List[str],
    y: pd.Series
) -> List[str]:
    """
    属性の値ドメインと目的変数 y のクラス集合から、
    '属性*クラス' の文字列リスト（直積）を返す。

    Parameters
    ----------
    domain : List[str]
        属性の値ドメイン（文字列リスト）。
    y : pd.Series
        目的変数。ユニークなクラスラベルを持つ。

    Returns
    -------
    List[str]
        ['val1*classA', 'val1*classB', 'val2*classA', ...] のようなリスト。
    """
    # y のクラスを文字列化してソート
    classes = sorted(y.dropna().unique().astype(str))
    # ドメイン × クラス の直積
    return [f"{val}*{cls}" for val in domain for cls in classes]


def make_cross_tab_from_domain(
    frequency: pd.Series,
    domain: List[str],
    column,
    y: pd.Series
) -> pd.DataFrame:
    """
    頻度シリーズと属性ドメイン＋クラス情報から
    クロス集計表を生成する。

    Parameters
    ----------
    frequency : pd.Series
        インデックスが '属性*クラス' の形をした頻度シリーズ。
    domain : List[str]
        属性の値ドメイン（インデックスになる）。
    y : pd.Series
        目的変数。ユニークなクラスラベルを持つ。

    Returns
    -------
    pd.DataFrame
        行: domain の各属性値, 列: y の各クラス, 値: frequency の対応値（ない場合は 0）。
    """
    # 属性×クラス の直積リストを取得
    pairs = attribute_class_set(domain, y)

    # frequency を辞書化（キー: 'val*class', 値: 頻度）
    freq_dict = frequency.to_dict()

    # クラスラベルを同じ順序で取り出す
    classes = sorted(y.dropna().unique().astype(str))

    # DataFrame のデータ部分を組み立て
    data = {
        cls: [freq_dict.get(f"{column}%{val}*{cls}", 0) for val in domain]
        for cls in classes
    }

    # クロス集計表を作成
    df = pd.DataFrame(data, index=domain)

    # 負の値があれば 0 にクリップ（必要なら）
    return df.clip(lower=0)

