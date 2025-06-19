from typing import Dict
import pandas as pd
from pandas import Index
from typing import Dict

def dimension_reduction(
    chi_square_scores: Dict[str, float],
    n_components: int
) -> Index:
    """
    カイ二乗統計量に基づいて特徴量を上位 n_components 個選択し、
    選択された属性名の Index を返す。

    Parameters
    ----------
    chi_square_scores : Dict[str, float]
        各属性（列名）に対応するカイ二乗統計量の辞書。キーが属性名、値がそのカイ二乗値。
    n_components : int
        選択する次元（特徴量）の個数。辞書内に存在する属性数以上を指定するとすべて返す。

    Returns
    -------
    pandas.Index
        カイ二乗統計量が大きい順にソートし、上位 n_components 個の属性名を格納した Index。
        例えば、辞書内に属性が ['A','B','C','D'] あり、n_components=2 なら
        Index(['B','D']) のように返す。
    """
    if n_components <= 0:
        # 0 以下の指定は空の Index を返却
        return Index([], dtype=object)

    # pandas Series に変換し、カイ二乗統計量で降順ソート
    chi_series = pd.Series(chi_square_scores)
    sorted_columns = chi_series.sort_values(ascending=False).index

    # n_components が総属性数を超える場合は全属性を返す
    n = min(n_components, len(sorted_columns))
    selected = sorted_columns[:n]

    return selected


if __name__=="main":
        
