import numpy as np
import pandas as pd

def calculate_cramer_chi_statistic(
    cross_tab: pd.DataFrame,
    domain: list[set]=None
) -> float:
    """
    各 interval（カテゴリ集合）内のクラス分布と全体分布との差に基づいて
    χ²統計量を計算し、すべての interval について合算した値を返す。

    Parameters
    ----------
    cross_tab : pd.DataFrame
        行インデックスがカテゴリ値、列がクラスラベルのクロステーブル。
        例: index=['A', 'B', 'C'], columns=[0, 1, 2] のように、複数クラスに対応。
    intervals : list[set]
        各要素が「そのビンに含まれるカテゴリ値の集合」を表すリスト。
        例: [{'A', 'B'}, {'C'}]

    Returns
    -------
    float
        すべての interval に対する χ²統計量の合計値。
        各 interval の χ² は次の式で計算：
            obs_i = interval におけるクラス i の観測度数
            exp_i = (全体クラス i の度数) × (interval 内合計 / 全体合計)
            χ²_interval = Σ_i (obs_i − exp_i)² / exp_i
        ただし、exp_i が 0 の場合はその要素を無視して計算する。
    """
    # 全体のクラスごとの合計度数と総サンプル数を取得
    class_totals = cross_tab.sum(axis=0).to_numpy()      # 例: [total_class0, total_class1, ...]
    grand_total = class_totals.sum()                     # 全サンプル数

    chi_values = []
    if domain is None:
        domain = [{label} for label in cross_tab.index]

    for interval in domain:
        # interval に含まれるカテゴリ行の合計度数（クラスごと）
        observed = cross_tab.loc[list(interval)].sum(axis=0).to_numpy()
        interval_total = observed.sum()

        # 期待度数を計算
        # exp = class_totals * (interval_total / grand_total)
        expected = class_totals * (interval_total / grand_total)

        # exp が 0 のクラスは計算から除外
        mask = expected > 0
        chi_interval = np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask])
        chi_values.append(chi_interval)

    return float(np.sum(chi_values))

