import numpy as np
import pandas as pd
from scipy.stats import chi2

def cross_chimerge(
        cross_table: pd.DataFrame,
        max_interval_len: int,
        initial_intervals: list[set]=None,
        p_value: float=0.05
)-> list[set]:
    """
    カテゴリ値を χ² マージし、最終的に max_interval_len 個の区間にまとめる。

    Parameters
    ----------
    cross_table : pd.DataFrame
        行がカテゴリ（属性値）、列がターゲットクラスのクロス集計表。
        例）index=['A','B','C'], columns=[0,1] など。
    max_interval_len : int
        最終的に残す区間（ビン）の数。これを超える場合は強制マージを行う。
    p_value : float, default=0.05
        χ² 検定の有意水準（自由度=1 としてカイ2乗臨界値を計算）。
    initial_intervals : list[set], optional
        もしあらかじめ「カテゴリをまとめた初期区間」を渡したい場合に指定。
        None の場合は、各カテゴリを 1 つずつの set にして初期化する。

    Returns
    -------
    intervals : list[set]
        最終的にマージされた区間のリスト。各要素は「カテゴリ値の集合（set）」。
    """
    # 0) 初期化：intervals_copy を作成
    if initial_intervals is None:
        intervals = [{idx} for idx in cross_table.index]
    else:
        # シャローコピーしておく
        intervals = initial_intervals

    # 1) χ² 閾値を計算（自由度 df_chi ）
    num_classes = cross_table.shape[1]
    df_chi = num_classes - 1
    threshold_chi = chi2.ppf(1 - p_value, df=df_chi )

    # 2) 隣接ペアの χ² を計算する関数
    def compute_adjacent_chi(interval_list: list[set]) -> list[float]:
        chi_list = []
        for i in range(len(interval_list) - 1):
            # 各区間に含まれるカテゴリ値で絞り込み、ターゲットごとの合計を取る
            cnt1 = cross_table.loc[list(interval_list[i])].sum().values
            cnt2 = cross_table.loc[list(interval_list[i+1])].sum().values

            tot = cnt1.sum() + cnt2.sum()
            comb = cnt1 + cnt2

            # いずれかのセルの合計が 0 の場合、χ² を 0 に近似する
            if np.any(comb == 0) or cnt1.sum() == 0 or cnt2.sum() == 0:
                chi_list.append(0.0)
            else:
                exp1 = comb * cnt1.sum() / tot
                exp2 = comb * cnt2.sum() / tot
                # クラス毎に (O−E)^2/E を計算して総和をとる
                chi_val = ((cnt1 - exp1) ** 2 / exp1 + (cnt2 - exp2) ** 2 / exp2).sum()
                chi_list.append(chi_val)
        return chi_list

    # 3) メインループ：１つずつ隣接ペアをマージしていく
    while True:
        # 区間が１つになったら打ち切り
        if len(intervals) <= 1:
            break
        chi_values = compute_adjacent_chi(intervals)
        min_idx = int(np.argmin(chi_values))
        min_chi = chi_values[min_idx]
        # （A）閾値ベースでマージするか：最小 χ² が閾値未満の場合、マージ
        if min_chi < threshold_chi and len(intervals) > max_interval_len:
            # どうしても「閾値ベース→最大数ベース」の順にしたい場合は下記のように分岐を調整。
            # しかしここでは「最大数までマージする」ために、閾値未満かどうかにかかわらず進めたいなら else 部分を使う。
            pass
        # （B）最大数ベース：現在の区間数が max_interval_len を下回ったら打ち切る
        if len(intervals) <= max_interval_len:
            break
        # それ以外のケース：最小 χ² を持つペアをマージする
        new_set = intervals[min_idx] | intervals[min_idx + 1]
        # intervals リストを更新（min_idx と min_idx+1 をまとめる）
        intervals = intervals[:min_idx] + [new_set] + intervals[min_idx + 2:]
    return intervals



def low_freq_merge_categories(
    cross_tab: pd.DataFrame,
    low_threshold: float 
) -> tuple[list[set], pd.Index]:
    """
    カテゴリ属性のクロス集計において、頻度が低いカテゴリをまとめ、
    残りを個別ビンとして返す。

    Parameters
    ----------
    cross_tab : pd.DataFrame
        行 index にカテゴリ値、列に集計対象クラスが入ったクロステーブル。
        例：index=['A','B','C'], columns=['class1','class2',...]
    threshold: float
        低頻度と判断する頻度数の閾値
    Returns
    -------
    domain_list : list of set
        - 低頻度カテゴリをまとめた set（リストの先頭要素）
        - 残りの各高頻度カテゴリを 1 要素 set としたリスト要素
        例： [{'A','B'}, {'C'}, {'D'}, …]
    low_freq_indices : pd.Index
        低頻度カテゴリの index。一括でマージ対象のカテゴリを参照したい場合に利用。

    Notes
    -----
    - 元コードではフラグ管理を用いて一度に複数のカテゴリを順次マージしていましたが、
      ここでは「低頻度カテゴリを先にまとめる → 残りを個別に set 化」という
      2 段階で処理することで可読性と保守性を高めています:contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}。
    - Pandas の機能を活用し、シリーズのフィルタリングやソートを簡潔に行っています:contentReference[oaicite:10]{index=10}。
    """
    # 1) カテゴリごとの合計頻度を計算
    freq_per_category: pd.Series = cross_tab.sum(axis=1)

    # 2) 低頻度カテゴリを抽出
    low_freq_indices: pd.Index = freq_per_category[
        freq_per_category < low_threshold
    ].index

    # 3) 各カテゴリを頻度順にソートしたリストを取得
    sorted_categories = list(
        freq_per_category.sort_values().index
    )

    # 4) 低頻度カテゴリと高頻度カテゴリを分離
    low_freq_cats = [cat for cat in sorted_categories if cat in low_freq_indices]
    high_freq_cats = [cat for cat in sorted_categories if cat not in low_freq_indices]

    # 5) ドメイン（ビン）を構築
    domain_list: list[set] = []
    if low_freq_cats:
        # 低頻度カテゴリをひとまとめにする
        domain_list.append(set(low_freq_cats))

    # 高頻度カテゴリはそれぞれを個別 set にして順序を維持して追加
    for cat in high_freq_cats:
        domain_list.append({cat})

    return domain_list, low_freq_indices



def low_freq_merge_numbers(
    cross_table: pd.DataFrame,
    low_threshold: float 
) -> tuple[list[set], pd.Index]:
    """
    インデックスが連続していなくても動作するように、
    低頻度インデックスを“ソートされたリスト上での隣接要素”とまとめる実装。

    Parameters
    ----------
    cross_table : pd.DataFrame
        行インデックスが任意の値をとり得るクロステーブル。
        例: インデックスが [10, 15, 20, 30] のように飛び飛びでもOK。
        列は集計対象のクラスや値など任意。

    low_threshold : float
        「(全行合計) × low_threshold 未満」の行インデックスを
        低頻度とみなす

    Returns
    -------
    domain_merge : list[set]
        低頻度インデックスを、それぞれ“ソート順で隣接する要素”とまとめたビン（集合）を返す。
        その結果、最終的に重複しない集合のリストとなる。

        例: 
          インデックス [10, 15, 20, 30, 40, 50]
          低頻度 [15, 30, 40]
          → ソートインデックス: [10, 15, 20, 30, 40, 50]
          → ビン化結果: [{10}, {15,20}, {30,40,50}]

    low_freq_indices : pd.Index
        低頻度と判定されたインデックスをそのまま返す。
    """
    # 1) 各行インデックスの合計頻度を取得
    freq_per_index: pd.Series = cross_table.sum(axis=1)

    # 2) 閾値（全体合計×low_threshold）を計算し、低頻度インデックスを抽出
    low_freq_indices: pd.Index = freq_per_index[freq_per_index < low_threshold].index

    # 3) インデックスをソートしたリストに変換
    sorted_indices: list = sorted(freq_per_index.index.tolist())

    low_set = set(low_freq_indices)
    visited: set = set()
    domain: list[set] = []

    # 4) ソート順に沿って、低頻度 or 高頻度を判定しながら隣接マージ
    for idx in sorted_indices:
        if idx in visited:
            # 既にビン化済みならスキップ
            continue

        if idx not in low_set:
            # 高頻度要素はいったん単独ビンとする
            domain.append({idx})
            visited.add(idx)

        else:
            # 低頻度要素は、次のソート順要素とともにまとめてビン化する
            group: set = {idx}
            visited.add(idx)
            j = idx

            # 以下で“ソートリスト中の隣接要素”を順次チェック、
            while True:
                #Python のリストメソッド list.index(value)ß
                pos = sorted_indices.index(j)
                if pos + 1 >= len(sorted_indices):
                    # 最後尾に到達したら終了
                    break

                next_idx = sorted_indices[pos + 1]
                
                # 隣接要素をビンに追加
                group.add(next_idx)
                visited.add(next_idx)

                # もし隣接要素が低頻度ならさらにその次も見る
                if next_idx in low_set:
                    j = next_idx

                    continue
                else:
                    # 隣接が高頻度になったらここでまとめ終わる
                    break

            domain.append(group)

    # 5) domain 内の重複集合はこの実装では発生せず、domain がそのまま最終結果となる
    return domain, low_freq_indices




def category_attribute_class_order(
    cross_table: pd.DataFrame,
    intervals: list[set]
) -> list[set]:
    """
    各 interval（インデックスの集合）における「クラスラベル 1 の割合」を計算し、
    その割合が小さい順に intervals をソートして返す。クラスラベルは2つと仮定している

    Parameters
    ----------
    cross_tab : pd.DataFrame
        行インデックスがカテゴリ値、列がクラスラベルに対応するクロステーブル。
        例: index=['A','B','C'], columns=[0,1] のように、クラスが 2 列ある想定。

    intervals : list[set]
        各要素が「そのインターバルに含まれるカテゴリ値（行インデックス）を持つ集合」。
        例: [{'A','B'}, {'C'} など]

    Returns
    -------
    sorted_intervals : list[set]
        クラスラベル 1 の割合が小さい順に並べ替えた intervals のリスト。
        各 interval がゼロサンプルの場合は割合を 0 として扱う。
    """
    def class1_ratio(interval: set) -> float:
        # interval に含まれる行を取り出し、クラスごとに合計
        counts = cross_table.loc[list(interval)].sum()
        total = counts.sum()
        # 全サンプル数が 0 の場合は 0 とみなす
        if total == 0:
            return 0.0
        # クラス列の 2 列目（iloc[1]）がクラス 1 の個数と想定
        return counts.iloc[1] / total

    # sorted() の key に、各     interval   に対する class1_ratio を指定,class_ratioの順に並び替えてくれる
    sorted_intervals = sorted(intervals, key=class1_ratio)
    return sorted_intervals


