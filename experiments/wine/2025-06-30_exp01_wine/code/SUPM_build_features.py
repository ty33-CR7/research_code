from typing import Literal


def SUPM_Domain(
        original_domain_column :list,
        column_type: Literal['numerical', 'categorical'],
        bins: int = 5) -> list[set]:
    """
    SUPMによるドメインを定義、数値データはL個の等間隔、カテゴリデータはmod Lを、list[set]を用いて表す。
    """
    if column_type not in ('numerical', 'categorical'):
        raise ValueError(f"column_type は 'numerical' または 'categorical' で指定してください。: {column_type}")
    if column_type == 'numerical':
        def split_list_into_tuples(lst, n):
            """リストをn個ずつのタプルに分割"""
            return [set(lst[i:i+n]) for i in range(0, len(lst), n)]
        SUPM_Domain=split_list_into_tuples(original_domain_column, bins)
    else:
        def mod_grouping(lst, n):
            """インデックスの mod n に基づいてグループ化してタプル化"""
            groups = [[] for _ in range(n)]
            for i, val in enumerate(lst):
                groups[i % n].append(val)
            return [set(group) for group in groups]
        SUPM_Domain = mod_grouping(original_domain_column, bins)
    return SUPM_Domain

