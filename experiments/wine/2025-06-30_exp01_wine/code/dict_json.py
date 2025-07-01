import json
import os
from typing import Dict, List, Set, Any

def dict_domain_to_json(
    dict_domain: Dict[str, List[Set[Any]]],
    file_path: str
) -> int:
    """
    dict[str, list[set]] を JSON ファイルに保存する。

    Parameters
    ----------
    dict_domain : dict[str, list[set]]
        保存対象の辞書。属性名をキー、ビン集合を要素とするリストを値とする。
    file_path : str
        JSON ファイルを書き出すパス。

    Returns
    -------
    int
        0: 正常終了、1: エラー発生
    """
    try:
        domains_serializable = {
            attr: [sorted(list(s)) for s in bins]
            for attr, bins in dict_domain.items()
        }
        with open(file_path, "w") as f:
            json.dump(domains_serializable, f, ensure_ascii=False, indent=2)
        return 0
    except Exception as e:
        print(f"[Error] dict_domain_to_json: {e}")
        return 1


def json_to_dict_domain(file_path: str) -> Dict[str, List[Set[Any]]]:
    """
    JSON ファイルを読み込んで dict[str, list[set]] を復元する。

    Parameters
    ----------
    file_path : str
        読み込む JSON ファイルのパス。

    Returns
    -------
    dict[str, list[set]]
        JSON 内のネストされたリストを集合に変換して返す。

    Raises
    ------
    FileNotFoundError
        file_path が存在しない場合。
    json.JSONDecodeError
        JSON のパースに失敗した場合。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")

    with open(file_path, "r") as f:
        data = json.load(f)

    dict_domain: Dict[str, List[Set[Any]]] = {}
    for attr, bins_as_lists in data.items():
        dict_domain[attr] = [set(inner_list) for inner_list in bins_as_lists]
    return dict_domain




