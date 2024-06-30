import json
import datetime as dt
from typing import Dict, List, Any


def time2str(_time: dt.datetime | dt.time | dt.timedelta) -> str:
    return f'{_time:%Y-%m-%d %H:%M:%S}'


def write_content(path, content: str | Dict[str, Any] | List[Any], cover: bool = True, as_json=False):
    """
    :param path:
    :param content: if write as .txt file, we change it to `List[str]` if is .json, can belong dict or list
    :param cover: decide used covered write or append write, `True` for 'w+' `False` for 'a+'.
    :param as_json: if `True` using `json.dump` to store, otherwise, using `IO.TextIOWrapper.write` to store

    :return: None
    """
    open_mode = 'w+' if cover else 'a+'

    with open(path, open_mode) as ostream:
        if as_json:
            json.dump(content, ostream)
        else:
            if isinstance(content, str):
                content: List[str] = [content]
            for line in content:
                ostream.write(f'{line}\n')