import json
import argparse
import datetime as dt
from typing import Dict, List, Any
from constant import STATUS_LEN


def print_info(status: str, info: str | Dict[str, Any], args: argparse.Namespace):
    """
    Using to print the program message on terminal with format:
    `Process-{proc_id}|[@param{status}]|[progress / total patient]|[patient id]|info, time:{start time}`
    Args:
        status: like Start, End, Error, Loading
        info: What info want to print
        args: must contain
            <br>- proc_id
            <br/>- patient_progress: like "current patient's index/# of patient"
            |- pid: patient's id
    """
    _proc = f'Process-{args.proc_id:02}'
    _status = f'{status:^{STATUS_LEN}}'
    _p_prog = f'{args.patient_progress}'
    t0 = dt.datetime.now()

    if isinstance(info, dict):
        info_dict = info.copy()
        info: str = ''
        for key, value in info_dict.items():
            info = f'{info} {key}: {value},'
    if len(info) < 1:
        print(f'{_proc}|[{_status}]|[{_p_prog}]|[{args.pid}]| time:{t0:%Y-%m-%d %H:%M:%S}')
    else:
        print(f'{_proc}|[{_status}]|[{_p_prog}]|[{args.pid}]| {info} time:{t0:%Y-%m-%d %H:%M:%S}')


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