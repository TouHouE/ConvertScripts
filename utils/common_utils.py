import re
import json
import argparse
import datetime as dt
from typing import Dict, List, Any, Callable
from constant import STATUS_LEN
import os
import logging

import numpy as np
from sklearn.model_selection import train_test_split, KFold
import pydicom as pyd

from utils import hooker

__all__ = ['is_numerical', 'time2str', 'print_info', 'write_content', 'make_vista_style_table', 'load_json', 'load_dcm', 'make_final_namespace', 'delete_file']


def is_numerical(foo: str):
    foo = foo.split('(')[0]
    pack = foo.split('.')
    if len(pack) > 2:
        return False

    if len(pack) == 2:
        return re.fullmatch('^[0-9]{1,}\.[0-9]{1,}[%]{0,1}$', foo)
    return re.fullmatch('^[0-9]{1,}[%]{0,1}$', foo) is not None


def time2str(_time: dt.datetime | dt.time | dt.timedelta) -> str:
    return f'{_time:%Y-%m-%d %H:%M:%S}'


def print_info(status: str, info: str | Dict[str, Any], args: argparse.Namespace):
    """
    Using to print the program message on terminal with format:
    `Process-{proc_id}|[@param {status}]|[progress / total patient]|[patient id]|info, time:{start time}`
    Args:
        status: like Start, End, Error, Loading
        info: What info want to print
        args: must contain
            <br>- proc_id: currently process's id
            <br>- patient_progress: like "current patient's index/# of patient"
            <br>- pid: patient's id
    """
    _proc: str = f'Process-{args.proc_id:02}'
    _status: str = f'{status:^{STATUS_LEN}}'
    _p_prog: str = f'{args.patient_progress}'
    t0: dt.datetime = dt.datetime.now()

    # convert the dict structure data into normal string.
    if isinstance(info, dict):
        info_dict = info.copy()
        info: str = ''
        for key, value in info_dict.items():
            info = f'{info} {key}: {value},'
    if len(info) < 1:  # No additional information for show
        print(f'{_proc}|[{_status}]|[{_p_prog}]|[{args.pid}]| time:{t0:%Y-%m-%d %H:%M:%S}')
    else:
        print(f'{_proc}|[{_status}]|[{_p_prog}]|[{args.pid}]| {info} time:{t0:%Y-%m-%d %H:%M:%S}')


@hooker.disk_reconnect_watir
def write_content(path, content: str | Dict[str, Any] | List[Any], cover: bool = True, as_json=False, **kwargs) -> None:
    """
    Store the content in disk.
    :param path: fully path of the file, must include file extension like .json or .txt
    :param content: if write as .txt file, we change it to `List[str]` if is .json, can belong dict or list
    :param cover: decide used covered write or append write, `True` for 'w+' `False` for 'a+'.
    :param as_json: if `True` using `json.dump` to store, otherwise, using `IO.TextIOWrapper.write` to store
    :param kwargs: additional variable, but must contain a key call :type argparse.Namespace: `gargs`
    :return: None
    """
    open_mode = 'w+' if cover else 'a+'
    # if not os.path.exists(path):

    with open(path, open_mode, encoding='utf-8') as ostream:
        if as_json:
            json.dump(content, ostream)
        else:
            if isinstance(content, str):
                content: List[str] = [content]
            for line in content:
                ostream.write(f'{line}\n')
    return None


def make_vista_style_table(raw_table, args):
    def _test_pack_mapper(test_pack):
        test_pack['label'] = test_pack['mask']
        return test_pack
    num_fold: int = 2 if (num_fold := args.num_fold) <= 2 else num_fold
    test_ratio: float = test_ratio if 0 <= (test_ratio := args.test_ratio) < 1 else test_ratio / len(raw_table['data'])

    ids_list = np.arange(len(raw_table['data']))
    try:
        train_ids_list, test_ids_list = train_test_split(ids_list, test_ratio)
    except TypeError:
        train_ids_list, test_ids_list = train_test_split(ids_list, test_size=test_ratio)
    folder = KFold(num_fold, shuffle=True, random_state=114514)
    test_list = [_test_pack_mapper(raw_table['data'][test_index]) for test_index in test_ids_list]
    train_list = list()
    append_train: Callable = train_list.append

    for current_fold, (_, val_index_seq) in enumerate(folder.split(train_ids_list)):
        for val_index in val_index_seq:
            train_pack = raw_table['data'][val_index]
            train_pack['fold'] = current_fold
            train_pack['label'] = train_pack['mask']
            append_train(train_pack)
    vista_table = dict(label=raw_table['name'], training=train_list, testing=test_list)
    return vista_table


def load_json(path):
    with open(path, 'r') as jin:
        return json.load(jin)


def load_dcm(path, **kwargs) -> pyd.FileDataset | None:

    try:
        return pyd.dcmread(path, **kwargs)
    except pyd.filereader.InvalidDicomError:
        return None


def make_final_namespace(args: argparse.Namespace) -> argparse.Namespace:
    if (config_path := getattr(args, 'config', None)) is None:
        return args

    config = load_json(config_path)

    for key, value in config.items():
        if getattr(args, key, None) is not None:
            continue
        setattr(args, key, value)
    return args


def delete_file(dst_file: str | os.PathLike) -> int:
    """

    Args:
        dst_file:

    Returns(int): using an integer to describe the status of delete `dst_file`.
    0 : OK
    -1: the `dst_file` does not exist(or already deleted).
    -2: Unexpected error.
    """
    try:
        os.remove(dst_file)
        status = 0
    except FileNotFoundError:
        logging.error(f'{dst_file} does not exist')
        status = -1
    except Exception:
        status = -2
    return status

