import json
import argparse
import multiprocessing as mp
import dataclasses as DC
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Union, TypeVar, Generic
import os
import numpy as np

from utils import convert_utils as CUtils

__all__ = [
    'Partition', 'IspCtPair', 'DebugCard', 'SegmentMetaPack'
]
T = TypeVar('T')


@dataclass
class Partition(Generic[T]):
    """
        This class using to package all of multi-processing scripts needed data. It contains:
        Attributes:
            proc_id: The id of the current process
            data: An iterable of waiting for each processing processed.
            args: handle arguments come from command-line
            Optional attributes:
            out_dir: The directory to save the output(*.nii.gz & *.json come from dcm2niix.exe) to.
            buf_dir: The directory to save the buffer(*.dcm file) to.
            err_dir: The directory to save the error message to.
            PID: same as `proc_id`
            patient_list: same as `data`

    """
    proc_id: int
    data: Iterable[T]
    args: argparse.Namespace
    out_dir: str = field(default='out')
    buf_dir: str = field(default='buf')
    err_dir: str = field(default='err')

    def __post_init__(self):
        if self.out_dir is None:
            self.out_dir = './out'
        if self.buf_dir is None:
            self.buf_dir = './buf'
        if self.err_dir is None:
            self.err_dir = './err'

    @property
    def PID(self) -> int:
        return self.proc_id

    @property
    def patient_list(self) -> Sequence[Iterable]:
        return self.data

@dataclass
class IspCtPair:
    image: str
    cp: float | int
    pid: str
    uid: str
    mask: str


class DictableDataClass:
    def __dict__(self):
        result = dict()
        for key, value in DC.asdict(self).items():
            if value is None:
                continue
            if DC.is_dataclass(value):
                value = dict(value)
            result[key] = value
        return result


@dataclass(frozen=True, eq=True, order=True)
class Detail(DictableDataClass):
    path: str | os.PathLike = field(hash=True)
    vessel: str | os.PathLike = field(hash=True)
    desc: str = field(hash=True)

    @property
    def json(self):
        return self.__dict__()


@dataclass
class SegmentMetaPack(DictableDataClass):
    cp: float | int = field(hash=True)
    pid: str = field(hash=True)
    uid: str = field(hash=True)
    image: str | os.PathLike = field(hash=True)
    mask: str | os.PathLike
    plaque: Optional[str | os.PathLike] = field(default=None, hash=True)
    details: Optional[Sequence[Detail | dict]] = field(default=None, hash=True)

    def __post_init__(self):
        if self.details is not None:
            self.details = tuple(sorted([Detail(**_detail) for _detail in self.details]))

    @property
    def json(self):
        return self.__dict__()


class DebugCard:
    debug_attr_name: list

    def debug_card(self):
        card = {key: getattr(self, key, None) for key in self.debug_attr_name}
        return f'{json.dumps(card, indent=4)}'



