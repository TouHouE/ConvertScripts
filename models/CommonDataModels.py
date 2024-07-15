import json
import argparse
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np

from utils import convert_utils as CUtils

__all__ = [
    'Partition', 'IspCtPair', 'DebugCard'
]
@dataclass
class Partition:
    proc_id: int
    data: list | np.ndarray | Iterable
    args: argparse.Namespace
    out_dir: str = field(default='./out')
    buf_dir: str = field(default='./buf')
    err_dir: str = field(default='./err')

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
    def patient_list(self) -> list | np.ndarray | Iterable:
        return self.data

@dataclass
class IspCtPair:
    image: str
    cp: float | int
    pid: str
    uid: str
    mask: str


class DebugCard:
    debug_attr_name: list

    def debug_card(self):
        card = {key: getattr(self, key, None) for key in self.debug_attr_name}
        return f'{json.dumps(card, indent=4)}'



