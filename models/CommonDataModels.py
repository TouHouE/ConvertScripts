import argparse
import multiprocessing as mp
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from utils import convert_utils as CUtils

__all__ = [
    'Partition'
]
@dataclass
class Partition:
    proc_id: int
    data: list | np.ndarray | Iterable
    args: argparse.Namespace
    out_dir: str
    buf_dir: str
    err_dir: str

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

