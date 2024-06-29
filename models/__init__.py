import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from models import TaipeiDataModels as Taipei
from models import HsinchuDataModels as Hsinchu


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
