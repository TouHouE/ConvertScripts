import datetime as dt
import os
import pandas as pd
import pydicom as pyd
from typing import Tuple, Any, Union, Iterable
import re
import shutil
import subprocess as sp
import traceback
from dataclasses import dataclass
import numpy as np
import nibabel as nib
from operator import methodcaller

@dataclass
class Partition:
    proc_id: int
    data: list | np.ndarray | Iterable
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


def _get_cp(dcm: pyd.FileDataset) -> int | float:
    # cp = None
    for tag_candidate in [(0x0020, 0x9241), (0x01f1, 0x1041), (0x7005, 0x1004), (0x7005, 0x1005)]:
        cp = dcm.get(tag_candidate)
        if cp is None:
            continue
        if (cand_cp := cp.value).isdigit():
            return float(cand_cp)
        else:
            return float(cand_cp[:-1])

    return .0


def get_tag(dcm: pyd.FileDataset, tag: Tuple[int, int], default_value: Any = None, need_state: bool = True) -> tuple[str, Any | None] | Union[Any, None]:
    if (info := dcm.get(tag)) is not None:
        return 'Normal', info.value if need_state else info.value
    return 'Unreadable', default_value if need_state else default_value

def _make_path(df: pd.DataFrame):
    pid = df['pid'].unique()[0].lower()
    snum = int(df['snum'].unique()[0])
    uid = df['uid'].unique()[0]
    cp = float(df['cp'].unique()[0])
    return f'{pid}/{snum}/{uid}/{cp}'
class DCMFile:
    path: str
    uid: str
    snum: int
    cp: int| float
    pid: str
    inum: int
    disk_size: int
    size_unit: str

    def __init__(self, dcm_path, size_unit='m'):
        self.path = dcm_path
        dcm = pyd.dcmread(self.path)
        self.disk_size = os.stat(dcm_path).st_size
        self.size_unit = size_unit
        self.cp = _get_cp(dcm)
        status, self.uid = get_tag(dcm, (0x0020, 0x000e))
        status, self.stime = get_tag(dcm, (0x0008, 0x0030))
        status, self.snum = get_tag(dcm, (0x0020, 0x0011), None)
        status, self.inum = get_tag(dcm, (0x0020, 0x0013))
        status, self.pid = get_tag(dcm, (0x0010, 0x0020))

    def __eq__(self, other: Any) -> bool:
        cp2 = (float(self.cp) * float(other.cp)) ** .5
        same_cp = (cp2 == self.cp) and (cp2 == other.cp)
        snum2 = (int(self.snum) * int(other.snum)) ** .5
        same_snum = (self.snum == snum2) and (other.snum == snum2)
        same_uid = self.uid == other.uid
        same_inum = self.inum == other.inum
        same_pid = self.pid.lower() == other.pid.lower()

        if not same_pid:
            return False
        if snum2 != 0:
            if not same_snum:
                return False
        if not same_uid:
            return False
        if cp2 != 0:
            if not same_cp:
                return False
        if not same_inum:
            return False
        return True

    def __dict__(self):
        return {
            'uid': self.uid,
            'snum': self.snum,
            'path': self.path,
            'pid': self.pid,
            'cp': self.cp,
            'stime': self.stime,
            'inum': self.inum,
            'size': self.disk_size,
            'sunit': self.size_unit
        }


class CTFile:
    shape: tuple
    abs_path: str
    dcm_size: int
    nii_size: int

    def __init__(self, df: pd.DataFrame, buf_dir='./buf', out_dir='./out', error_dir='./on_error'):
        self.df = df
        self.dcm_size = df['size'].sum()
        self.size = df.size

        self.snum = self.df['snum'].unique().tolist()[0]
        self.cp = self.df['cp'].unique().tolist()[0]
        self.uid = self.df['uid'].unique().tolist()[0]
        self.pid = self.df['pid'].unique().tolist()[0]

        # <pid>/<snum>/<uid>/<cp>
        suffix = _make_path(df)
        self.buf_root = buf_dir
        self.out_root = out_dir
        self.buf_dir = f'{buf_dir}/{suffix}'
        self.out_dir = f'{out_dir}/{suffix}'
        self.info_dir = f'{out_dir}/{self.pid}'

        self.error_dst = f'{error_dir}/{self.pid}.txt'
        self.ignore_tag0008_0030 = not df['stime'].is_unique
        os.makedirs(self.buf_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(error_dir, exist_ok=True)
        self._copy2buf()
        self.store_nifit()
    def _copy2buf(self):
        for didx in range(len(self.df)):
            path = self.df['path'].iloc[didx]
            dcm_name = re.split('[/\\\]', path)[-1]
            dcm = pyd.dcmread(path)
            dst = f'{self.buf_dir}/{dcm_name}'

            if self.ignore_tag0008_0030:
                dcm[(0x0008, 0x0030)].value = dt.time(0, 0, 0, 0)
                dcm.save_as(dst)
            else:
                shutil.copyfile(path, dst)

    def clean_buf_dir(self):
        try:
            shutil.rmtree(self.buf_dir)
        except Exception as e:
            with open(self.error_dst, 'a+') as fout:
                fout.write(f'{"=" * 30}\n')
                fout.write(traceback.format_exc())

    def store_nifit(self) -> None:
        commandline(self.out_dir, self.buf_dir)
        target = list(filter(lambda x: not x.endswith('.json'), os.listdir(self.out_dir)))[0]
        self.abs_path = f'{self.out_dir}/{target}'
        self.nii_size = os.stat(self.abs_path).st_size
        # self.shape = nib.load(self.abs_path).get_fdata().shape
        self.clean_buf_dir()

    def compress_ratio(self) -> float:
        return self.nii_size / self.nii_size

    def str_ratio(self):
        return f'{self.abs_path},{self.nii_size},{self.dcm_size},{self.compress_ratio()}'

    def __repr__(self) -> str:
        txt = '=' * 30
        txt = f'{txt}\nPatient ID   : {self.pid}'
        txt = f'{txt}\nSeries Number: {self.snum}'
        txt = f'{txt}\nSeries UID   : {self.uid}'
        txt = f'{txt}\nCardiac Phase: {self.cp}'
        txt = f'{txt}\nStorage Path : {self.abs_path}'

        return txt
def split2ct(df: pd.DataFrame, buf_dir='./buf', out_dir='./out') -> list[CTFile]:
    all_split = []

    for snum_value, snum_entity in df.groupby('snum'):
        for uid_value, uid_entity in snum_entity.groupby('uid'):
            for cp_value, cp_entity in uid_entity.groupby('cp'):
                all_split.append(CTFile(cp_entity, buf_dir, out_dir))

    return all_split


def commandline(ct_output_path: str, buf_path: str, verbose: int = 0, dcm2niix_path: str = './lib/dcm2niix.exe'):
    if verbose == 1:
        kwargs = dict()
        print(f'{" DCM2NIIX INFO ":=^40}')
    else:
        kwargs = dict(stdout=sp.DEVNULL, stderr=sp.STDOUT, creationflags=sp.CREATE_NO_WINDOW)


    sp.call(
        [dcm2niix_path,
         '-w', '1',  # if target is already, 1:overwrite it. 0:skip it
         '-z', 'y',  # Do .gz compress,
         '-o', ct_output_path,
         buf_path
         ], **kwargs
    )
    if verbose == 1:
        print(f'{" DCM2NIIX INFO End ":=^40}')


def get_now(t0):
    # t0 = dt.datetime.now()
    return f'{t0:%Y-%m-%d %H:%M:%S}'