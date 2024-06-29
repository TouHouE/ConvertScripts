import re
import os
import shutil
import traceback
from typing import Any, Callable

import pandas as pd
import pydicom as pyd

from utils import convert_utils as CUtils
from utils.data_typing import CardiacPhase


class HsinchuDicomCollector:
    """
        This class is using to collect all of .dcm from same patient.
    """
    path: str
    uid: str
    snum: int  # Series Number
    cp: CardiacPhase
    pid: str
    inum: int  # Instance Number
    disk_size: int
    size_unit: str

    def __init__(self, dcm_path, size_unit='m'):
        self.path = dcm_path
        dcm = pyd.dcmread(self.path)
        self.disk_size = os.stat(dcm_path).st_size
        self.size_unit = size_unit
        self.cp = CUtils.get_cp(dcm)
        status, self.uid = CUtils.get_tag(dcm, (0x0020, 0x000e))
        status, self.stime = CUtils.get_tag(dcm, (0x0008, 0x0030))
        status, self.snum = CUtils.get_tag(dcm, (0x0020, 0x0011), None)
        status, self.inum = CUtils.get_tag(dcm, (0x0020, 0x0013))
        status, self.pid = CUtils.get_tag(dcm, (0x0010, 0x0020))

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


class HsinchuCTHandler:
    """
        This class is using to handle a series of .dcm, its main functional is converted a series of .dcm to .nii.gz
    """
    shape: tuple
    abs_path: str
    dcm_size: int
    nii_size: int

    def __init__(self, df: pd.DataFrame, buf_dir, out_dir, error_dir, dcm2niix):
        self.df = df
        self.dcm_size = df['size'].sum()
        self.size = df.size

        self.snum = self.df['snum'].unique().tolist()[0]
        self.cp = self.df['cp'].unique().tolist()[0]
        self.uid = self.df['uid'].unique().tolist()[0]
        self.pid = self.df['pid'].unique().tolist()[0].lower()

        # <pid>/<snum>/<uid>/<cp>
        suffix = CUtils.make_path(df)
        self.buf_root = buf_dir
        self.out_root = out_dir
        self.dcm2niix = dcm2niix
        self.buf_dir = f'{buf_dir}/{suffix}'
        self.out_dir = f'{out_dir}/{suffix}'
        self.info_dir = f'{out_dir}/{self.pid}'

        self.error_dst = f'{error_dir}/{self.pid}.txt'
        self.ignore_tag0008_0030 = not df['stime'].is_unique
        os.makedirs(self.buf_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(error_dir, exist_ok=True)
        self._copy2buf(copy_func=CUtils.fix_copy if self.ignore_tag0008_0030 else CUtils.shutil_copy)
        self.store_nifit()

    def _copy2buf(self, copy_func: Callable):
        for didx in range(len(self.df)):
            path = self.df['path'].iloc[didx]
            dcm_name = re.split('[/\\\]', path)[-1]
            dcm = pyd.dcmread(path)
            dst = f'{self.buf_dir}/{dcm_name}'
            src_pack: tuple[pyd.FileDataset, str] = (dcm, path)
            copy_func(src_pack, dst)

    def clean_buf_dir(self):
        try:
            shutil.rmtree(self.buf_dir)
        except Exception as e:
            with open(self.error_dst, 'a+') as fout:
                fout.write(f'{"=" * 30}\n')
                fout.write(traceback.format_exc())

    def store_nifit(self) -> None:
        CUtils.commandline(self.out_dir, self.buf_dir, dcm2niix_path=self.dcm2niix)
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
