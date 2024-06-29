import shutil
import datetime as dt
import subprocess as sp
from typing import Tuple, Any, Union, Callable

import pandas as pd
import pydicom as pyd

from utils.data_typing import CardiacPhase


def _confirm_str(var) -> str:
    if isinstance(var, bytes) or isinstance(var, bytearray):
        return var.decode('ISO_IR 100', 'strict')
    if isinstance(var, int) or isinstance(var, float):
        return str(var)
    return var


def find_isp_uid(isp: pyd.FileDataset) -> str:
    def _make_sure_is_str(_str):
        if isinstance(_str, str):
            return _str
        else:
            return _str.decode('ISO_IR 100', 'strict')

    uid_pack = isp.get((0x0008, 0x1115))
    if uid_pack is None:
        return _make_sure_is_str(isp.get((0x01e1, 0x1046)).value).split('_')[-3]

    return _make_sure_is_str(uid_pack.value[0][(0x0020, 0x000e)].value)


def shutil_copy(src_pack: tuple[pyd.FileDataset, str], dst: str) -> None:
    shutil.copy(src_pack[1], dst)


def fix_copy(src_pack: tuple[pyd.FileDataset, str], dst: str) -> None:
    """
        Usually only de-identify by Siemens device will use this function to ignore Study Time not equally problem.
    :param src_pack:
    :param dst:
    :return:
    """
    dcm = src_pack[0]
    dcm[(0x0008, 0x0030)].value = dt.time(0, 0, 0, 0)
    dcm.save_as(dst)


def get_cp(dcm: pyd.FileDataset) -> CardiacPhase:
    for tag_candidate in [(0x0020, 0x9241), (0x01f1, 0x1041), (0x7005, 0x1004), (0x7005, 0x1005)]:
        cp = dcm.get(tag_candidate)
        if cp is None:
            continue
        cp = _confirm_str(cp.value)

        if len(cp.replace(' ', '')) < 1:
            return .0

        if all(_part.isdigit() for _part in cp.split('.')):
            return float(cp)

        if not cp[-1].isdigit():
            return float(cp[:-1])

    return .0


def get_tag(
        dcm: pyd.FileDataset, tag: Tuple[int, int],
        default_value: Any = None, need_state: bool = True
) -> tuple[str, Any | None] | Union[Any, None]:
    if (info := dcm.get(tag)) is not None:
        return 'Normal', info.value if need_state else info.value
    return 'Unreadable', default_value if need_state else default_value


def get_desc(dcm: pyd.FileDataset) -> str:
    """
        Trying to get the description from an ISP Dicom file, the possible tag are (0x0008, 0x1032) and (0x0008, 0x103e)
    :param dcm: Single ISP dicom file
    :return: The description of the isp dicom, or empty string if no description could be found
    """
    for tag in [(0x0008, 0x1032), (0x0008, 0x103e)]:
        desc = dcm.get(tag)
        if desc is not None:
            return desc.value
    print(dcm)
    return ''


def commandline(ct_output_path: str, buf_path: str, verbose: int = 0, dcm2niix_path: str = './lib/dcm2niix.exe'):
    if verbose == 1:
        kwargs = dict()
        print(f'{" DCM2NIIX INFO ":=^40}')
    else:
        kwargs = dict(stdout=sp.DEVNULL, stderr=sp.STDOUT)

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


def make_path(df: pd.DataFrame):
    pid = df['pid'].unique()[0].lower()
    snum = int(df['snum'].unique()[0])
    uid = df['uid'].unique()[0]
    cp = float(df['cp'].unique()[0])
    return f'{pid}/{snum}/{uid}/{cp}'