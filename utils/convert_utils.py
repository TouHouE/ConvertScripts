import os
import re
import json
import shutil
import zipfile
import argparse
import traceback
import datetime as dt
import subprocess as sp
from typing import Tuple, Any, Union, Callable, List, Optional, Dict

import pandas as pd
import pydicom as pyd

from utils.data_typing import CardiacPhase, PatientId
from utils import common_utils as ComUtils
from utils.hooker import disk_reconnect_watir

def _confirm_str(var) -> str:
    if var is None:
        return ''

    if isinstance(var, bytes) or isinstance(var, bytearray):
        return var.decode('ISO_IR 100', 'strict')
    if isinstance(var, int) or isinstance(var, float):
        return str(var)
    return var


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


def find_dicom_shape(dicom_container: List[pyd.FileDataset] | pyd.FileDataset) -> Tuple[int, int]:
    if not isinstance(dicom_container, list):
        dicom_container = [dicom_container]
    for dicom in dicom_container:
        image = getattr(dicom, 'pixel_array', None)
        if image is None:
            continue



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


def find_cp(dcm: pyd.FileDataset) -> CardiacPhase:
    """
    Get cardiac from those candidate.
        0x0020, 0x9241: NominalPercentageOfCardiacPhase
        0x01f1, 0x1041: LSCINTCT_GATING_DELAY
        0x7005, 0x1004: Toshiba/Canon Cardiac Reconstruction Gating Phase in Percent SH
        0x7005, 0x1005: Cardiac Reconstruction Gating Phase in ms SH 1
    Args:
        dcm:

    Returns:

    """
    if (_buf := dcm.get((0x0008, 0x0070))) is None:
        device_factory: str = ''
    else:
        device_factory = _confirm_str(_buf.value)
    candidate_list: List[Tuple[int, int]] = [(0x0020, 0x9241)]

    if device_factory in ['toshiba', 'canon']:
        candidate_list.extend([(0x7005, 0x1004), (0x7005, 0x1005), (0x01f1, 0x1041)])
    else:
        candidate_list.extend([(0x01f1, 0x1041), (0x7005, 0x1004), (0x7005, 0x1005)])

    for tag_candidate in candidate_list:
        cp = dcm.get(tag_candidate)
        if cp is None:
            continue
        cp = _confirm_str(cp.value)

        if len(cp.replace(' ', '')) < 1:
            return .0

        if all(_part.isdigit() for _part in cp.split('.')):
            return float(cp)
        cp = re.sub('[^0-9\.]', '', cp)
        cp_pack = cp.split('.')
        cp0_len = len(cp_pack[0])

        if cp0_len > 2:
            continue
        return float(cp)
    return .0


def find_desc(dcm: pyd.FileDataset) -> str:
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


def get_tag(
        dcm: pyd.FileDataset, tag: Tuple[int, int],
        default_value: Any = None, need_state: bool = True
) -> tuple[str, Any | None] | Union[Any, None]:
    if (info := dcm.get(tag)) is not None:
        return 'Normal', info.value if need_state else info.value
    return 'Unreadable', default_value if need_state else default_value


def get_init_prepare_df(dcm_collector) -> Dict[str, Any]:
    _prepare_df = dict()
    for key in dcm_collector.__dict__():
        _prepare_df[key] = []
    return _prepare_df


@disk_reconnect_watir
def commandline(ct_output_path: str, buf_path: str, verbose: int = 0, dcm2niix_path: str = './lib/dcm2niix.exe', **pykwargs) -> None:
    show_error_msg = pykwargs.get('debug', False)
    kwargs = dict()
    dcm2niix_args = list()
    if verbose == 1 or show_error_msg:
        kwargs = dict()
        # dcm2niix_args.extend(['-v', '2'])
        print(f'{" DCM2NIIX INFO ":=^40}')
    else:
        kwargs['stdout'] = sp.DEVNULL
        kwargs['stderr'] = sp.STDOUT

    sp.call(
        [dcm2niix_path,
         '-w', '1',  # if target is already, 1:overwrite it. 0:skip it
         '-z', 'y',  # Do .gz compress,
         *dcm2niix_args,
         '-o', ct_output_path,
         buf_path
         ], **kwargs
    )
    if verbose == 1:
        print(f'{" DCM2NIIX INFO End ":=^40}')
    return None

def make_path(df: pd.DataFrame):
    pid = df['pid'].unique()[0].lower()
    snum = int(df['snum'].unique()[0])
    uid = df['uid'].unique()[0]
    cp = float(df['cp'].unique()[0])
    return f'{pid}/{snum}/{uid}/{cp}'


def legal_isp_dicom_path(path: str) -> bool:
    return path.endswith('.dcm')


def legal_ct_dicom_path(path: str) -> bool:
    """
    Check if a path is legal, if got <instance number>[<number>].dcm mean duplication file.
    the file extension must .dcm.
    :param path:
    :return : is legal or not
    """
    file_name = re.split('[/\\\]', path)[-1]
    is_dcm = path.endswith('.dcm')
    pure_name = file_name.split('.')[0]
    not_dup = re.match(r'.*\[[0-9]{1,}\]', pure_name) is None
    # not_dup = file_name.split('.')[0].isdigit()
    return is_dcm and not_dup


def legal_patient_folder(path: str | List[Union[str, PatientId]], ignore_condition: Optional[List[PatientId]] = None) -> bool:
    if isinstance(path, str):
        patient_name: str = re.split('[/\\\]', path)[-1]
    else:
        patient_name: str = path[-1]
        path: str = '/'.join(path)
    is_dir = os.path.isdir(path)
    is_ignore_patient = False

    if ignore_condition is not None:
        is_ignore_patient = patient_name in ignore_condition
    return is_dir and not is_ignore_patient


def unzip(args, folder, member) -> str | List[str]:
    """
        Return a list of error message if got error, otherwise, the return will be the patient id
    :param args:
    :param folder:
    :param member:
    Returns:

    """
    patient_name = member.split('.')[0]
    try:
        with zipfile.ZipFile(f'{args.data_root}/{folder}/{member}', 'r') as unzipper:
            top = unzipper.filelist[0]
            if top.filename == patient_name:
                # This statement for if the patient folder already in *.zip
                # Thus, I don't prepare a folder to store all CT-series
                dst = f'{args.data_root}/{folder}'
            else:
                # This statement for if the patient folder not in the *.zip.
                dst = f'{args.data_root}/{folder}/{patient_name}'

            unzipper.extractall(dst)
    except Exception as e:
        content = ['=' * 30, f'{args.data_root}/{folder}/{member}', traceback.format_exc()]
        return content

    return patient_name


def record_offal_sample(offal_isp, offal_ct, args):
    """
    Store all offal data under corresponding patient's folder with json format
    Args:
        offal_isp(List[str]): All unmatched isp finding path (The source path)
        offal_ct(List[str]): All unmatched ct series path (The .nii.gz storage path)
        args:

    Returns:
        None
    """
    unpair_path = rf'{args.meta_dir}/unpair/{args.pid}'
    os.makedirs(unpair_path, exist_ok=True)
    unpair_obj = dict(isp=[], ct=[])
    for oisp in offal_isp:
        unpair_obj['isp'].append(oisp.folder_name)
    for o_ct in offal_ct:
        # Here is old method.
        # unpair_obj['ct'].append(o_ct.final_path)
        unpair_obj['ct'].append(o_ct.get_store_path())
    info = {
        'Offal CT': len(unpair_obj['ct']),
        'Offal ISP': len(unpair_obj['isp'])
    }
    ComUtils.print_info('Offal CT&ISP', info, args=args)
    with open(f'{unpair_path}/unpair.json', 'w+') as jout:
        json.dump(unpair_obj, jout)





# Filter function


def filter_legal_patient_folder(
        args: argparse.Namespace, ignore_condition: Optional[List[PatientId]] = None
) -> List[PatientId]:
    """

    Args:
        args: the script's argument
        ignore_condition(Optional[List[PatientId])]: all ignored patient id

    Returns:
        A list that store all patient id
    """
    legal_patient_list: List[PatientId] = list()

    for x in os.listdir(args.data_root):
        if legal_patient_folder([args.data_root, x], ignore_condition):
            legal_patient_list.append(PatientId(x))
    return legal_patient_list


def filter_legal_dcm(dcm_list: List[str], is_ct=True) -> List[str]:
    legal_dcm: List[str] = list()
    judge_func: Callable = legal_ct_dicom_path if is_ct else legal_isp_dicom_path

    for name in dcm_list:

        if judge_func(name):
            legal_dcm.append(name)
    return legal_dcm


def filter_not_dir(file_name_list: List[str]) -> List[str]:
    return list(filter(lambda x: 'DIRFILE' not in x, file_name_list))


