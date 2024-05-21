import sys
import zipfile
import argparse
import traceback
import multiprocessing as mp
import json
from functools import partial
import datetime as dt
import subprocess as sp
import numpy as np
from typing import Union, Tuple, Any, Callable, List, TypeVar, NewType
import pydicom as pyd
import os
import re
import nibabel as nib
import isp_helper as ISPH
import dataclasses
import shutil
from operator import methodcaller

DIGIT2LABEL_NAME = {
    1: 'RightAtrium',
    2: 'RightVentricle',
    3: 'LeftAtrium',
    4: 'LeftVentricle',
    5: 'MyocardiumLV',
    6: 'Aorta',
    7: 'Coronaries8',
    8: 'Fat',
    9: 'Bypass',
    10: 'Plaque'
}
LABEL_NAME2DIGIT = {value: key for key, value in DIGIT2LABEL_NAME.items()}
CardiacPhase = NewType('CardiacPhase', Union[float, int])
FilePathPack = NewType('FilePathPack', tuple[pyd.FileDataset, str])
IS_ZIP: methodcaller = methodcaller('endswith', '.zip')

@dataclasses.dataclass
class Partition:
    PID: int
    patient_list: list[str]
    args: argparse.Namespace
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


def _confirm_str(var) -> str:
    if isinstance(var, bytes) or isinstance(var, bytearray):
        return var.decode('ISO_IR 100', 'strict')
    if isinstance(var, int) or isinstance(var, float):
        return str(var)

    return var


def _get_cp(dcm: pyd.FileDataset) -> int | float:
    # cp = None
    for tag_candidate in [(0x0020, 0x9241), (0x01f1, 0x1041), (0x7005, 0x1004), (0x7005, 0x1005)]:
        cp = dcm.get(tag_candidate)
        if cp is not None:
            cp = _confirm_str(cp.value)
            if all(_part.isdigit() for _part in cp.split('.')):
                return float(cp)

            if '%' in cp:
                return float(_confirm_str(cp).replace('%', ''))

    return .0


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
    # dcm = pyd.dcmread(src)
    dcm[(0x0008, 0x0030)].value = dt.time(0, 0, 0, 0)
    dcm.save_as(dst)


def get_tag(dcm: pyd.FileDataset, tag: Tuple[int, int], default_value: Any = None, need_state: bool = True) -> tuple[str, Any | None] | Union[Any, None]:
    if (info := dcm.get(tag)) is not None:
        return 'Normal', info.value if need_state else info.value
    return 'Unreadable', default_value if need_state else default_value


def legal_dcm_path(path: str) -> bool:
    """
    Check if a path is legal, if got <instance number>[<number>].dcm mean duplication file.
    the sub name must dcm.
    :param path:
    :return : is legal or not
    """
    file_name = re.split('[/\\\]', path)[-1]
    is_dcm = path.endswith('.dcm')
    # not_dup = re.match('\[[1-9]{1,}\]', path.split('.')[0][-3:]) is None
    not_dup = file_name.split('.')[0].isdigit()
    return is_dcm and not_dup


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


class ISPContainer:
    shape: tuple[int, int, int] | list[int, int, int]
    final_path: dict[str, str]
    plaque_num: int
    tissue_list: list[pyd.FileDataset]
    path_list: list[pyd.FileDataset]
    is_saved: bool
    def __init__(
            self, isp_list: list[str], pid: str,
            folder_name: str, output_dir: str = './mask',
            args: argparse.Namespace=None, verbose: int = 0
    ) -> None:
        """
        An ISPContainer object is used to package an isp annotation folder, each folder has multiple "Findings"
        :param isp_list: Only contains a sequence of isp annotation dicom file's path
        :param pid: Patient ID
        :param folder_name: This parameter is used to easily identify the original path
        :param output_dir: An ISPContainer object goal is to save the tissue mask, and plaque mask into nifit file,
        the mask will be saved in :param output_dir
        :param verbose: if is 1, then print all process detail, if is 0, don't print anything.
        """
        self.output_dir: str = output_dir
        self.total_list: list[str] = isp_list
        self.folder_name: str = folder_name
        self.pid: str = pid
        self.verbose: int = verbose
        self.args = args

        self.tissue_list = []
        self.path_list = []
        self.plaque_num = 0
        self.final_path = dict()
        self.is_saved = False
        uni_uid = set()

        # Process all isp dicom file under specify folder.
        for isp, isp_path in zip(map(pyd.dcmread, isp_list), isp_list):
            uni_uid.add(find_isp_uid(isp))

            if (_shape := get_tag(isp, (0x07a1, 0x1007), default_value=None, need_state=False)) is not None:
                self.shape = _shape

            if isp.ImageType[-1] != 'PATH':
                self.tissue_list.append(isp)
                continue

            self.path_list.append(isp)
            tamar_list: pyd.DataElement | None = isp.get((0x07a1, 0x1050))
            if tamar_list is not None:
                if not isinstance(tamar_list.value, list):
                    self.plaque_num += len(list(filter(lambda x: ISPH.legal_plaque(x), tamar_list.value)))
        desc = get_desc(isp)
        self.comment = desc

        cp = 0
        snum = 0
        if verbose == 1:
            print(f'{"Comment":13}: {self.comment}')

        if len(desc_pack := re.split('[, _]', desc)) > 0:
            if verbose == 1:
                print(desc_pack)
            pure_digit = list(filter(lambda x: x.isdigit(), desc_pack))
            cp_candidate = list(filter(lambda x: '%' in x, desc_pack))
            # print(f'Pure digit: {pure_digit}, cp candidate: {cp_candidate}')
            # Mean the format is cp%<sep>snum
            if len(cp_candidate) != 0:
                cp = float(_cp) if (_cp := cp_candidate[0][:-1]).isdigit() else 0
                # Need second check snum is exist or not
                snum = int(pure_digit[0]) if len(pure_digit) > 0 else snum
            # Mean the format is cp<sep>snum
            if len(pure_digit) > 1:
                cp = float(pure_digit[0])
                snum = int(pure_digit[1])

        self.snum = snum
        self.cp = cp

        # This for debug usage.
        if len(uni_uid) > 1:
            print(uni_uid)
        self.uid = uni_uid.pop()

    def _collect_tissue(self) -> np.ndarray:
        union_mask: np.ndarray | None = None
        for tisp in self.tissue_list:
            organ_mask, organ_name = ISPH.reconstruct_mask(tisp)
            organ_name = organ_name.replace(' ', '')
            organ_mask = np.rot90(organ_mask, k=3)
            if organ_name not in LABEL_NAME2DIGIT.keys():
                continue

            if union_mask is None:
                union_mask = np.zeros_like(organ_mask)
            union_mask[organ_mask != 0] = LABEL_NAME2DIGIT[organ_name]
        return union_mask

    def _collect_plaque(self, union_mask: np.ndarray, host_ct: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray | None]:
        union_plaque: np.ndarray | None = None
        if self.plaque_num < 1:
            return union_mask, union_plaque
        union_plaque = np.zeros_like(host_ct.get_fdata(), dtype=np.int16)

        for pisp in self.path_list:
            pack_list = ISPH.reconstruct_plaque(pisp, host_ct)
            for pack in pack_list:
                pmask, pname = pack
                union_mask[pmask != 0] = LABEL_NAME2DIGIT['Plaque']
                union_plaque[pmask != 0] = 1
        return union_mask, union_plaque

    def store_mask(self, ct: 'CTContainer'):
        if self.is_saved:
            return
        host_ct = ct.nifit_ct
        store_path = f'{self.output_dir}/{self.pid}/{self.uid}/{ct.cp}/{self.folder_name}'
        os.makedirs(store_path, exist_ok=True)
        union_mask: np.ndarray = self._collect_tissue()

        try:
            union_mask, union_plaque = self._collect_plaque(union_mask, host_ct)
        except Exception as e:
            with open(f'{self.args.err_dir}/plaque_error.txt', 'a+') as fout:
                # fout.write(f'{"=" * 30}\n')
                fout.write(f"{str(self)}Got error during collect_plaque\n")
                fout.write(traceback.format_exc())
            union_plaque = None

        mask_nii = nib.Nifti1Image(union_mask, host_ct.affine)
        nib.save(mask_nii, f'{store_path}/union_mask.nii.gz')
        self.final_path['mask'] = f'{store_path}/union_mask.nii.gz'

        if union_plaque is not None:
            plaque_nii = nib.Nifti1Image(union_plaque, host_ct.affine)
            nib.save(plaque_nii, f'{store_path}/union_plaque.nii.gz')
            self.final_path['plaque'] = f'{store_path}/union_plaque.nii.gz'
        self.is_saved = True
    def __repr__(self):
        ctxt = f'{"="*30}\n'
        ctxt = f'{ctxt}Patient ID   :{self.pid}\n'
        ctxt = f'{ctxt}Series NUM   :{self.snum}\n'
        ctxt = f'{ctxt}UID          :{self.uid}\n'
        ctxt = f'{ctxt}Cardiac Phase:{self.cp:4}\n'
        ctxt = f'{ctxt}Comment      :{self.comment}\n'
        ctxt = f'{ctxt}Has Plaque   :{self.plaque_num}\n'

        if len(self.final_path) > 0:
            for key, value in self.final_path.items():
                ctxt = f'{ctxt}{key:13}:{value}\n'

        return ctxt
    def get_store_path(self):
        _final_path = self.final_path.copy()
        for key, value in _final_path.items():
            _final_path[key] = value.replace(self.args.dst_root, '')
        return _final_path


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


class DeDuplicateCT:
    uid: str
    pid: str
    snum: int
    cp: CardiacPhase

    def __init__(self, init_ct: 'CTContainer'):
        self.candidate: list['CTContainer'] = []
        self.candidate.append(init_ct)
        self.uid = init_ct.uid
        self.pid = init_ct.pid
        self.snum = init_ct.snum
        self.cp = init_ct.cp

        self.append = self.candidate.append
        self.index = self.candidate.index

    def __getitem__(self, item):
        return self.candidate[item]

    def __eq__(self, ct: 'CTContainer'):
        return ct in self.candidate

    def _largest_ct(self):
        larger_idx = -1
        volume = 0
        for idx, ct in enumerate(self.candidate):
            if (cand_v := np.prod(ct.shape)) > volume:
                larger_idx = idx
                volume = cand_v

        for idx, ct in enumerate(self.candidate):
            if idx != larger_idx:
                self.candidate[idx].clean_buf()

        return self.candidate[larger_idx]

    def __call__(self, isp: 'ISPContainer'):
        final_ct: 'CTContainer' = None

        if isp is None:
            return self._largest_ct()

        if len(self.candidate) == 1:
            return self.candidate[0]

        for idx, ct in enumerate(self.candidate):
            if ct.shape == isp.shape:
                final_ct = ct
                self.candidate.remove(ct)
                break
            else:
                ct.clean_buf()

        if final_ct is None:
            return self._largest_ct()

        return final_ct


class CTContainer:
    def __init__(
            self,
            dicom_list: list[tuple[pyd.FileDataset, str]],
            pid: str,
            uid: str,
            snum: int,
            cp: int | float,
            fix_tag0008_0030: bool,
            args: argparse.Namespace,
            verbose: int = 1, buf_dir='./buf', output_dir='./out'
    ) -> None:
        """
            A CTContainer object is used to manage all dicom file with same series uid and same series number(if possible),
            and same cardiac phase(if possible)
        :param dicom_list:
        :param uid:  Series Instance UID
        :param snum: Series Instance Number
        :param cp: Cardiac Phase
        :param fix_tag0008_0030:
        :param verbose:
        :param buf_dir:
        :param output_dir:
        """
        self.verbose: int = verbose
        self.pid: str = pid
        self.has_pair: bool = False
        self.fix_tag: bool = fix_tag0008_0030
        self.snum: int = snum
        self.uid: str = uid
        self.cp: CardiacPhase = cp
        self.buf_dir: str = buf_dir
        cand_cnt = 0
        self.buf_path: str = f'{buf_dir}/{pid}/{snum}/{uid}/{cp}/cand_0'
        self.ct_output_path: str = f'{output_dir}/{pid}/{snum}/{uid}/{cp}/cand_0'
        self.args = args

        # Store all candidate
        while os.path.exists(self.buf_path):
            self.buf_path = self.buf_path.replace(f"cand_{cand_cnt}", f"cand_{cand_cnt + 1}")
            cand_cnt += 1
        self.ct_output_path: str = self.ct_output_path.replace(f"cand_{cand_cnt - 1}", f"cand_{cand_cnt}")

        os.makedirs(self.buf_path, exist_ok=True)
        os.makedirs(self.ct_output_path, exist_ok=True)

        if fix_tag0008_0030:
            copy_func = fix_copy
        else:
            copy_func = shutil_copy
        # Copy all needed dicom into <buf_dir>/<uid>/cp/cand_<cand_cnt>
        for pack in dicom_list:
            dicom_name = re.split('[/\\\]', pack[1])[-1]
            copy_func(pack, f'{self.buf_path}/{dicom_name}')
        self.shape = (*pack[0].pixel_array.shape[:2], len(dicom_list))
        if verbose == 1:
            print(f'{"=" * 30}')
            print(f'Patient ID   :{self.pid}')
            print(f'Series Num   :{self.snum}')
            print(f'UID          :{self.uid}')
            print(f'Cardiac Phase:{self.cp:4}')
            print(f'Erasing Tag  :{self.fix_tag}')

    def clean_buf(self):
        shutil.rmtree(self.buf_path)
        del self.buf_path

    def __eq__(self, other) -> bool:
        if isinstance(other, ISPContainer):
            if self.verbose:
                print("=" * 30)
                print(f'Compare    | CT:{self.final_path} vs ISP: {other.folder_name}')
                print(f'Compare UID({self.uid == other.uid})| CT:{self.uid} vs ISP: {other.uid}')
                print(f'Compare NUM({self.snum == other.snum})| CT:{self.snum} vs ISP: {other.snum}')
                print(f'Compare CP({self.cp == other.cp}) | CT:{self.cp} vs ISP: {other.cp}')
            cp2 = (float(self.cp) * float(other.cp)) ** .5
            same_cp = (cp2 == self.cp) and (cp2 == other.cp)
            snum2 = (int(self.snum) * int(other.snum)) ** .5
            same_snum = (self.snum == snum2) and (other.snum == snum2)
            same_uid = self.uid == other.uid
            is_equal = False


            # Strict equally
            if snum2 != 0:
                is_equal = same_snum
            is_equal = is_equal or same_uid

            if cp2 != 0:
                is_equal = is_equal and same_cp
            return is_equal
        if isinstance(other, CTContainer):
            return self.ct_output_path == other.ct_output_path

    def store(self):
        """
            Using CommandLine to convert the dicom series into a .nii.gz format file.
        :return:
        """
        # Using dcm2niix to convert dicom file into nii.gz file.
        commandline(self.ct_output_path, self.buf_path, self.verbose, dcm2niix_path=args.dcm2niix)
        self.final_path = list(filter(lambda x: not x.endswith('.json'), os.listdir(self.ct_output_path)))[0]
        self.nifit_ct = nib.load(f'{self.ct_output_path}/{self.final_path}')
        self.final_path = f'{self.ct_output_path}/{self.final_path}'
        # self.pid = pid
        if self.verbose == 1:
            print(f'Final path   :{self.final_path}')
            print(f'CT Shape     :{self.nifit_ct.get_fdata().shape}')
        self.clean_buf()  # After all process, clean the buffer space.

    def __repr__(self):
        ctxt = ''
        ctxt = f'{ctxt}Patient ID   :{self.pid}\n'
        ctxt = f'{ctxt}Series Num   :{self.snum}\n'
        ctxt = f'{ctxt}UID          :{self.uid}\n'
        ctxt = f'{ctxt}Cardiac Phase:{self.cp:4}\n'
        ctxt = f'{ctxt}Final Path   :{self.final_path}\n'
        return ctxt

    def get_store_path(self):
        _final_path = self.ct_output_path
        _final_path = _final_path.replace(self.args.dst_root, '')
        return _final_path


def process_isp(isp_root: str, pid: str, args: argparse.Namespace, output_dir: str = './mask') -> list[ISPContainer]:
    isp_list = []
    if args.large_ct:
        _, pid = pid.split('/')
        # isp_root = f'{isp_root}/{folder}'

    print(f'Walk under {isp_root}/{pid}, is exist?{os.path.exists("{isp_root}/{pid}")}')
    for root, dirs, files in os.walk(f'{isp_root}/{pid}', topdown=True):
        # The ISP file name format only end with .dcm
        legal_isp = list(filter(lambda x: x.endswith('.dcm'), [f'{root}/{name}' for name in files]))
        print(f'Before filtering: {len(files)} after filtering: {len(legal_isp)} at {root}')
        if len(legal_isp) < 1 or len(dirs) > 0:
            continue
        folder_name = re.split('[/\\\]', root)[-1]
        isp_list.append(ISPContainer(legal_isp, pid, folder_name, output_dir, args=args))
    return isp_list


def process_ct(
        ct_root: str, pid: str, args,
        output_dir: str, buf_dir: str
) -> tuple[list[DeDuplicateCT], list[Any]]:
    pid_ct_list: [DeDuplicateCT] = []
    error_ct_list: [str] = []

    if args.large_ct:
        folder, pid = pid.split('/')
        ct_root = f'{ct_root}/{folder}'

    for root, dirs, files in os.walk(f'{ct_root}/{pid}', topdown=True):
        legal_dcm = list(filter(lambda x: legal_dcm_path(x), [f'{root}/{name}' for name in files]))
        if len(legal_dcm) < 10 or len(dirs) > 0:
            continue
        cp2uid_map: dict[CardiacPhase, list[str]] = dict()
        cp2dcm_map: dict[CardiacPhase, list[tuple[pyd.FileDataset, str]]] = dict()
        cp2time_map: dict[CardiacPhase, list[str]] = dict()
        cp2snum_map: dict[CardiacPhase, int] = dict()

        # Do single folder at there.
        # Loading all of dcm file under current folder.
        for dcm, dcm_path in zip(list(map(partial(pyd.dcmread, force=True), legal_dcm)), legal_dcm):
            if len(dcm) == 0:
                error_ct_list.append(dcm_path)
                continue
            # status, cp = get_tag(dcm, (0x0020, 0x9241), 0)
            cp: float | int = _get_cp(dcm)
            status, uid = get_tag(dcm, (0x0020, 0x000e))
            status, stime = get_tag(dcm, (0x0008, 0x0030))
            status, snum = get_tag(dcm, (0x0020, 0x0011), None)
            if cp2uid_map.get(cp) is None: # Initial The first cases.
                cp2uid_map[cp] = list()
                cp2dcm_map[cp] = list()
                cp2time_map[cp] = list()
            cp2uid_map[cp].append(uid)
            cp2dcm_map[cp].append((dcm, dcm_path))
            cp2time_map[cp].append(stime)
            cp2snum_map[cp] = snum
        # Done.

        # Declare the CTContainer object into DeDuplicateCT
        for cardiac_phase in cp2dcm_map:
            corresponding_dcm: list[tuple[pyd.FileDataset, str]] = cp2dcm_map[cardiac_phase]
            uid: str = set(cp2uid_map[cardiac_phase]).pop()
            fix_stime: bool = len(set(cp2time_map[cardiac_phase])) > 1

            buffer_ct = CTContainer(
                corresponding_dcm, pid, uid, args=args, cp=cardiac_phase, snum=cp2snum_map[cardiac_phase],
                fix_tag0008_0030=fix_stime, output_dir=output_dir, buf_dir=buf_dir, verbose=0
            )
            if buffer_ct in pid_ct_list: # This value is True mean, it may duplicate ct-series happen
                pid_ct_list[pid_ct_list.index(buffer_ct)].append(buffer_ct)
            else: # Here is the first-time
                pid_ct_list.append(DeDuplicateCT(buffer_ct))

    print(f'CT-stage Done.')
    return pid_ct_list, error_ct_list


def single_main(pid: str, args: argparse.Namespace, ct_path_args=None, isp_path_args=None) -> list:
    if ct_path_args is None:
        ct_path_args = dict(output_dir=args.out_dir, buf_dir=args.buf_dir)
    if isp_path_args is None:
        isp_path_args = dict(output_dir=args.mask_dir)

    # ct_root = r'F:\CCTA'
    # isp_root = r'F:\CCTA Result'
    ct_root = args.data_root
    isp_root = args.isp_root
    # print(f'Root of CT: {ct_root}, Root of ISP: {isp_root}')
    # Start Loading all CT dicom file and ISP dicom file into program.
    ct_pack = process_ct(ct_root, pid, args=args, **ct_path_args)
    ct_list: list[DeDuplicateCT | CTContainer] = ct_pack[0]
    ct_error_list: list[str] = ct_pack[1]
    isp_list: list[ISPContainer] = process_isp(isp_root, pid, args=args, **isp_path_args)
    # Loading Done.

    print(f'Size of CT list : {len(ct_list)}')
    print(f'Size of ISP list: {len(isp_list)}')
    pair_list = list()
    raw_pair_list = list()
    offal_isp = []

    # Using to match all of isp to correctly CT series
    while len(isp_list) > 0:
        isp = isp_list.pop(0)

        if isp not in ct_list:  # The isp cannot match any CT
            # TODO: How do I reuse the un-match ISP?
            offal_isp.append(isp)
            continue

        # The isp maybe can match to multiple CT
        for idx, duplicate_ct in enumerate(ct_list):
            if duplicate_ct != isp:
                continue

            if isinstance(duplicate_ct, DeDuplicateCT):
                final_ct = duplicate_ct(isp)
                final_ct.store()
                ct_list[idx] = final_ct
                final_ct.has_pair = True
                duplicate_ct = final_ct

            isp.store_mask(duplicate_ct)
            raw_pair_list.append((duplicate_ct, isp))
            pair = {
                # 'image': duplicate_ct.final_path,
                'image': duplicate_ct.get_store_path(),
                'cp': duplicate_ct.cp,
                'pid': pid,
                'uid': duplicate_ct.uid
            }
            # pair.update(isp.final_path)
            pair.update(isp.get_store_path())
            pair_list.append(pair)
    else:
        # All of here is unpair CT, even that, there are good self-training data.
        for idx, dup_ct in enumerate(ct_list):
            if isinstance(dup_ct, DeDuplicateCT):
                ct = dup_ct(None)
                ct.store()
                ct_list[idx] = ct
            # End of isinstance judge
        # End of iterative unpair CT
    # End of processing unpair CT

    offal_ct = list(filter(lambda _ct: not _ct.has_pair, ct_list))
    # print(f'unpair ISP: {len(offal_isp)}')
    # print(f'unpair CT : {len(offal_ct)}')
    # print(f'Successful Pairs: {len(pair_list)}')
    # breakpoint()
    if len(offal_isp) > 0 or len(offal_ct) > 0:
        unpair_path = rf'{args.meta_dir}/unpair/{pid}'
        os.makedirs(unpair_path, exist_ok=True)
        unpair_obj = dict(isp=[], ct=[])
        for oisp in offal_isp:
            unpair_obj['isp'].append(oisp.folder_name)
        for o_ct in offal_ct:
            # Here is old method.
            # unpair_obj['ct'].append(o_ct.final_path)
            unpair_obj['ct'].append(o_ct.get_store_path())
        with open(f'{unpair_path}/unpair.json', 'w+') as jout:
            json.dump(unpair_obj, jout)
        # pass
    # End of store unpair ct and isp.
    return pair_list, ct_error_list


def full_pid(partition: Partition) -> list:
    proc_id = partition.PID
    pid_list = partition.patient_list
    args: argparse.Namespace = partition.args
    n_pid = len(pid_list)
    results = []

    for pidx, pid in enumerate(pid_list):
        t0 = dt.datetime.now()
        print(f'Process-{proc_id:02}|[Start]|[{pidx}/{n_pid}]|{pid}, time:{t0:%Y-%m-%d %H:%M:%S}')
        try:
            pid_result, ct_error_list = single_main(pid, args)
            results.extend(pid_result)

            if args.large_ct:
                folder, pid = pid.split('/')
                meta_dir = f'{args.meta_dir}/{folder}'
                err_dir = f'{args.err_dir}/{folder}'
                os.makedirs(meta_dir, exist_ok=True)
                os.makedirs(err_dir, exist_ok=True)
            else:
                meta_dir = args.meta_dir
                err_dir = args.err_dir

            with open(rf'{meta_dir}/{pid}.json', 'w+') as jout:
                json.dump(pid_result, jout)
            with open(rf'{err_dir}/{pid}.txt', 'a+') as fout:
                for err_file in ct_error_list:
                    fout.write(f'[{t0:%Y-%m-%d %H:%M:%S}]|{err_file}\n')
            suffix = 'Done'

            # with open()
        except Exception as e:
            # traceback.
            if args.large_ct:
                folder, _pid = pid.split('/')
                err_dir = f'{args.err_dir}/{folder}'
                os.makedirs(f'{err_dir}', exist_ok=True)
            else:
                err_dir = args.err_dir
                _pid = pid

            with open(rf'{err_dir}/{_pid}.txt', 'a+') as fout:
                fout.write(f'{e.args}\n')
                fout.write(traceback.format_exc())
            suffix = 'Error'

        tn = dt.datetime.now()
        print(f'Process-{proc_id:02}|[{suffix:^5}]|[{pidx}/{n_pid}]|{pid}, time:{tn:%Y-%m-%d %H:%M:%S}, cost:{tn - t0}')
    return results


def test_main():
    sample_pair = dict(name=DIGIT2LABEL_NAME, data=[])
    legal_file_patint = list(filter(lambda x: os.path.isdir(rf'F:\CCTA\{x}'), os.listdir(r'F:\CCTA')))
    world = ['0001', '2098', '0003', '0060', *np.random.choice(list(set(legal_file_patint) - {'2098', '0003', '0060'}), size=7, replace=False)]
    print(world)
    for pid in world:
        supervised_list = single_main(pid)
        sample_pair['data'].extend(supervised_list)
        print(json.dumps(sample_pair, indent=4))
    # import json
    # with open('./static/sample_pair.json', 'w+') as jout:
    #     json.dump(sample_pair, jout)


def processed_data_list(args: argparse.Namespace) -> list:
    if (ip := args.ignore_path) is not None:
        if ip == 'no':
            return list()
        with open(ip, 'r') as jin:
            pdata = json.load(jin)
        pdata = [ctxt.split('.')[0] for ctxt in pdata]
        return pdata
    meta_dir = args.meta_dir
    pdata = [name.split('.')[0] for name in os.listdir(meta_dir) if name.endswith('.json')]
    return pdata


def unzip(args, folder, member) -> str | None:
    """
        Return None if got error, otherwise, the return will be the patient id
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
        with open(f'{args.err_dir}/Unzip_error.txt', 'a+') as fout:
            fout.write(f'{"=" * 30}\n{args.data_root}/{folder}/{member}\n')
            fout.write(traceback.format_exc())
            fout.write('\n')
        return None

    return patient_name


def get_legal_pair(ignore_list: list[str], args: argparse.Namespace) -> list[str]:
    if not args.large_ct: # Process 502-CT ignore list
        print(f'Is 502CT')
        ALL_MEMBER = os.listdir(args.data_root)
        print(f'Size of member: {len(ALL_MEMBER)}')
        return list(
            filter(lambda x: os.path.isdir(rf'{args.data_root}/{x}') and x not in ignore_list,
                   os.listdir(args.data_root)))
    # Process for Large CT(around 2500 patient)
    print(f'Trying to processing Large CT')
    legal_path: list[str] = list()
    candidates_folder: list[str] = os.listdir(args.data_root)

    # There are a lot of useless folder and annotation folder under `args.data_root`
    for idx, folder in enumerate(candidates_folder):
        all_member: list[str] = os.listdir(f'{args.data_root}/{folder}')
        # Because there have a lot of unzip patient under some folder, this line choosing already unzip patient
        folder_member: list[str] = list(filter(lambda x: re.fullmatch('[0-9]{4}', x) is not None, all_member))
        # This line choosing zipped patient
        zip_member: list[str] = list(filter(IS_ZIP, all_member))
        # Second times detect
        zip_member: list[str] = list(filter(lambda x: x.split('.')[0] not in folder_member, zip_member))
        # An legal folder must store patient not other things.
        # If `folder_member` and `zip_member` doesn't contain any member, just ignore current folder
        is_legal_folder: bool = len(folder_member) == 0 and len(zip_member) == 0
        is_annotation_folder = 'ct_isp' in folder
        if is_legal_folder or is_annotation_folder:
            continue  # not a legal folder
        print(f'# of zipped patient: {len(zip_member)} in {folder}')
        print(f'# of unzip patient: {len(folder_member)} in {folder}')
        folder_member = [f'{folder}/{pname}' for pname in folder_member]
        if len(zip_member) > 0:
            for member in zip_member:
                # The method `unzip` would unzip the zip_member and return the corresponding
                # patient name, if got unzip error, it would return `None`
                if (patient_name := unzip(args, folder, member)) is not None:
                    # Adding the `patient_name` into `folder_member`
                    folder_member.append(f'{folder}/{patient_name}')
                    print(f'Unzip: {patient_name}')
                # End of unzip
            # End of iterative unzip all of zip_member
        # End of *.zip file process.
        legal_path.extend(folder_member)
    # End of all possible.
    return legal_path


def start_main(args: argparse.Namespace):
    if (w_ratio := args.worker_ratio) is None:
        nproc: int = args.num_workers
    else:
        nproc: int = mp.cpu_count() // w_ratio
    args.nproc = nproc
    print(f'# of workers: {nproc}')
    sample_pair = dict(name=DIGIT2LABEL_NAME, data=[])
    ignore_list: list[str] = processed_data_list(args)
    legal_file_patient = get_legal_pair(ignore_list, args)
    print(f'The number of patients waiting to be processed: {len(legal_file_patient)}')
    sub_world = np.array_split(legal_file_patient, nproc)
    sub_world = [Partition(PID=i, patient_list=sworld, args=args) for i, sworld in enumerate(sub_world)]

    with mp.Pool(processes=nproc) as pool:
        sample_pair['data'].extend(pool.map(full_pid, sub_world))

    with open(rf'{args.meta_dir}/sample.json', 'w+') as jout:
        json.dump(sample_pair, jout)


def mask_sure_folder_exist(args: argparse.Namespace):
    os.makedirs(f'{args.meta_dir}', exist_ok=True)
    os.makedirs(f'{args.out_dir}', exist_ok=True)
    os.makedirs(f'{args.buf_dir}', exist_ok=True)
    os.makedirs(f'{args.mask_dir}', exist_ok=True)
    os.makedirs(f'{args.err_dir}', exist_ok=True)


if __name__ == '__main__':
    # from argparse import ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--isp_root', type=str)
    parser.add_argument('--large_ct', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--worker_ratio', type=float, default=None)
    parser.add_argument('--ignore_path', type=str, default=None)
    parser.add_argument('--out_dir', default='./NiiTaipei/out')
    parser.add_argument('--buf_dir', default='./NiiTaipei/buf')
    parser.add_argument('--mask_dir', default='./NiiTaipei/mask')
    parser.add_argument('--meta_dir', default='./NiiTaipei/meta')
    parser.add_argument('--err_dir', default='./NiiTaipei/err')
    parser.add_argument('--dst_root', default='./NiiTaipei')
    parser.add_argument('--dcm2niix', default='./lib/dcm2niix.exe')
    args = parser.parse_args()
    mask_sure_folder_exist(args)
    start_main(args)
