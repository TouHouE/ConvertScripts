import re
import os
import shutil
import argparse
import traceback
from typing import NewType, Union, Callable
import datetime as dt

import numpy as np
import pydicom as pyd
import nibabel as nib

import isp_helper as ISPH
from constant import LABEL_NAME2DIGIT
from utils import convert_utils as CUtils
from utils.hooker import obj_hooker
from utils.data_typing import CardiacPhase, FilePathPack
from models.CommonDataModels import DebugCard

__all__ = [
    'TaipeiCTDeduplicator', 'TaipeiCTHandler', 'TaipeiISPHandler', "TaipeiFactory"
]


class TaipeiISPHandler(DebugCard):
    shape: tuple[int, int, int] | list[int, int, int]
    final_path: dict[str, str]
    plaque_num: int
    tissue_list: list[pyd.FileDataset]
    path_list: list[pyd.FileDataset]
    is_saved: bool

    @obj_hooker
    def __init__(
            self, isp_list: list[str], pid: str,
            folder_name: str, output_dir: str = './mask',
            args: argparse.Namespace = None, verbose: int = 0
    ) -> None:
        """
        A `TaipeiISPHandler` object is used to package an isp annotation folder, each folder has multiple "Findings"
        Args:
            isp_list: Only contains a sequence of isp annotation dicom file's path
            pid: Patient ID
            folder_name: This parameter is used to easily identify the original path
            output_dir: A `TaipeiISPHandler` object goal is to save the tissue mask, and plaque mask into nifit file,
            the mask will be saved in :param output_dir
            verbose: if is 1, then print all process detail, if is 0, don't print anything.
        """
        self.output_dir: str = output_dir
        self.total_list: list[str] = isp_list
        self.folder_name: str = folder_name
        self.pid: str = pid
        self.verbose: int = args.verbose
        self.args = args
        self.debug_attr_name = ['output_dir', 'total_list', 'folder_name', 'pid', 'desc', 'cp', 'snum', 'uid']

        self.tissue_list = []
        self.path_list = []
        self.plaque_num = 0
        self.final_path = dict()
        self.is_saved = False
        isp, uni_uid = self._init_tissu_and_plaque_list()
        assert len(self.tissue_list) > 0, 'No tissue list'
        desc: str = CUtils.find_desc(isp)
        self.comment = desc

        cp, snum = self._analysis_comment(desc)
        self.snum = snum
        self.cp = cp

        # This for debug usage.
        if len(uni_uid) > 1:
            print(uni_uid)
        self.uid = uni_uid.pop()

    @obj_hooker
    def _analysis_comment(self, desc) -> tuple[CardiacPhase, int]:
        cp = 0
        snum = 0
        if self.verbose == 1:
            print(f'{"Comment":13}: {self.comment}')

        if len(desc_pack := re.split('[, _]', desc)) > 0:
            if self.verbose == 1:
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
        return cp, snum

    def _init_tissu_and_plaque_list(self):
        isp_list = self.total_list
        uni_uid = set()

        for isp, isp_path in zip(map(pyd.dcmread, isp_list), isp_list):
            uni_uid.add(CUtils.find_isp_uid(isp))

            if (_shape := CUtils.get_tag(isp, (0x07a1, 0x1007), need_state=False)) is not None:
                self.shape = _shape

            if isp.ImageType[-1] != 'PATH':
                self.tissue_list.append(isp)
                continue

            self.path_list.append(isp)
            tamar_list: pyd.DataElement | None = isp.get((0x07a1, 0x1050))
            if tamar_list is not None:
                if not isinstance(tamar_list.value, list):
                    self.plaque_num += len(list(filter(lambda x: ISPH.legal_plaque(x), tamar_list.value)))
        return isp, uni_uid

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

    @obj_hooker
    def store_mask(self, ct: 'TaipeiCTHandler'):
        if self.is_saved:
            return
        host_ct = ct.nifit_ct
        store_path = os.path.join(self.output_dir, self.pid, self.uid, f'{ct.cp}', self.folder_name)
        # store_path = f'{self.output_dir}/{self.pid}/{self.uid}/{ct.cp}/{self.folder_name}'
        os.makedirs(store_path, exist_ok=True)
        union_mask: np.ndarray = self._collect_tissue()

        try:
            union_mask, union_plaque = self._collect_plaque(union_mask, host_ct)
        except Exception as e:
            perr = os.path.join(self.args.err_dir, 'plaque_error.txt')
            with open(perr, 'a+') as fout:
                # fout.write(f'{"=" * 30}\n')
                fout.write(f"{str(self)}Got error during collect_plaque\n")
                fout.write(traceback.format_exc())
            union_plaque = None

        mask_nii = nib.Nifti1Image(union_mask, host_ct.affine)
        mask_path = os.path.join(store_path, 'union_mask.nii.gz')
        nib.save(mask_nii, mask_path)
        self.final_path['mask'] = mask_path

        if union_plaque is not None:
            plaque_nii = nib.Nifti1Image(union_plaque, host_ct.affine)
            plq_path = os.path.join(store_path, 'union_plaque.nii.gz')
            nib.save(plaque_nii, plq_path)
            self.final_path['plaque'] = plq_path
        self.is_saved = True

    def get_store_path(self):
        _nii_storage_path = self.final_path.copy()
        for key, value in _nii_storage_path.items():
            tmp = value.replace(self.args.dst_root, '')
            if tmp[0] in ['/', r'\\']:
                tmp = tmp[1:]
            _nii_storage_path[key] = tmp
        return _nii_storage_path

    def __repr__(self):
        ctxt = f'{"=" * 30}\n'
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


class TaipeiCTHandler(DebugCard):
    @obj_hooker
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
        :param fix_tag0008_0030: To correct different date in same series .dcm file.
        :param verbose:
        :param buf_dir:
        :param output_dir:
        """
        self.verbose: int = args.verbose
        self.pid: str = pid
        # self.has_pair: bool = False
        self.fix_tag: bool = fix_tag0008_0030
        self.snum: int = snum
        self.uid: str = uid
        self.cp: CardiacPhase = cp
        self.buf_dir: str = buf_dir
        # self.buf_path: str = f'{buf_dir}/{pid}/{snum}/{uid}/{cp}/cand_0'
        self.buf_path: str = os.path.join(buf_dir, f'{pid}', f'{snum}', f'{uid}', f'{cp}', 'cand_0')
        # self.ct_output_path: str = f'{output_dir}/{pid}/{snum}/{uid}/{cp}/cand_0'
        self.ct_output_path: str = os.path.join(output_dir, f'{pid}', f'{snum}', uid, f'{cp}', 'cand_0')
        self.nii_gz_file_name: str
        self.args = args
        self.len_dcm = len(dicom_list)
        self.debug_attr_name = ['pid', 'snum', 'uid', 'cp', 'buf_dir', 'buf_path', 'ct_output_path', 'fix_tag0008_0030', 'len_dcm']
        self.paired_isp: [TaipeiISPHandler] = list()
        # Store all candidate.
        # The function of remove duplicate dicom file is implement at DedupDCM
        self._make_needed_folder()
        pack = self._copy_dcm_into_buffer(dicom_list, CUtils.fix_copy if fix_tag0008_0030 else CUtils.shutil_copy)
        self.shape = (*pack[0].pixel_array.shape[:2], len(dicom_list))

        if verbose == 1:
            print(f'{"=" * 30}')
            print(f'Patient ID   :{self.pid}')
            print(f'Series Num   :{self.snum}')
            print(f'UID          :{self.uid}')
            print(f'Cardiac Phase:{self.cp:4}')
            print(f'Erasing Tag  :{self.fix_tag}')

    def _copy_dcm_into_buffer(
            self, dicom_list: list[tuple[pyd.FileDataset, str]], copy_func: Callable
    ) -> tuple[pyd.FileDataset, str]:
        for pack in dicom_list:
            dicom_name = re.split('[/\\\]', pack[1])[-1]
            dst_path = os.path.join(self.buf_path, dicom_name)
            copy_func(pack, dst_path)
            if self.verbose == 1:
                print(f'Copying {pack[1]} -> {dst_path}')
        return pack

    def _make_needed_folder(self):
        cand_cnt = 0
        while os.path.exists(self.buf_path):
            self.buf_path = self.buf_path.replace(f"cand_{cand_cnt}", f"cand_{cand_cnt + 1}")
            cand_cnt += 1
        self.ct_output_path: str = self.ct_output_path.replace(f"cand_{cand_cnt - 1}", f"cand_{cand_cnt}")

        os.makedirs(self.buf_path, exist_ok=True)
        os.makedirs(self.ct_output_path, exist_ok=True)
        if self.verbose == 1:
            print(f'buf_path: {self.buf_path}')
            print(f'ct_output_path: {self.ct_output_path}')

    def _remake_needed_folder(self, remake_path: str) -> None:
        os.makedirs(remake_path, exist_ok=True)

    def clean_buf(self):
        shutil.rmtree(self.buf_path)
        del self.buf_path

    def delete_self(self):
        shutil.rmtree(self.ct_output_path)

    @obj_hooker
    def store(self):
        """
            Using CommandLine to convert the dicom series into a .nii.gz format file.
        :return:
        """
        # Using dcm2niix to convert dicom file into nii.gz file.
        if self.snum == 2 and self.uid == "1.2.840.113654.2.70.1.299644915472865846444756882987166668049" and self.cp == 75.:
            # breakpoint()
            pass

        CUtils.commandline(self.ct_output_path, self.buf_path, self.verbose, dcm2niix_path=self.args.dcm2niix, gargs=self.args)
        # Got .nii.gz file name
        self.final_path = list(filter(lambda x: not x.endswith('.json'), os.listdir(self.ct_output_path)))[0]
        # Turn it into abs path
        self.final_path = os.path.join(self.ct_output_path, self.final_path)
        self.nifit_ct = nib.load(self.final_path)
        # self.final_path = f'{self.ct_output_path}/{self.final_path}'
        # self.pid = pid
        if self.verbose == 1:
            print(f'Final path   :{self.final_path}')
            print(f'CT Shape     :{self.nifit_ct.get_fdata().shape}')
        self.clean_buf()  # After all process, clean the buffer space.

    def get_store_path(self) -> str:
        _final_path: str = self.final_path
        _final_path: str = _final_path.replace(self.args.dst_root, '')
        if _final_path[0] in ['/', r'\\']:
            _final_path = _final_path[1:]
        return _final_path

    def __eq__(self, other) -> bool:

        if isinstance(other, TaipeiISPHandler):
            if self.verbose:
                print("=" * 30)
                print(f'Compare    | CT:{getattr(self, "final_path", None)} vs ISP: {other.folder_name}')
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
        if isinstance(other, TaipeiCTHandler):
            return self.ct_output_path == other.ct_output_path

    def __repr__(self):
        ctxt = ''
        ctxt = f'{ctxt}Patient ID   :{self.pid}\n'
        ctxt = f'{ctxt}Series Num   :{self.snum}\n'
        ctxt = f'{ctxt}UID          :{self.uid}\n'
        ctxt = f'{ctxt}Cardiac Phase:{self.cp:4}\n'
        ctxt = f'{ctxt}Final Path   :{self.final_path}\n'
        return ctxt

    def num_paired_isp(self) -> int:
        return len(self.paired_isp)

    @property
    def has_pair(self) -> bool:
        return len(self.paired_isp) > 0

class TaipeiCTDeduplicator(object):
    uid: str
    pid: str
    snum: int
    cp: CardiacPhase

    def __init__(self, init_ct: TaipeiCTHandler):
        self.candidate: list[TaipeiCTHandler] = []
        self.candidate.append(init_ct)
        self.uid = init_ct.uid
        self.pid = init_ct.pid
        self.snum = init_ct.snum
        self.cp = init_ct.cp

        self.append = self.candidate.append
        self.index = self.candidate.index

    def __getitem__(self, item):
        return self.candidate[item]

    def __eq__(self, ct: Union[TaipeiCTHandler, TaipeiISPHandler, 'TaipeiCTDeduplicator']):
        return ct in self.candidate

    def _largest_ct(self):
        """
            If matching isp annotation doesn't exist, just choose the largest slice `TaipeiCTHandler` object.
        """
        largest_idx = -1
        volume = 0
        elect_ct: TaipeiCTHandler

        # Find out the most possible TaipeiCTHandler by who have the largest volume.
        for idx, ct in enumerate(self.candidate):
            if (cand_v := np.prod(ct.shape)) > volume:
                largest_idx = idx
                volume = cand_v
        else:  # The elected TaipeiCTHandler need remove from candidate list.
            elect_ct = self.candidate.pop(largest_idx)
            # self.candidate.remove(elect_ct)
            pass

        self._clean_reject()
        return elect_ct

    def _clean_reject(self):
        # This statement aim to clean all .dcm file under buffer and
        # the .nii.gz file stored from rejected TaipeiCTHandler
        for ct in self.candidate:
            ct.clean_buf()
            ct.delete_self()

    def __call__(self, isp: TaipeiISPHandler):
        elect_ct: TaipeiCTHandler | None = None

        if isp is None:
            return self._largest_ct()
        # If only a TaipeiCTHandler in candidate, just return that one.
        if len(self.candidate) == 1:
            # No reject CT exists; we didn't need to call TaipeiCTDeduplicator._clean_reject method.
            return self.candidate[0]

        for idx, ct in enumerate(self.candidate):
            if ct.shape == isp.shape:
                elect_ct = ct
                self.candidate.remove(ct)
                break
        if elect_ct is None:
            return self._largest_ct()

        self._clean_reject()
        return elect_ct

    def num_paired_isp(self):
        return sum(raw_ct.num_paired_isp() for raw_ct in self.candidate)


class TaipeiFactory:

    @classmethod
    def create_CT(cls) -> [TaipeiCTHandler, TaipeiCTDeduplicator]:
        #@TODO implement CT part
        pass

    @classmethod
    def create_isp(cls, legal_isp_list, pid, folder_name, output_dir, args=None) -> TaipeiISPHandler:
        try:
            isp = TaipeiISPHandler(legal_isp_list, pid, folder_name, output_dir, args=args)
        except AssertionError as ase:
            return None
        return isp