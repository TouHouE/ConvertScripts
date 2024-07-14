import re
import os
import json
import argparse
import traceback
import datetime as dt
import multiprocessing as mp
from copy import deepcopy
from functools import partial
from operator import methodcaller, itemgetter
from typing import Tuple, Any, Callable, List, Dict, Iterable, Optional

import numpy as np
import nibabel as nib
import pydicom as pyd

import models
from constant import DIGIT2LABEL_NAME, LABEL_NAME2DIGIT
from utils import convert_utils as CUtils
from utils import common_utils as ComUtils
from utils.data_typing import CardiacPhase, FilePathPack, IspCtPair, PatientId

IS_ZIP: methodcaller = methodcaller('endswith', '.zip')
WRAP_ERR: itemgetter = itemgetter(1)
WRAP_DATA: itemgetter = itemgetter(0)


def collect_ct_info(legal_dcm: List[str]) -> Dict[str, Dict[CardiacPhase, Any]]:
    cp2uid_map: dict[CardiacPhase, list[str]] = dict()
    cp2dcm_map: dict[CardiacPhase, list[tuple[pyd.FileDataset, str]]] = dict()
    cp2time_map: dict[CardiacPhase, list[str]] = dict()
    cp2snum_map: dict[CardiacPhase, int] = dict()

    # Do single folder at there.
    # Loading all of dcm file under current folder.
    for dcm, dcm_path in zip(list(map(partial(pyd.dcmread, force=True), legal_dcm)), legal_dcm):
        cp: float | int = CUtils.find_cp(dcm)
        _, uid = CUtils.get_tag(dcm, (0x0020, 0x000e))
        _, stime = CUtils.get_tag(dcm, (0x0008, 0x0030))
        _, snum = CUtils.get_tag(dcm, (0x0020, 0x0011), None)

        if cp2uid_map.get(cp) is None:  # Initial The first cases.
            cp2uid_map[cp] = list()
            cp2dcm_map[cp] = list()
            cp2time_map[cp] = list()
        cp2uid_map[cp].append(uid)
        cp2dcm_map[cp].append((dcm, dcm_path))
        cp2time_map[cp].append(stime)
        cp2snum_map[cp] = snum
    return {
        'uid': cp2uid_map,
        'dcm': cp2dcm_map,
        'time': cp2time_map,
        'snum': cp2snum_map
    }


def build_isp_list(
        isp_root: str, pid: str, args: argparse.Namespace, output_dir: str = './mask'
) -> list[models.taipei.TaipeiISPHandler]:
    isp_list: List[models.taipei.TaipeiISPHandler] = []
    # breakpoint()
    for root, dirs, files in os.walk(f'{isp_root}/{pid}', topdown=True):
        # The ISP file name format only end with .dcm
        legal_isp_list: List[str] = CUtils.filter_legal_dcm([f'{root}/{name}' for name in files], is_ct=False)
        if len(legal_isp_list) < 1 or len(dirs) > 0:
            continue

        folder_name = re.split('[/\\\]', root)[-1]
        isp_list.append(models.taipei.TaipeiISPHandler(legal_isp_list, pid, folder_name, output_dir, args=args))
    return isp_list


def build_ct_list(
        ct_root: str, pid: str, args,
        output_dir: str, buf_dir: str
) -> tuple[list[models.taipei.TaipeiCTDeduplicator], list[Any]]:
    """
    To construct all of different .dcm ct series into `models.TaipeiDataModels.TaipeiCTHandler`
    :param ct_root: The path that store all patient folder
    :param pid: represent the id of a patient.
    :param output_dir: where to store the .nii file
    :param buf_dir: where to cache the .dcm series

    :return : A tuple with a list of success entities and a list of except entities

    """
    pid_ct_list: List[models.taipei.TaipeiCTDeduplicator] = []
    error_ct_list: List[str] = []

    for root, dirs, files in os.walk(f'{ct_root}/{pid}', topdown=True):
        legal_dcm = CUtils.filter_legal_dcm([f'{root}/{name}' for name in files])
        if len(legal_dcm) < 10 or len(dirs) > 0:
            continue

        # Do single folder at there.
        # Loading all of dcm file under current folder.
        total_cpmap = collect_ct_info(legal_dcm)
        # Depack here
        cp2uid_map: dict[CardiacPhase, list[str]] = total_cpmap['uid']
        cp2dcm_map: dict[CardiacPhase, list[tuple[pyd.FileDataset, str]]] = total_cpmap['dcm']
        cp2time_map: dict[CardiacPhase, list[str]] = total_cpmap['time']
        cp2snum_map: dict[CardiacPhase, int] = total_cpmap['snum']
        # Done.

        # Declare the models.taipei.TaipeiCTHandler object into models.taipei.TaipeiCTDeduplicator
        for cardiac_phase in cp2dcm_map:
            corresponding_dcm: list[tuple[pyd.FileDataset, str]] = cp2dcm_map[cardiac_phase]
            uid: str = set(cp2uid_map[cardiac_phase]).pop()
            fix_stime: bool = len(set(cp2time_map[cardiac_phase])) > 1

            buffer_ct = models.taipei.TaipeiCTHandler(
                corresponding_dcm, pid, uid, args=args, cp=cardiac_phase, snum=cp2snum_map[cardiac_phase],
                fix_tag0008_0030=fix_stime, output_dir=output_dir, buf_dir=buf_dir, verbose=args.verbose
            )
            # To deduplicate CT series,
            # we collect all CT series with the same attributes into 'TaipeiCTDeduplicator'
            if buffer_ct in pid_ct_list:
                # Already exist a `TaipeiCTDeduplicator` with the same attribute.
                pid_ct_list[pid_ct_list.index(buffer_ct)].append(buffer_ct)
            else:
                # To handle potential duplicate series,
                # we store a new `TaipeiCTDeduplicator` instead of raw ct series.
                pid_ct_list.append(models.taipei.TaipeiCTDeduplicator(buffer_ct))

    return pid_ct_list, error_ct_list


def build_ct_isp_pair(
        ct_list: list[models.taipei.TaipeiCTDeduplicator | models.taipei.TaipeiCTHandler],
        isp: models.taipei.TaipeiISPHandler
) -> List[IspCtPair]:
    sub_pair: List[IspCtPair] = list()
    # The `confusion_ct maybe a Deduplicator or a CTHandler, I suppose deduplicate and match to isp in same for loop.
    for idx, confusion_ct in enumerate(ct_list):
        if confusion_ct != isp:
            continue
        # The `final_ct` is the only ct after deduplicator
        final_ct: models.taipei.TaipeiCTHandler

        if isinstance(confusion_ct, models.taipei.TaipeiCTDeduplicator):
            final_ct = confusion_ct(isp)
            final_ct.store()    # Store .nii.gz ct file after deduplicate.
            final_ct.has_pair = True
            ct_list[idx] = final_ct     # Replace the deduplicator with only ct(`models.taipei.TaipeiCTHandler`)
        else:
            final_ct = confusion_ct
        # Store the mask into disk.
        isp.store_mask(final_ct)
        pair: IspCtPair = IspCtPair({
            'image': final_ct.get_store_path(),
            'cp': final_ct.cp,
            'pid': final_ct.pid,
            'uid': final_ct.uid
        })
        pair.update(isp.get_store_path())
        sub_pair.append(pair)
    return sub_pair


def patient_proc(
        pid: str, args: argparse.Namespace, ct_path_args: Optional[Dict[str, Any]] = None, isp_path_args=None
) -> Tuple[List[IspCtPair], List[str]]:
    """
    Process single patient data and return
    Args:
        pid(str): patient id
        args(argparse.Namespace): The user arguments and some print info
        ct_path_args(Dict[str, Any]): is Optional, store `output_dir`, and `buf_dir` for ```build_ct_list``` function
        isp_path_args(Dict[str, Any]): is Optional, store `output_dir` for ```build_isp_list``` function

    Returns(Tuple[List[IspCtPair], List[str]]):

    """
    if ct_path_args is None:
        ct_path_args = dict(output_dir=args.out_dir, buf_dir=args.buf_dir)
    if isp_path_args is None:
        isp_path_args = dict(output_dir=args.mask_dir)

    ct_root = args.data_root
    isp_root = args.isp_root
    # Start Loading all CT dicom file and ISP dicom file into program.
    ct_pack = build_ct_list(ct_root, pid, args=args, **ct_path_args)

    ct_list: list[models.taipei.TaipeiCTDeduplicator | models.taipei.TaipeiCTHandler] = WRAP_DATA(ct_pack)
    ct_error_list: list[str] = WRAP_ERR(ct_pack)

    isp_list: list[models.taipei.TaipeiISPHandler] = build_isp_list(isp_root, pid, args=args, **isp_path_args)
    # Loading Done.
    # Show some information.
    info = f'len ct: {len(ct_list)}, len isp: {len(isp_list)}'
    ComUtils.print_info('Loaded CT&ISP', info, args)
    # Show information done.

    pair_list: List[IspCtPair] = list()
    offal_isp = list()
    remain_isp: int = len(isp_list)

    # Using to match all of isp to correctly CT series
    while remain_isp > 0:
        isp = isp_list.pop(0)

        if isp not in ct_list:  # The isp cannot match any CT
            offal_isp.append(isp)
            remain_isp = len(isp_list)
            continue
        # The isp maybe can match to multiple CT
        sub_pair_list: List[IspCtPair] = build_ct_isp_pair(ct_list, isp)
        pair_list.extend(sub_pair_list)
        remain_isp = len(isp_list)
    else:
        # All of here is unpair CT, even that, there are good self-training data.
        for idx, dup_ct in enumerate(ct_list):
            if isinstance(dup_ct, models.taipei.TaipeiCTDeduplicator):
                ct = dup_ct(None)
                ct.store()
                ct_list[idx] = ct
            # End of isinstance judge
        # End of iterative unpair CT
    # End of processing unpair CT

    offal_ct = list(filter(lambda _ct: not _ct.has_pair, ct_list))
    if len(offal_isp) > 0 or len(offal_ct) > 0:
        CUtils.record_offal_sample(offal_isp, offal_ct, args)
        # pass
    # End of store unpair ct and isp.
    return pair_list, ct_error_list


def start_proc(partition: models.Partition) -> List[IspCtPair]:
    """
    Each process will start from this method
    Args:
        partition(models.Partition): include assign sub data sequence, process id and user arguments.

    Returns:
        A list of IspCtPair ( that is a dict
    """
    proc_id: int = partition.PID
    pid_list: List[str] | np.ndarray | Iterable = partition.patient_list
    args: argparse.Namespace = deepcopy(partition.args)
    setattr(args, 'proc_id', proc_id)
    n_pid = len(pid_list)
    results = []

    for pidx, pid in enumerate(pid_list):
        t0: dt.datetime = dt.datetime.now()
        patient_progress: str = f'{pidx}/{n_pid}'
        setattr(args, 'patient_progress', patient_progress)
        setattr(args, 't0', t0)
        setattr(args, 'pid', pid)
        ComUtils.print_info('Start', '', args)

        try:
            patient_pack: Tuple[List[IspCtPair], List[str]] = patient_proc(pid, args)
            pid_result: List[IspCtPair] = WRAP_DATA(patient_pack)
            raw_error_list: List[str] = WRAP_ERR(patient_pack)
            results.extend(pid_result)
            error_list_to_store: List[str] = [f'[{ComUtils.time2str(t0)}]|{err_file}' for err_file in raw_error_list]

            ComUtils.write_content(rf'{args.meta_dir}/{pid}.json', pid_result, as_json=True)

            if len(error_list_to_store) > 0:    # If no error don't write down any error message.
                ComUtils.write_content(rf'{args.err_dir}/{pid}.txt', error_list_to_store, cover=False, as_json=False)
            end_status: str = 'Done'
        except Exception as e:
            error_content = [e.args, traceback.format_exc()]
            ComUtils.write_content(rf'{args.err_dir}/{pid}.txt', error_content, cover=False, as_json=False)
            end_status: str = 'Error'

        tn = dt.datetime.now()
        ComUtils.print_info(
            end_status, info=dict(cost=tn - t0), args=args
        )

    return results


def unzip_proc(args, folder, member) -> str | None:
    """
        Return None if got error, otherwise, the return will be the patient id
    :param args:
    :param folder:
    :param member:
    Returns:

    """
    result = CUtils.unzip(args, folder, member)

    if isinstance(result, list):
        ComUtils.write_content(f'{args.err_dir}/Unzip_error.txt', result, cover=False, as_json=False)
        return None
    return result


def initial_ignore_list(args: argparse.Namespace) -> List[PatientId]:
    pdata: List[PatientId] = list()
    if (ip := args.ignore_path) is not None:
        if ip == 'no':
            return pdata

        with open(ip, 'r') as jin:
            ignore_files_in_json: List[PatientId] = json.load(jin)
        for ctxt in ignore_files_in_json:
            patient_id = ctxt.split('.')[0]
            pdata.append(PatientId(patient_id))
        return pdata

    meta_dir: str = args.meta_dir
    pdata = [PatientId(name.split('.')[0]) for name in os.listdir(meta_dir) if name.endswith('.json')]
    return pdata


def initial_legal_pair(ignore_list: list[str], args: argparse.Namespace) -> list[str]:
    return CUtils.filter_legal_patient_folder(args, ignore_list)


def main(args: argparse.Namespace):
    if (pf := getattr(args, 'patient_folder', None)) is not None:
        org_nw = args.num_workers
        org_wr = args.worker_ratio
        args.num_workers = 1
        args.worker_ratio = None


        print(f'Using patient_folder mode, re-setting:\n num_workers: {org_nw} -> 1\n worker_ratio: {org_wr} -> None ')

    if (w_ratio := args.worker_ratio) is None:
        nproc: int = args.num_workers
    else:
        nproc: int = mp.cpu_count() // w_ratio

    setattr(args, 'nproc', nproc)
    print(f'# of workers: {nproc}')
    sample_pair = dict(name=DIGIT2LABEL_NAME, data=[])  # Initial the final store table
    if (patient_folder := getattr(args, 'patient_folder', None)) is None:
        ignore_list: list[PatientId] = initial_ignore_list(args)
        legal_file_patient: List[str] = initial_legal_pair(ignore_list, args)
        # TODO: This line only for debugging.
        # legal_file_patient = np.random.choice(legal_file_patient, size=100, replace=False)
        print(f'The number of patients waiting to be processed: {len(legal_file_patient)}')
    else:
        patient_id = re.split(r'[/\\]', patient_folder)[-1]
        data_root = patient_id[:-len(patient_id)]
        legal_file_patient: List[str] = [patient_id]
        args.data_root = data_root

    sub_world = np.array_split(legal_file_patient, nproc)
    sub_world = [models.Partition(proc_id=i, data=sworld, args=args) for i, sworld in enumerate(sub_world)]
    if nproc == 1:  # Only 1 thread, The Pool object is useless
        sample_pair['data'].extend(start_proc(sub_world[0]))
    else:
        with mp.Pool(processes=nproc) as pool:
            all_results = pool.map(start_proc, sub_world)
            for proc_result in all_results:
                sample_pair['data'].extend(proc_result)
    ComUtils.write_content(rf'{args.meta_dir}/raw_sample.json', sample_pair, as_json=True)

    if all(needed_attr is not None for needed_attr in [getattr(args, 'test_ratio'), getattr(args, 'num_fold')]):
        develop_model_table = ComUtils.make_vista_style_table(sample_pair.copy())
        ComUtils.write_content(rf'{args.meta_dir}/vista_table.json', develop_model_table, as_json=True)
    return


def mask_sure_folder_exist(args: argparse.Namespace):
    os.makedirs(f'{args.meta_dir}', exist_ok=True)
    os.makedirs(f'{args.out_dir}', exist_ok=True)
    os.makedirs(f'{args.buf_dir}', exist_ok=True)
    os.makedirs(f'{args.mask_dir}', exist_ok=True)
    os.makedirs(f'{args.err_dir}', exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--patient_folder', type=str, required=False, help='single patient folder this is exclusive with `--data_root`')
    parser.add_argument('--isp_root', type=str)
    parser.add_argument('--large_ct', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--worker_ratio', type=float, default=None)
    parser.add_argument('--ignore_path', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='./NiiTaipei/out')
    parser.add_argument('--buf_dir', type=str, default='./NiiTaipei/buf')
    parser.add_argument('--mask_dir', type=str, default='./NiiTaipei/mask')
    parser.add_argument('--meta_dir', type=str, default='./NiiTaipei/meta')
    parser.add_argument('--err_dir', type=str, default='./NiiTaipei/err')
    parser.add_argument('--dst_root', type=str, default='./NiiTaipei')
    parser.add_argument('--dcm2niix', type=str, default='./lib/dcm2niix.exe')
    parser.add_argument('--test_ratio', type=float, required=False)
    parser.add_argument('--num_fold', type=int, required=False)
    parser.add_argument('--verbose', type=int, default=0)
    global_args = parser.parse_args()
    mask_sure_folder_exist(global_args)
    main(global_args)
