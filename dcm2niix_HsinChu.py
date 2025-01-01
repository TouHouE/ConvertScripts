"""
This Script is using to convert the raw image which come from HsinChu Branch Hospital into the deeplearning usable format
Nifit file,
"""
import re
import os
import argparse
import traceback
import datetime as dt
import multiprocessing as mp
from copy import deepcopy
from typing import List, Dict, Any, Iterable
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import numpy as np
import pandas as pd

import models
from utils import convert_utils as CUtils
from utils import common_utils as ComUtils


def find_legal_dcm(dcm_name) -> bool:
    not_dup = re.fullmatch('.*\[[0-9]{1,}\]\.dcm', dcm_name) is None
    return not_dup


def dcm_collector2nifti_handler(
        dcm_collector_list: List[models.hsinchu.DicomCollector],
        buf_dir: str, out_dir: str, err_dir: str, dcm2niix: str
) -> list[models.hsinchu.NiftiHandler]:
    """
        The convert diagram is dcm collector -> pd.DataFrame -> nifti handler
    Args:
        dcm_collector_list(List[models.hsinchu.DicomCollector]):
        buf_dir: where the dcm2niix application to process.
        out_dir: the dictionary path to store .nii.gz file
        err_dir: the dictionary path to store the error message
        dcm2niix: dcm2niix application location.

    Returns(List[models.hsinchu.HsinchuNiftiHandler]):
        A list that store instance nii.gz file.
    """
    all_nifti: List[models.hsinchu.NiftiHandler] = []
    prepare_df: Dict[str, Any] = CUtils.get_init_prepare_df(dcm_collector_list[0])

    for dcm_entity in dcm_collector_list:
        for key, value in dcm_entity.__dict__().items():
            prepare_df[key].append(value)

    # Because the database operation are more useful, so we build the pd.DataFrame
    df: pd.DataFrame = pd.DataFrame.from_dict(prepare_df)

    for snum_value, snum_entity in df.groupby('snum'):  # Group with same Series Number
        for uid_value, uid_entity in snum_entity.groupby('uid'):    # Group with same Series Instance UID
            for cp_value, cp_entity in uid_entity.groupby('cp'):    # Group with same Cardiac Phase.
                nifti_handler: models.hsinchu.NiftiHandler = models.hsinchu.NiftiHandler(
                    cp_entity, buf_dir, out_dir, error_dir=err_dir, dcm2niix=dcm2niix
                )
                all_nifti.append(nifti_handler)

    return all_nifti


def collect_patient_dcm(
        single_patient_path: str, args: argparse.Namespace,
) -> List[models.hsinchu.DicomCollector]:
    """
        Using to collect all of .dcm file under a patient folder.
    Args:
        single_patient_path:
        args:

    Returns:

    """
    total_dcm: List[models.hsinchu.DicomCollector] = []
    single_patient_path = str(single_patient_path)
    for roots, dirs, files in os.walk(single_patient_path, topdown=True):
        # files: List[str] = list(filter(lambda x: find_legal_dcm(x), files))
        files = list(filter(lambda f: f.endswith(".dcm"), files))
        if len(files) == 0:
            continue
        # print(files)

        for name in files:
            dcm_file: models.hsinchu.DicomCollector = models.hsinchu.DicomCollector(os.path.join(roots, name))
            # print(dcm_file)
            if dcm_file not in total_dcm:   # drop up the duplicate collector
                total_dcm.append(dcm_file)
    return total_dcm


def single_proc(partition: models.Partition):
    proc_id: int = partition.proc_id
    data: List[str] | np.ndarray | Iterable = list(filter(lambda x: os.path.isdir(x), partition.data))
    out_dir: str = partition.out_dir
    buf_dir: str = partition.buf_dir
    err_dir: str = partition.err_dir
    args: argparse.Namespace = deepcopy(partition.args)
    num_patient: int = len(data)
    setattr(args, 'proc_id', proc_id)

    for idx, patient_root in enumerate(data):
        patient_folder_name = re.split('[/\\\]', patient_root)[-1]
        t0 = dt.datetime.now()
        patient_progress: str = f'{idx}/{num_patient}'
        setattr(args, 'patient_progress', patient_progress)
        setattr(args, 't0', t0)
        setattr(args, 'pid', patient_root)
        ComUtils.print_info('Start', '', args)

        try:
            dcm_collector_list: List[models.hsinchu.DicomCollector] = collect_patient_dcm(
                patient_root, args
            )
            nifti_handler_list: List[models.hsinchu.NiftiHandler] = dcm_collector2nifti_handler(
                dcm_collector_list, buf_dir=buf_dir, out_dir=out_dir, err_dir=err_dir, dcm2niix=args.dcm2niix
            )

            # Write down some information about current ct, like file_path, compress ratio.
            content: List[str] = [_ct.str_ratio() for _ct in nifti_handler_list]
            ComUtils.write_content(f'{nifti_handler_list[0].info_dir}/info.txt', content, cover=False, as_json=False)
            end_status = 'Done'
        except Exception as e:
            end_status = 'Error'
            os.makedirs(err_dir, exist_ok=True)
            ComUtils.write_content(f'{err_dir}/{patient_folder_name}.txt', traceback.format_exc(), as_json=False)

        tn = dt.datetime.now()
        ComUtils.print_info(end_status, dict(cost=tn - t0), args)
    return None


def resource_allocate_main(n_proc: int, out_dir, buf_dir, err_dir, args):
    stack = [
        args.data_root
    ]

    for task_root in stack:
        suffix = re.split('[/\\\]', task_root)[-1]
        raw_patient_collections = os.listdir(task_root)
        if args.single_patient_dir is not None:
            raw_patient_collections = list(set(raw_patient_collections) - {args.single_patient_dir})
        if len(raw_patient_collections) == 0:
            continue
        all_patient = list(filter(lambda x: os.path.isdir(x), [f'{task_root}/{pid}' for pid in os.listdir(task_root)]))

        segment_patient = np.array_split(all_patient, n_proc)
        odir = os.path.join(out_dir, suffix)
        bdir = os.path.join(buf_dir, suffix)
        edir = os.path.join(err_dir, suffix)
        for _dir in [odir, bdir, edir]:
            os.makedirs(_dir, exist_ok=True)

        partitions = [models.Partition(i, segment, args, odir, bdir, edir) for i, segment in enumerate(segment_patient)]

        with mp.Pool(n_proc) as pooler:
            pooler.map(single_proc, partitions)


def main(args):
    if (w_ratio := args.worker_ratio) is None:
        nproc = args.num_workers
    else:
        nproc = mp.cpu_count() // w_ratio
    print(f'# of workers: {nproc}')
    out_dir = args.out_dir
    buf_dir = args.buf_dir
    err_dir = args.err_dir

    for _dir in [out_dir, buf_dir, err_dir]:
        os.makedirs(_dir, exist_ok=True)

    resource_allocate_main(nproc, out_dir, buf_dir, err_dir, args=args)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=False)
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--single_patient_dir', type=str, required=False)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--worker_ratio', type=float, default=None)
    # parser.add_argument('--dst_root', type=str, default='./NiiHsinChu')
    parser.add_argument('--out_dir', default='./NiiHsinChu/out')
    parser.add_argument('--buf_dir', default='./NiiHsinChu/buf')
    parser.add_argument('--err_dir', default='./NiiHsinChu/err')
    parser.add_argument('--dcm2niix', default='./lib/dcm2niix.exe')
    gargs: argparse.Namespace = ComUtils.make_final_namespace(parser.parse_args())
    print(f'Argument:\n{gargs}')
    main(gargs)
    print(f'Script Done.')