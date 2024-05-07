"""
This Script is using to convert the raw image which come from HsinChu Branch Hospital into the deeplearning usable format
Nifit file,
"""
import datetime as dt
import traceback
from functools import partial
import pydicom as pyd
import nibabel as nib
import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import subprocess as sp
import isp_helper as ISPH
from operator import methodcaller
import re
from scripts_class import DCMFile, CTFile, split2ct, Partition, get_now


def find_legal_dcm(dcm_name):
    not_dup = re.fullmatch('.*\[[0-9]{1,}\]\.dcm', dcm_name) is None
    return not_dup


def _get_cp(dcm: pyd.FileDataset) -> int | float:
    # cp = None
    for tag_candidate in [(0x0020, 0x9241), (0x01f1, 0x1041), (0x7005, 0x1004), (0x7005, 0x1005)]:
        cp = dcm.get(tag_candidate)
        if cp is None:
            continue
        if (cand_cp := cp.value).is_digit():
            return float(cand_cp)
        else:
            return float(cand_cp[:-1])

    return .0


def collect_patient_dcm(single_patient_path: str, out_dir='./out/hsinchu', buf_dir='./buf/hsinchu'):
    prepare_df = dict()
    total_dcm = []
    for roots, dirs, files in os.walk(single_patient_path, topdown=True):
        files = list(filter(lambda x: find_legal_dcm(x), files))
        if len(files) == 0:
            continue

        for name in files:
            dcm_file = DCMFile(f'{roots}/{name}')
            if dcm_file not in total_dcm:
                total_dcm.append(dcm_file)

    # ---------------------- All DCM are collected.------------------------ #

    # Initialize the dataframe candidate. especially the key.
    for key in total_dcm[0].__dict__():
        prepare_df[key] = []

    # Now, we extract the required data from ```DCMFile``` into the dataframe candidate.
    for dcm_entity in total_dcm:
        for key, value in dcm_entity.__dict__().items():
            prepare_df[key].append(value)
    # Because the database operation are more useful, so we build the pd.DataFrame
    df = pd.DataFrame.from_dict(prepare_df)
    # This method can group all different series, uid, cardiac phase, as single nifit file.
    return split2ct(df, out_dir=out_dir, buf_dir=buf_dir)


def middle(partition):
    current_pid = partition.proc_id
    data = list(filter(lambda x: os.path.isdir(x), partition.data))
    total = len(data)
    out_dir = partition.out_dir
    buf_dir = partition.buf_dir
    err_dir = partition.err_dir

    for idx, patient_root in enumerate(data):
        patient_folder_name = re.split('[/\\\]', patient_root)[-1]
        t0 = dt.datetime.now()
        print(f'Process-{current_pid}|[Start]|{idx}/{total}|{patient_root}, {get_now(t0)}')
        try:
            all_group_ct: list[CTFile] = collect_patient_dcm(patient_root, out_dir, buf_dir)
            with open(f'{all_group_ct[0].info_dir}/info.txt', 'a+') as fout:
                for ct in all_group_ct:
                    fout.write(f'{ct.str_ratio()}\n')

            status = 'Done'
        except Exception as e:
            status = 'Error'
            os.makedirs(err_dir, exist_ok=True)
            with open(f'{err_dir}/{patient_folder_name}.txt', 'w+') as fout:
                fout.write(traceback.format_exc())
        tn = dt.datetime.now()
        print(f'Process-{current_pid}|[{status:^5}]|{idx}/{total}|Cost: {tn - t0}|{patient_root}, {get_now(tn)}')


def start_point(n_proc: int, out_dir='./out/hsinchu', buf_dir='./buf/hsinchu', err_dir='./err/hsinchu'):
    stack = [
        r'G:\RAW\batch1',
        r"G:\RAW\batch2-2024-04-30\202309CCTA(13例)", r"G:\RAW\batch2-2024-04-30\202310CCTA(14例)",
        r"G:\RAW\batch2-2024-04-30\202311CCTA(16例)", r"G:\RAW\batch2-2024-04-30\202312CCTA(26例)",
    ]

    for task_root in stack:
        suffix = re.split('[/\\\]', task_root)[-1]
        all_patient = list(filter(lambda x: os.path.isdir(x), [f'{task_root}/{pid}' for pid in os.listdir(task_root)]))
        segment_patient = np.array_split(all_patient, n_proc)
        odir = f'{out_dir}/{suffix}'
        bdir = f'{buf_dir}/{suffix}'
        edir = f'{err_dir}/{suffix}'
        partitions = [Partition(i, segment, odir, bdir, edir) for i, segment in enumerate(segment_patient)]

        with mp.Pool(n_proc) as pooler:
            pooler.map(middle, partitions)


def task1():
    nproc = mp.cpu_count() // 4
    out_dir = r'G:\NiiHsinChu\out'
    buf_dir = r'G:\NiiHsinChu\buf'
    err_dir = r'G:\NiiHsinChu\err'
    for dir in [out_dir, buf_dir, err_dir]:
        os.makedirs(dir, exist_ok=True)

    start_point(nproc, out_dir, buf_dir, err_dir)


if __name__ == '__main__':
    task1()