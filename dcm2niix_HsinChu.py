"""
This Script is using to convert the raw image which come from HsinChu Branch Hospital into the deeplearning usable format
Nifit file,
"""
import argparse
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


def collect_patient_dcm(single_patient_path: str, args, out_dir='./out/hsinchu', buf_dir='./buf/hsinchu'):
    prepare_df = dict()
    total_dcm = []
    single_patient_path = str(single_patient_path)
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
    return split2ct(df, out_dir=out_dir, buf_dir=buf_dir, dcm2niix=args.dcm2niix)


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
        print(f'Process-{current_pid:02}|[Start]|[{idx}/{total}]|{patient_root}, {get_now(t0)}')

        try:
            all_group_ct: list[CTFile] = collect_patient_dcm(patient_root, out_dir, buf_dir, args=args)
            # Write down some information about current ct, like file_path, compress ratio.
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
        print(f'Process-{current_pid}|[{status:^5}]|[{idx}/{total}]|{patient_root}, {get_now(tn)}, cost: {tn - t0}')


def start_point(n_proc: int, out_dir, buf_dir, err_dir, args):
    stack = [
        args.data_root
    ]

    for task_root in stack:
        suffix = re.split('[/\\\]', task_root)[-1]
        all_patient = list(filter(lambda x: os.path.isdir(x), [f'{task_root}/{pid}' for pid in os.listdir(task_root)]))
        segment_patient = np.array_split(all_patient, n_proc)
        odir = f'{out_dir}/{suffix}'
        bdir = f'{buf_dir}/{suffix}'
        edir = f'{err_dir}/{suffix}'
        for _dir in [odir, bdir, edir]:
            os.makedirs(_dir, exist_ok=True)

        partitions = [Partition(i, segment, odir, bdir, edir) for i, segment in enumerate(segment_patient)]

        with mp.Pool(n_proc) as pooler:
            pooler.map(middle, partitions)


def task1(args):
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

    start_point(nproc, out_dir, buf_dir, err_dir, args)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--worker_ratio', type=float, default=None)
    # parser.add_argument('--dst_root', type=str, default='./NiiHsinChu')
    parser.add_argument('--out_dir', default='./NiiHsinChu/out')
    parser.add_argument('--buf_dir', default='./NiiHsinChu/buf')
    parser.add_argument('--err_dir', default='./NiiHsinChu/err')
    parser.add_argument('--dcm2niix', default='./lib/dcm2niix.exe')
    args = parser.parse_args()

    task1(args)