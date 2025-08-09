import argparse
import json
import os
import re
from os.path import join, exists
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import MutableMapping

import pandas as pd
import nibabel as nib

DEBUG: bool = os.environ.get("DEBUG", "0") == "1"

def dp(_obj):
    if DEBUG:
        print(_obj)
    return _obj


@dataclass(slots=True)
class HsinchuImage:
    rel_path: str
    pid: str = field(init=False)
    series: str = field(init=False)
    cp: float = field(init=False)
    year: int = field(init=False)
    month: int = field(init=False)
    day: int = field(init=False)
    
    def __post_init__(self):
        filename = os.path.basename(self.rel_path)
        file_desc = filename.replace('.nii.gz', '').split("_")
        _pid, _cp, _series, _year, _month, _day = [None] * 6
        while True:
            try:
                if _pid is None:
                    _pid = re.split(r'[\\/]', self.rel_path)[2]
                if _cp is None:
                    _cp = float(file_desc[0])
                series_cand_1, series_cand_2 = file_desc[-2:]
                if _series is None:
                    if not series_cand_2.isdecimal():
                        _series = series_cand_1
                        full_date = file_desc[-3]
                    else:
                        _series = series_cand_2
                        full_date = file_desc[-2]
                if any(time_comp is None for time_comp in [_year, _month, _day]):
                    _year = int(full_date[: 4])
                    _month = int(full_date[4: 6])
                    _day = int(full_date[6: 8])
            except Exception as e:
                print("""=====================================================
                      Error: No candidate datetime string found in the program. Please manually enter the following variables:
                      1. `_year`
                      2. `_month`
                      3. `_day`
                      ==================================================""")
                breakpoint()
                continue
            break
        self.pid = _pid
        self.cp = _cp
        self.series = _series
        self.year = _year
        self.month = _month
        self.day = _day



def load_info(path) -> list[str]:    
    if exists((info_path := join(path, 'info.txt'))):
        with open(info_path, 'r', encoding='utf-8') as loader:
            return list(line.strip('\n') for line in loader.readlines())    
    info_as_list = list()
    for roots, dirs, files in os.walk(path):
        files = list(filter(lambda name: '.nii.gz' in name, files))
        if len(files) == 0:
            continue
        fake_windows_abs_path = join(r'G:\Hsinchu', *(re.split(r'[\\/]', roots)[6:]))
        info_as_list.extend([f'{join(fake_windows_abs_path, fname)},' for fname in files])        

    return info_as_list



def load_line_worker(line, args) -> HsinchuImage | None:
    win_abs_path, *_ = line.split(',')
    rel_path = join(*dp(re.split(r'[\\/]', win_abs_path)[2:]))
    # rel_path = win_abs_path.replace(r"G:\Hsinchu\\", "")
    # '/media/shard/data/ct_scan/Hsinchu'
    nii = nib.load(join(args.hsinchu_root, rel_path))
    if nii.ndim > 3:
        return 
    
    return HsinchuImage(rel_path)


def build_each_line(line_group, args) -> list[HsinchuImage]:
    result_list: list[HsinchuImage] = list()
    with ProcessPoolExecutor(5) as pooler:
        pooler_rep = list()
        for line in line_group:
            pooler_rep.append(pooler.submit(load_line_worker, line, args))
        
        for rep in as_completed(pooler_rep):
            if (result := rep.result()) is None:
                continue                
            result_list.append(result)
    return result_list


def get_all_patient_list(args) -> MutableMapping[str, list[HsinchuImage]]:
    # hsinchu_abs_root = '/media/shard/data/ct_scan/Hsinchu/out'
    hsinchu_abs_root = join(args.hsinchu_root, 'out')
    all_frag_dir = os.listdir(hsinchu_abs_root)
    patient_map: MutableMapping[str, list[HsinchuImage]] = dict()
    for frag in all_frag_dir:
        # e.g.: /media/shard/data/ct_scan/Hsinchu/out + RawHsinChu13
        abs_frag_path = join(hsinchu_abs_root, frag)
        for pid in os.listdir(abs_frag_path):
            all_line = load_info(join(abs_frag_path, pid))
            patient_map[pid] = build_each_line(all_line)
    return patient_map


def main(args):
    # Step-1. Try to Loading all of static 3D CT Stack
    pid2ct_stack_list: MutableMapping[str, list[HsinchuImage]] = get_all_patient_list(args)
    just_list = list()
    for sub_list in pid2ct_stack_list.values():
        just_list.extend(sub_list)
    df = pd.DataFrame([asdict(data) for data in just_list])

    os.makedirs(args.output_dir, exist_ok=True)    
    if not (oname := args.output_name).endswith('.csv'):
        oname = f"{oname}.csv"
    final_path = join(args.output_dir, oname)
    # df.to_csv('/home/user/workspace/hsu/ListHsinchu/all_hsinchu.csv', index=False, index_label=False)
    df.to_csv(final_path, index=False, index_label=False)
    print('Done')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hsinchu_root', type=str, default='/media/shard/data/ct_scan/Hsinchu/out')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--output_name', type=str, default='all_hsinchu.csv')
    main(parser.parse_args())