from typing import Callable
import argparse
import multiprocessing as mp
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(name)s %(levelname)s] %(message)s')
import os
import re

# import nibabel as nib
import numpy as np
from monai import transforms as MT
import torch

from models.CommonDataModels import Partition
from utils.common_utils import write_content, load_json


def patient_worker(pid, loader: Callable, saver: Callable, args, **kwargs):
    def _load_nii_gz(_path, _union_label=None):
        if os.path.exists(_path):
            return loader(_path)
        if _union_label is not None:
            return torch.zeros_like(_union_label)
        return None
    def _pop_head_slash(_path):
        if _path is None:
            return None
        if _path[0] not in ['/', r'\\']:
            return _path
        return _path[1:]

    paired_list = load_json(os.path.join(args.root, args.meta_dir, pid))
    repeat_map = dict()
    deduplicate_meta_table = list()
    proc_id = kwargs.get('proc_id')
    prog = kwargs.get('prog')

    for pack in paired_list:
        if repeat_map.get((image_path := pack['image'])) is None:
            repeat_map[image_path] = list()
        repeat_map[image_path].append(pack)
    for image_name, pack_list in repeat_map.items():
        if len(pack_list) == 1:
            deduplicate_meta_table.append(pack_list[0])
            continue
        logging.info(f'Proc-{proc_id}|[{prog}]|Pair repeat: {image_name}')
        p0 = pack_list[0]
        pre_union_label_path = _pop_head_slash(p0.get('label', p0.get('mask')))
        union_label_path = os.path.join(args.root, pre_union_label_path)
        union_label = _load_nii_gz(union_label_path)
        all_plq = [_load_nii_gz(os.path.join(args.root, _pop_head_slash(_pack.get('plaque', '_'))), union_label) for _pack in pack_list]
        bg = torch.zeros_like(union_label)
        merge_label_dir = os.path.join(args.mask_dir, p0['pid'], p0['uid'], str(p0['cp']))

        for plq in all_plq:
            union_label[plq == 1] = 10
            bg[plq == 1] = 1
        saver(union_label, union_label.meta, filename=os.path.join(args.root, merge_label_dir, 'merge_union_label.nii.gz'))
        saver(bg, union_label.meta, filename=os.path.join(args.root, merge_label_dir, 'merge_plaque.nii.gz'))
        deduplicate_meta_table.append({
            'image': image_name,
            'mask': os.path.join(merge_label_dir, 'merge_union_label.nii.gz'),
            'plaque': os.path.join(merge_label_dir, 'merge_plaque.nii.gz'),
            'cp': p0['cp'],
            'uid': p0['uid'],
            # 'snum': p0['snum'],
        })

    return deduplicate_meta_table

def worker(partitions: Partition):
    data = partitions.data
    proc_id = partitions.proc_id
    args = partitions.args
    loader = MT.Compose([
        MT.LoadImage(), MT.EnsureChannelFirst(), MT.Orientation(axcodes='RAS')
    ])
    saver = MT.SaveImage(output_postfix='', separate_folder=False)
    total = len(data)
    final_table = list()

    for i, patient_id in enumerate(data):
        prog = f'{i}/{total}'
        logging.info(f'Proc-{proc_id}|[{prog}]|[ Start ][{patient_id}]')
        final_table.extend(patient_worker(patient_id, loader, saver, args, prog=prog, proc_id=proc_id))
        logging.info(f'Proc-{proc_id}|[{prog}]|[ Next ]')
    logging.info(f'Proc-{proc_id}|[ Process Done ]')
    return final_table


def get_patient_dir(args):
    if (pid := getattr(args, 'patient_id')) is not None:
        logging.info(f'The `patient_id` is setting, the `num_worker` will set to `1`')
        args.num_worker = 1
        return [f'{pid}.json']

    return list(filter(lambda _x: _x.endswith('.json') and re.fullmatch(r'\d+\.json', _x) is not None,
                       os.listdir(os.path.join(args.root, args.meta_dir))))


def main(args):
    all_patient_dir = get_patient_dir(args)
    mapping_patient = np.array_split(all_patient_dir, args.num_worker)
    partitions = [Partition(proc_id=proc_id, data=_data, args=args) for proc_id, _data in enumerate(mapping_patient)]

    with mp.Pool(args.num_worker) as pooler:
        worker_collections = pooler.map(worker, partitions)
    merge_list = list()
    for worker_list in worker_collections:
        merge_list.extend(worker_list)
    write_content(os.path.join(args.root, args.meta_dir, 'Deduplicate_PLQ_Table.json'), merge_list, as_json=True)



    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This scripts is using to merge that plaque is storage in multi-file')
    parser.add_argument('--root', type=str,
                        help='The root directory of `image_dir`, `mask_dir`, like: `/<path>/NiiTaipei', )
    parser.add_argument('--image_dir', type=str,
                        help='input folder, the patient must under here, like if path is `/<path>/NiiTaipei/out`, just given `out`')
    parser.add_argument('--mask_dir', type=str,
                        help='mask folder, the patient must under here, like if path is `/<path>/NiiTaipei/mask`, just given `mask`')
    parser.add_argument('--meta_dir', type=str,
                        help='meta folder, the patient must under here, like if path is `/<path>/NiiTaipei/meta`, just given `meta`')
    parser.add_argument('--data_pair_json', type=str,
                        help='json file that store each image and mask pair json file, also its exclusive with `--meta_dir`')

    parser.add_argument('--patient_id', type=str, help='Specified this argument let only merge one patient')

    parser.add_argument('--num_worker', type=int, default=1, help='number of worker process')

    main(parser.parse_args())
