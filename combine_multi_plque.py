from typing import Callable, MutableMapping, Mapping, Optional
import argparse
import multiprocessing as mp
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(name)s %(levelname)s] %(message)s')
import os
import re
from operator import itemgetter

# import nibabel as nib
import numpy as np
from monai.data import MetaTensor
from monai import transforms as MT
import torch

from models.CommonDataModels import Partition, SegmentMetaPack, Detail
from utils.common_utils import write_content, load_json, make_final_namespace, delete_file
from utils import hooker


def patient_worker(pid, loader: Callable, saver: Callable, args, **kwargs):
    def _load_nii_gz(_path: str | os.PathLike,
                     _union_label: Optional[MetaTensor | torch.Tensor] = None) -> MetaTensor | torch.Tensor | None:
        if os.path.exists(_path):
            return loader(_path)
        if _union_label is not None:
            return torch.zeros_like(_union_label)
        return None

    def _pop_head_slash(_path: str | os.PathLike) -> str | os.PathLike:
        if _path is None:
            return None
        if _path[0] not in r'/\\':
            return _path
        return _pop_head_slash(_path[1:])

    paired_list: list[SegmentMetaPack] = [SegmentMetaPack(**_pack) for _pack in
                                          load_json(os.path.join(args.root, args.meta_dir, pid))]
    repeat_map_image_path2pack_list: MutableMapping[str, list[SegmentMetaPack]] = dict()
    deduplicate_meta_table = list()
    proc_id = kwargs.get('proc_id')
    prog = kwargs.get('prog')
    # First drop.
    if len(paired_list) == 0:
        logging.info(f'Proc-{proc_id}|[{prog}]|[ No paired ]')
        return deduplicate_meta_table
    # Step.1 Make the image name and all [mask, plaque] mapping relationship collections.
    for pack in paired_list:
        image_path = _pop_head_slash(pack.image)
        # Checking current image_path_as_key exist.
        if repeat_map_image_path2pack_list.get(image_path) is None:
            repeat_map_image_path2pack_list[image_path] = list()
        # Checking done.
        repeat_map_image_path2pack_list[image_path].append(pack)
        pass
    # Step.1 Done.
    merge_getter: itemgetter = itemgetter(1, 3, 4)
    # Step.2 Try to merge all of not unique mapping relationship.
    for image_name_cursor, pack_list in repeat_map_image_path2pack_list.items():
        if len(pack_list) == 1:
            deduplicate_meta_table.append(pack_list[0])
            continue
        logging.info(f'Proc-{proc_id}|[{prog}]|[ INFO ]|Pair repeat: {os.path.join(image_name_cursor)}')
        sample_pack: 'SegmentMetaPack' = pack_list[0]
        pre_union_label_path = sample_pack.mask.replace('\\', '/').lstrip('/')
        union_label_path = os.path.join(args.root, pre_union_label_path)
        union_label = _load_nii_gz(union_label_path)

        all_plq = [_load_nii_gz(os.path.join(args.root, _pack.plaque.replace('\\', '/').lstrip('/')), union_label) for
                   _pack in filter(lambda __pack: __pack.plaque is not None, pack_list)]
        all_details: list[Detail] = list()
        for _pack in pack_list:
            all_details.extend(getattr(_pack, 'details', list()))
        all_details = list(set(all_details))
        newest_union_plq = torch.zeros_like(union_label)
        cursor_image_name_comp: list[str] = re.split(r'[/\\]', image_name_cursor)
        img_pid, uid, cp = merge_getter(cursor_image_name_comp)
        merge_label_dir = os.path.join(args.mask_dir, img_pid, uid, str(cp))

        for plq in all_plq:
            union_label[plq == 1] = 10
            newest_union_plq[plq == 1] = 1
        try:
            saver(union_label, union_label.meta, filename=os.path.join(args.root, merge_label_dir, 'merge_union_label'))
            saver(newest_union_plq, union_label.meta, filename=os.path.join(args.root, merge_label_dir, 'merge_plaque'))
        except Exception as e:
            print(image_name_cursor)
            print(merge_getter(cursor_image_name_comp))
        deduplicate_pack = {
            'image': os.path.join(*image_name_cursor).replace('\\', '/').lstrip('/'),
            'mask': os.path.join(merge_label_dir, 'merge_union_label.nii.gz').replace('\\', '/').lstrip('/'),
            'plaque': os.path.join(merge_label_dir, 'merge_plaque.nii.gz').replace('\\', '/').lstrip('/'),
            'cp': cp,
            'uid': uid,
            'pid': img_pid
        }
        if len(all_details) > 0:
            deduplicate_pack['details'] = [detail.__dict__ for detail in all_details]
        deduplicate_meta_table.append(deduplicate_pack)

    return deduplicate_meta_table


def worker(partitions: Partition):
    data = partitions.data
    proc_id = partitions.proc_id
    args = partitions.args
    loader = MT.Compose([
        MT.LoadImage(), MT.EnsureChannelFirst(), MT.Orientation(axcodes='RAS')
    ])
    saver = MT.SaveImage(output_postfix='', separate_folder=False, print_log=False)
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


@hooker.timer
def clean_all_merge_sample(args):
    root_path = args.root
    mask_path = os.path.join(root_path, args.mask_dir)
    logging.info(f'Start Cleaning All Merge Sample under {mask_path}')
    for roots, dirs, files in os.walk(mask_path, topdown=True):
        files = list(filter(lambda x: x.endswith('.nii.gz') and 'merge' in x, files))
        if len(files) == 0:
            continue
        delete_file(os.path.join(roots, 'merge_plaque.nii.gz'))
        delete_file(os.path.join(roots, 'merge_plaque.nii.gz.nii.gz'))
        delete_file(os.path.join(roots, 'merge_union_label.nii.gz'))
        delete_file(os.path.join(roots, 'merge_union_label.nii.gz.nii.gz'))


def main(args: argparse.Namespace):
    all_patient_dir = get_patient_dir(args)
    clean_all_merge_sample(args)
    if args.just_clean_merge_mask:
        logging.info('Argument `just_clean_merge_mask` is set to True, thus the program will exit right now.')
        return
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
    parser = argparse.ArgumentParser(
        description='This scripts is using to merge that plaque is storage in multi-file, it will store all of meta pack into one json file.')
    parser.add_argument('--config', type=str, required=False,
                        help='If don\'t want apply argument, you could write all setting into this json file.')
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
    parser.add_argument('--just_clean_merge_mask', action='store_true', default=False)
    main(make_final_namespace(parser.parse_args()))
