import argparse
from copy import deepcopy
import os
from multiprocessing import Pool
import re
import traceback as tb
# import monai.transforms as MT
import nibabel as nib
import numpy as np
import pydicom as pyd

from models.CommonDataModels import Partition
from utils import common_utils as ComU
import isp_helper as IHelper


def pack_worker(pack_list: list[dict], partitions: Partition) -> list[dict]:
    """
    Wrapping all plaque and save into independency .nii.gz file. All remains process must complete at here.
    Args:
        pack_list: come from meta json file.
        partitions:

    Returns: A sequence of updated plaque file

    """
    def _mask_isp_dir_linked(_patient_id) -> dict[str, str | os.PathLike]:
        """
            This function is using to create a dictionary mapping an ISP title (folder name) to an absolute path.
        Args:
            _patient_id(str) the current patient id.

        Returns(dict[str, str | os.PathLike]): the key is an ISP title, the value is an absolute path.
        Example:
            >>> {
            >>>     'SE804 plaque results 40__ 8_ v2_ YHT': '<isp_parent_path>/0004/CT/20091120/SE804 plaque results 40__ 8_ v2_ YHT',
            >>>     ...
            >>> }

        """
        _linked = dict()
        _pid_isp_dir = os.path.join(partitions.args.isp_dir, _patient_id)
        for _roots, _dirs, _files in os.walk(_pid_isp_dir):
            if len([_file.endswith('.dcm') for _file in _files]) <= 0:
                continue
            # print(f'root: {_roots}, dir: {_dirs}')
            _isp_title = re.split(r'[/\\]', _roots)[-1]
            _linked[_isp_title] = _roots
        return _linked

    isp_name2isp_path = _mask_isp_dir_linked(partitions.args.pid)
    total = len(pack_list)
    for prog, pack in enumerate(pack_list):
        # print(pack)
        ComU.print_info('Start', {'pack_id': prog, 'total_pack': total}, partitions.args)
        print(type(pack))
        pid = pack['pid']
        cp = str(pack['cp'])
        uid = pack['uid']
        if 'plaque' not in pack.keys():
            continue

        plq_dir_path = os.path.join(partitions.args.root, partitions.args.mask_dir, pid, uid, cp, 'details')
        os.makedirs(plq_dir_path, exist_ok=True)
        pack['details'] = list()

        host_ct = nib.load(os.path.join(partitions.args.root, pack['image'].replace('\\', '/').lstrip('/')))
        mask_name_series: list[str] = re.split(r'[/\\]', pack['plaque'])
        isp_folder_name: str = mask_name_series[-2]
        current_isp_parent: str | os.PathLike = isp_name2isp_path[isp_folder_name]
        all_isp_file = os.listdir(current_isp_parent)
        for isp_file_name in all_isp_file:
            isp = pyd.dcmread(os.path.join(current_isp_parent, isp_file_name))
            if isp.ImageType[-1] != 'PATH':
                continue
            plq_list: list[list[np.ndarray, str]] = IHelper.reconstruct_plaque(isp, host_ct, return_plq_name=True)
            vessel_name = isp.ImageComments
            centerline = np.array(isp[(0x07a1, 0x1012)].value).reshape((-1, 3, 3)).astype(np.float16)
            centerline_store_path = os.path.join(plq_dir_path, vessel_name + 'centerline.npy')
            np.save(centerline_store_path, centerline)
            for plq_pack in plq_list:
                plq_nii = nib.Nifti1Image(plq_pack[0].astype(np.uint8), host_ct.affine, dtype=np.uint8)
                dst_path = os.path.join(plq_dir_path, plq_pack[1] + '.nii.gz')
                nib.save(plq_nii, dst_path)
                pack['details'].append({
                    'path': dst_path.lstrip(partitions.args.root),
                    'desc': plq_pack[1],
                    'vessel': centerline_store_path.lstrip(partitions.args.root)
                })

    return pack_list


def patient_worker(partition: Partition):
    """
    In this steps, we only handle at loading meta file.
    Args:
        partition:

    Returns:

    """
    args = partition.args
    args.proc_id = partition.proc_id
    meta_list = partition.data
    total = len(meta_list)

    root: str = args.root

    for prog, meta_json in enumerate(meta_list):
        # Setting UI needed data.
        args.pid = meta_json.split('.')[0]
        args.patient_progress = f'{prog}/{total}'
        ComU.print_info('Start', '', args=args)
        # Setting done.
        meta_path = os.path.join(root, args.meta_dir, meta_json)
        meta = ComU.load_json(meta_path)
        try:
            new_meta = pack_worker(meta, partition)
            ComU.write_content(meta_path, new_meta, cover=True, as_json=True)
            status = 'Done'
        except Exception as e:
            print(e.args)
            print(tb.format_exc())
            status = 'Failed'
        ComU.print_info(status, '', args=args)

def main(args: argparse.Namespace):
    num_worker = args.num_worker
    meta_path = os.path.join(args.root, args.meta_dir)
    all_meta_file = list(filter(lambda x: re.fullmatch('[0-9]{4}\.json', x) != None, os.listdir(meta_path)))
    if num_worker > len(all_meta_file):
        num_worker = len(all_meta_file)
    partition_collections = np.array_split(all_meta_file, num_worker)
    partitions: list[Partition] = [Partition(proc_id=pid, data=partition, args=deepcopy(args)) for pid, partition in
                                   enumerate(partition_collections)]

    with Pool(num_worker) as pooler:
        pooler.map(patient_worker, partitions)

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruct all of unique plaque and save.')
    parser.add_argument('--config', type=str, required=False,
                        help='You can save all of argument into a .json file. If config and argument both assign, argument have prior.')
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--isp_dir', type=str, default=None)
    parser.add_argument('--meta_dir', type=str, default='meta')
    parser.add_argument('--mask_dir', type=str, default='mask')
    parser.add_argument('--num_worker', type=int, default=1)

    main(ComU.make_final_namespace(parser.parse_args()))
