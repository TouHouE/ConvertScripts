"""
The needed keys for meta json2csv:
1. pid
2. cp
3. snum
4. year
5. month
6. day
7. image
8. label
9. desc
10. raw
"""
import argparse
import os
import re
import json
from os.path import join, exists

import pandas as pd


def load_json(path: str) -> list:
    with open(path, 'r') as loader:
        return json.load(loader)


def wrap_cp_from_path(image_path: str) -> float:
    """
        try to wrap possible cardiac phase from image path.
    """
    cand0 = re.split(r'[/\\]', image_path)[-3]
    if cand0 != '0.0':
        return float(cand0)
    bname = os.path.basename(image_path)
    cand1 = bname.split('_')[-3]
    if cand1.endswith("%"):
        cand1 = cand1.replace("%", "")
    try:
        cand1 = float(cand1)
    except Exception as e:
        return 0.0
    return cand1
    

def main(args):    
    # root = '/media/shard/data/ct_scan/Taipei_2897/meta'
    root: str = args.meta_path
    un_root = join(root, 'unpair')
    paired_json = load_json(join(root, 'Deduplicate_PLQ_Table.json'))
    # for un_pair_named in os.listdir(un_root):    
    unpaired_json_list = list()
    for pid in os.listdir(un_root):
        jpath = join(un_root, pid, 'unpair.json')
        # unpaired_json_list.extend([{'image': image_path } for image_path in load_json(jpath)['ct']])
        for image_path in load_json(jpath)['ct']:
            bname = os.path.basename(image_path)
            datestr = bname.split("_")[-2]
            pack = {
                'pid': pid,
                'cp': wrap_cp_from_path(image_path),
                'uid': re.split(r'[/\\]', image_path)[-4],
                'image': image_path,
                'mask': None,
                'plaque': None,
                'details': None,
                'year': datestr[:4],
                'month': datestr[4:6],
                'day': datestr[6:8]
            }
            unpaired_json_list.append(pack)    
    total_list = paired_json + unpaired_json_list
    total_df = pd.DataFrame(total_list)
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        oname = args.output_name
        if not oname.endswith('.csv'):
            oname = f'{oname}.csv'        
        total_df.to_csv(join(args.output_dir, oname))
    except Exception as e:
        if not args.do_bp:
            return
        breakpoint()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output_dir')
    parser.add_argument('--output_name', type=str, default='out.csv')
    parser.add_argument('--do_bp', action='store_true', default=False)
    main(parser.parse_args())