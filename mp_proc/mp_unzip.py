import os
import json
import argparse
import zipfile
import traceback as tb
from os.path import join, exists
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm.auto import tqdm

def load_json(path):
    with open(path, 'r') as loader:
        if path.endswith('.jsonl'):
            return [json.loads(line) for line in loader.readlines()]
        return json.load(loader)


def worker(zip_name, args):
    def _record_error(content, who):        
        with open(join(args.err_dir, f'{who}.txt'), 'a+') as _writer:
            _writer.write(f'{content}\n\n')            
    full_zip_path = join(args.root, zip_name)

    dst_unzip_folder = join(args.output_dir, os.path.basename(full_zip_path).replace('.zip', ''))
    try:
        zipManage = zipfile.ZipFile(full_zip_path, 'r')
    except zipfile.BadZipFile as e:
        print(full_zip_path)
        _record_error(f'Zip-File: {full_zip_path}\nMSG: {e.args}', who=zip_name.replace('.zip', ''))
        return

    bad_cnt = 0
    for zip_file in zipManage.namelist():
        if os.path.exists(join(dst_unzip_folder, zip_file)):
            continue
        try:
            zipManage.extract(zip_file, dst_unzip_folder)
        except Exception as e:            
            bad_cnt += 1
            _record_error(f'File: {zip_file}\nMSG: {e.args}', who=zip_name.replace('.zip', ''))
    zipManage.close()            
    return bad_cnt
    pass


def main(args):
    n_worker = args.n_worker
    os.makedirs(args.err_dir, exist_ok=True)
    root = args.root
    all_zip_file_name = list(filter(lambda name: name.endswith('.zip'), os.listdir(root)))
    if args.manual_json_list is not None:
        print(f'Loading all possible indicated patient...')
        manual_set = set(f'{pid}.zip' for pid in load_json(args.manual_json_list))
        org_len = len(all_zip_file_name)
        all_zip_file_name = list(set(all_zip_file_name) & manual_set)
        after_len = len(all_zip_file_name)
        print(f'After doing Intersections, The final size from {org_len} ---> {after_len}')
    pbar = tqdm(total=len(all_zip_file_name), desc='Start Unzip...')

    with ProcessPoolExecutor(n_worker) as launcher:
        stats = list()
        for zip_name in all_zip_file_name:
            stats.append(launcher.submit(worker, zip_name, args))
        for rep in as_completed(stats):
            bad_cnt = rep.result()
            pbar.update(1)
            pbar.set_postfix({"Bad File": bad_cnt})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_worker', default=1, type=int)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--root', type=str)
    parser.add_argument('--err_dir', type=str, default='./unzip_err')
    parser.add_argument('--manual_json_list', type=str, default='../test/ignore_pid.json')
    main(parser.parse_args())