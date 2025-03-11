import argparse
import multiprocessing as mp
import shutil
import os
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(name)s - %(levelname)s] %(message)s')

import numpy as np
from models.CommonDataModels import Partition

def worker(parts):
    proc_id = parts.proc_id
    data = parts.data
    args = parts.args
    data_size = len(data)

    for prog, element in enumerate(data):
        logging.info(f'Proc-{proc_id}|[ Start ]|[{prog}/{data_size}]|{element}')
        if os.path.isdir(os.path.join(args.root, element)):
            shutil.rmtree(os.path.join(args.root, element))
        else:
            os.remove(os.path.join(args.root, element))
    logging.info(f'Proc-{proc_id}|[ Done ]')


def main(args):
    nworkers = args.num_workers
    all_element = os.listdir(args.root)
    ans = input(f'{all_element}\nAre you sure you want to delete all files in {args.root}?[y/N]')
    if ans.lower() != 'y':
        print(f'Program End...')
        exit(0)
    if nworkers > (n_element := len(all_element)):
        print(f'# of workers ({nworkers}) is less than {n_element}')
        print(f'Change # of workers from {n_element} to {nworkers}')
        nworkers = n_element
    array_split = np.array_split(all_element, nworkers)
    parts = [Partition(proc_id, data, args) for proc_id, data in enumerate(array_split)]

    with mp.Pool(nworkers) as pool:
        pool.map(worker, parts)
    print(f'Removed {args.root} DONE.')
    print(f'Program End....')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--root', type=str)
    args = parser.parse_args()
    main(args)




