import datetime as dt
import numpy as np
import os
import json
import pandas as pd
import argparse
import prompt_scripts_utils as PSU


def load_nii(path) -> list[PSU.NiiCTContainer]:
    all_patient = os.listdir(path)
    all_nii = list()
    for pid in all_patient:
        try:
            _df = pd.read_csv(f'{path}/{pid}/info.txt', header=None, encoding='iso-8859-1')
        except Exception as e:
            print(f'{pid} did not contain info.txt, trying another way.')
            mapper = {0: [], 1: []}.copy()

            for roots, dirs, files in os.walk(f'{path}/{pid}', topdown=True):
                files = list(filter(lambda x: x.endswith('.nii.gz'), files))
                if files.__len__() < 1:
                    continue
                
                mapper[0].extend([f'{roots}/{fname}' for fname in files])
                mapper[1].extend([roots] * len(files))
            _df = pd.DataFrame(mapper, columns=list(mapper.keys()))
            
        _buf = PSU.info_txt2CT(pid, _df)
        all_nii.append(_buf)
    return all_nii


def load_report(path) -> pd.DataFrame:
    if path.endswith(".xlsx"):
        # with open(path, 'r+', encoding='iso-8859-1') as stream:
        return pd.read_excel(path)
    elif path.endswith('.csv'):
        return pd.read_csv(path)
    return None


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--report_file')
    parser.add_argument('--ct_root')
    parser.add_argument('--prompt_template_path', type=str, default='./template.json')
    parser.add_argument('--json_path', type=str, default='./medical-conversations.json')
    return parser


def merge_path_and_prompt(nii: PSU.NiiCTContainer, prompt_list: list, date: dt.datetime, id_cnt: int):
    """
        The size of prompt_list >> size of nii, delete keys 'pid' and 'date', add keys 'id' and 'image'
    :param nii:
    :param prompt_list:
    :param date:
    :param id_cnt:

    Returns:

    """
    final_prompt_list = []

    for nii_path, prompt in zip(np.random.choice(nii[date], len(prompt_list)), prompt_list):
        del prompt['date']
        del prompt['pid']
        prompt['image'] = nii_path
        prompt['id'] = id_cnt
        id_cnt += 1
        final_prompt_list.append(prompt)

    return final_prompt_list, id_cnt


def main(args: argparse.Namespace):
    if len((sep := args.report_file.split(','))) > 1:
        report_list: pd.DataFrame = pd.concat([load_report(sub_file) for sub_file in sep], axis=0, ignore_index=True)
    else:
        report_list: pd.DataFrame = load_report(sep[0])
    key_list = list(report_list.columns)
    if len((sep_ct := args.ct_root.split(','))) > 1:
        nii_list: list[PSU.NiiCTContainer] = list()
        for _ct_root in sep_ct:
            nii_list.extend(load_nii(_ct_root))
    else:
        nii_list: list[PSU.NiiCTContainer] = load_nii(sep_ct[0])

    final_prompt_list: list = []
    cnt = 0

    for row in range(len(report_list)):
        report = report_list.iloc[row]
        ccta: PSU.CCTA | None = PSU.build_ccta(report, args.prompt_template_path)
        
        if ccta is None:  # Because of report format got error
            print(f'CCTA-{row}| got something wrong')
            continue
        if ccta not in nii_list:  # The corresponding CT file was not found.
            print(f'CCTA-{row}| Not coressponding nii found')
            continue
        nii = nii_list[nii_list.index(ccta)]
        num_prompt = nii.len(ccta.check_date)
        print(f'CCTA-{row}| Produce # of Prompt: {num_prompt}')
        prompt_list = ccta.get_prompt(num_prompt)
        pack = merge_path_and_prompt(nii, prompt_list, ccta.check_date, cnt)
        final_prompt_list.extend(pack[0])
        cnt = pack[1]

    with open(args.json_path, 'w+') as jout:
        json.dump(final_prompt_list, jout)
    return None


if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
