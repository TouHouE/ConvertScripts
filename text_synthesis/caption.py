"""
Notice: If a caption is store with prefix, this is raw caption, it should be doing process in the next step.
"""
import argparse
import os
import json
from time import sleep
from concurrent.futures import ProcessPoolExecutor, as_completed
from os.path import join, exists

from google.api_core.exceptions import ResourceExhausted, InternalServerError
from tqdm.auto import tqdm
import pandas as pd
from google import generativeai as genai

KEY_MAP = {}

def get_key():
    for key_value, is_out in KEY_MAP.items():
        if is_out:
            continue
        KEY_MAP[key_value] = True
        return key_value

    for key in KEY_MAP.keys():
        KEY_MAP[key] = False
    sleep(1)
    return get_key()


def loading_token(args):
    with open(args.token_path, 'r', encoding='utf-8') as loader:
        imported_token_list = json.load(loader)
    for new_token in imported_token_list:
        KEY_MAP[new_token] = False


def load_template(args) -> str:
    with open(args.template_path, 'r', encoding='utf-8') as loader:
        return loader.read()

def asking(content, args, retry_timer=0, max_retry=5):
    if retry_timer >= max_retry:
        return None
    genai.configure(api_key=get_key())
    model = genai.GenerativeModel(args.gemini_model_name)
    try:
        response = model.generate_content(content)
    except ResourceExhausted:
        sleep(1)
        return asking(content)
    except InternalServerError:
        sleep(1)
        return asking(content)


    return response.text


def worker(report, temp, pid, ages, gender, args):    
    dst_path = join(args.cur_root, args.cache_dir, f'{pid}.json')
    no_report_file = join(args.cur_root, args.failed_dir, 'new_data_err.txt')
    gemini_error_file = join(args.cur_root, args.failed_dir, f'{args.mode}_gemini_error.txt')
    if os.path.exists(dst_path) or os.path.exists(dst_path.replace(".json", ".txt")):
        return
    if str(report).__len__() < 10:
        with open(no_report_file, 'a+') as out:
            out.write(f'No Report       | {pid}\n')
            return None
    jpack = asking(temp.replace("[INSERT_CHECKLIST]", report).replace("[INSERT_AGE]", str(ages)).replace("[INSERT_GENDER]", gender))    
    if jpack is None:
        with open(gemini_error_file, 'a+') as out:
            out.write(f"Generate Failed | {pid}\n")
        return None
    
    jobj = jpack.replace('```json', '').replace('```', '')
    try:
        jobj = json.loads(jobj)
        with open(dst_path, 'w+', encoding='utf-8') as jout:
            json.dump(jobj, jout, indent=2)
    except json.decoder.JSONDecodeError:
        with open(dst_path.replace('.json', '.txt'), 'w+', encoding='utf-8') as writer:
            writer.write(jobj)
    return jobj


def main(args):
    template = load_template(args)
    if args.mode == 'taipei':
        df = pd.read_excel(args.report_path, sheet_name='台大病歷資料')    
        report = df.loc[0, '報告內容']
        report_key = '報告內容'
        pid_key = '編號'
        age_key = '年紀'
        gender_label = '女'
    else:
        df = pd.read_excel(args.report_path)
        report_key = 'CCTA 報告'
        pid_key = '病歷號'
        age_key = '年齡'
        gender_label = 'F'
    # report = df.loc[0, ]

    pbar = tqdm(total=df.shape[0])

    with ProcessPoolExecutor(args.num_worker) as pooler:
        rep_list = list()
        for i in range(df.shape[0]):
            report = df.loc[i, report_key]
            pid = df.loc[i, pid_key]
            if pid == 'CVAI-1482':
                ages = 63
            else:
                ages = int(df.loc[i, age_key])
            gender = 'female' if df.loc[i, '性別'].lower() == gender_label.lower() else 'male'
            rep_list.append(pooler.submit(worker, report, template, pid, ages, gender))
        final_answer = list()
        done_cnt = 0
        for rep in as_completed(rep_list):
            pbar.update(1)
            done_cnt += 1
            print(f'Progress:{done_cnt}/{df.shape[0]}')
            if (jobj := rep.result()) is None:
                continue
            final_answer.append(jobj)
    with open(join(args.cur_root, 'caption.json'), 'w+') as jout:
        json.dump(jout, final_answer, indent=2)
    print(f'All done, {len(final_answer)} reports are generated.\nThe result is saved in {join(args.cur_root, "caption.json")}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='taipei', choices=['taipei', 'hsinchu'], help='Specify the hospital where the data was collected.')
    parser.add_argument('--report_path', type=str, required=True, help='Path to the report file.')
    parser.add_argument('--template_path', type=str, required=True, default='res/caption/template.txt', help='Path to the template file.')
    parser.add_argument('--token_path', type=str, required=True, help='Path to the token file.')
    parser.add_argument('--gemini_model_name', type=str, default='gemini-2.5-flash-preview-04-17', help='Name of the Gemini model to use.')
    parser.add_argument('--num_worker', type=int, default=3, help='Number of worker processes to use for generating captions.')

    parser.add_argument('--output_dir', type=str, default='output_dir', help='Directory to save the output files.')
    parser.add_argument('--output_mid_dir', type=str, default='caption', help='Intermediate output directory.')
    parser.add_argument('--failed_dir', type=str, default='error', help='Directory to save the failed responses.')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Directory for caching responses.')

    args = parser.parse_args()
    args.cur_root = join(args.output_dir, args.output_mid_dir, args.mode)
    os.makedirs(args.cur_root, exist_ok=True)
    os.makedirs(join(args.cur_root, args.failed_dir), exist_ok=True)
    os.makedirs(join(args.cur_root, args.cache_dir), exist_ok=True)
    loading_token(args)
    print(f'All response will be saved in {args.cur_root}')
    print(f'Using Gemini Model: {args.gemini_model_name}')
    main(args)
