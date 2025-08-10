import argparse
import os
import re
import json
from typing import Final
from os.path import join, exists

import pandas as pd
import jsonlines as jsonl
import google.generativeai as genai
from tqdm.auto import tqdm

from .api_caller import GeminiCaller

Model: GeminiCaller

def get_key():
    for key_value, is_out in KEY_MAP.items():
        if is_out:
            continue
        KEY_MAP[key_value] = True
        return key_value
    return input('Out of API Key, Please input a new API Key:')


def make_sure_required_folder_exists(args):
    os.makdedirs(
        join(args.cur_root, args.ok_folder), exist_ok=True
    )
    os.makedirs(
        join(args.cur_root, args.irr_folder), exist_ok=True
    )    


def load_json(path: str):
    content = list()
    with open(path, 'r', encoding='utf-8') as jin:
        if path.endswith('.jsonl'):
            content = [json.loads(line) for line in jin.readlines()]
        else:
            content = json.load(jin)
    return content

def load():
    with open(r"C:\Users\hsuwi\Downloads\v5-lite-20241124.md", 'r') as loader:
        return '\n'.join(loader.readlines())
sysmsg = load()

# Insert Your gemini token into here with following format:
# Token: False
KEY_MAP: dict[str, bool] = {}    

def declare_model(args):
    global Model
    Model = GeminiCaller(args)
    Model.set_User_Template("""Please generate a simulation 60 arounds visual question answering conversations with a given CCTA check list, ignoring information that cannot be obtained by vision, like imaging device. please do not mention any word like checklist, text report, etc.
        You can modify the answer to your liking, BUT make sure the answer still correct and helpful.
        Each conversation should following JSON format. Using a list to package all conversations.
        Please adding a new key "topic" that store what the question is about. Like organ, stenosis, etc.
        Do not make any annotation in the conversations json.
        Example:
        ```json
        [
            {"question": "What is the name of the organ?", "answer": "Lung", "topic": "Organ"},
            {"question": "What is the name of the stenosis?", "answer": "Lung stenosis", "topic": "Stenosis"},
        ]
        ```
        Here is a CCTA Check list:
        
        {}
        """)


def save_issue_content(content: str, cur_patient: str, args):
    repeat_time = 0
    irr_txt_name = args.irr_suffix.format(f"{cur_patient}-{repeat_time}")
    
    while exists(irr_txt_path):
        irr_txt_path = join(args.irr_folder, irr_txt_name)
        repeat_time += 1

    with open(irr_txt_path, 'w+', encoding='utf-8') as file_writer:
        file_writer.write(content)


def jsonize_content(raw_content: str, cur_patient: str, args):
    json_str = raw_content.split('```json')[-1].split("```")[0].strip('`\n')
    jsonizalble = True
    jobj: dict | None = None
    try:
        jobj = json.loads(json_str)
    except json.decoder.JSONDecodeError as e:
        jsonizalble = False
    
    if not isinstance(jobj, list) or not jsonizalble:
        save_issue_content(raw_content, cur_patient, args)
        
    return jobj


def main(args):
    global Model
    Model = GeminiCaller(args)

    assert args.mode in ['taipei', 'hsinchu'], f"args.mode must belong to [taipei, hsinchu], but given {args.mode}"
    if args.mode == 'taipei':
        df = pd.read_excel(args.report_path, sheet_name='台大病歷資料')
        report_key = '報告內容'
        pid_key = '病歷號'
    else:
        df = pd.read_excel(args.report_path)
        report_key = 'CCTA 報告'
        pid_key = '編號'    
    areport = df[report_key]
    patient_ids = df[pid_key]    

    done_patient = []    
    # Those are .txt file
    unjsonize_patient = ['-'.join(_ok.split('-')[:2]) for _ok in os.listdir(join(args.cur_root, args.irr_folder))]    

    if (had_jsonized_patient := exists(args.cur_root, args.inst_name)):
        jsonized_patient = load_json(join(args.cur_root, args.inst_name))
        done_patient = [_pack['patient_id'] for _pack in jsonized_patient]
        done_patient = list(set(done_patient))
    if len(done_patient) == 0:
        had_jsonized_patient = False
    print(f'Successfully load {done_patient} history.')    
        
    done_patient.extend(unjsonize_patient)
    done_patient = list(set(done_patient))
    mode, vqa_count = ('a', jsonized_patient[-1]['id'] + 1) if had_jsonized_patient else ('w', 0)    
    JsonWriter = jsonl.open(join(args.cur_root, args.inst_name), mode)    
    patient_counter = 0    
    
    while patient_counter < areport.shape[0]:
        cur_report = areport.iloc[patient_counter]
        cur_patient = patient_ids.iloc[patient_counter]
        patient_counter += 1
        if cur_patient in done_patient: # Do not repeat process
            continue    
        
        response = Model(cur_report)
        if response is None:
            patient_counter -= 1
            continue

        jobj = jsonize_content(response.text, cur_patient, args)
        desc = f'Report Progress:[{patient_counter}/{areport.shape[0]}]|{cur_patient}'
        
        for _pack in (bar := tqdm(jobj, total=len(jobj), desc=desc)):
            _pack['patient_id'] = cur_patient
            _pack['id'] = vqa_count
            vqa_count += 1
            JsonWriter.write(_pack)

        with open((args.cur_root, args.ok_folder, args.ok_suffix.format(cur_patient)), 'w+') as writer:
            json.dump(jobj, writer)
        JsonWriter.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['taipei', 'hsinchu'], default='taipei', help='Specify the hospital where the data was collected.')
    parser.add_argument('--report_path', type=str, required=True, help='Path to the report file, should be an excel file.')
    parser.add_argument('--gemini_model_name', type=str, default="gemini-1.5-flash")
    parser.add_argument('--system_message_file', type=str, required=True, default='res/text_synthesis/system_message.txt')
    parser.add_argument('--token_path', type=str, required=True)
    parser.add_argument('--newer_token_json', type=str, required=False)

    parser.add_argument('--output_dir', type=str, default='output_dir')
    parser.add_argument('--output_mid_dir', type=str, default='stage1')
    parser.add_argument('--inst_name', type=str, default='instruction.jsonl')

    parser.add_argument('--ok_folder', type=str, default='ok')
    parser.add_argument('--ok_suffix', type=str, default='{}_vqa.json')

    parser.add_argument('--irr_folder', type=str, default='irregular')
    parser.add_argument('--irr_suffix', type=str, default='{}-not_suitable2json.txt')
    args = parser.parse_args()
    args.cur_root = join(
        args.output_dir, args.output_mid_dir, args.mode
    )
    make_sure_required_folder_exists(args)
    main(args)