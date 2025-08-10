import argparse
import json
import os
import re
import time
import requests as req
from os.path import join, exists

import pandas as pd
import jsonlines as jsonl

from google.genai.errors import ClientError
from tqdm.auto import tqdm

from api_caller import GeminiCallerV2


Model: GeminiCallerV2


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as jin:
        return json.load(jin)


def declare_client(args):
    global Model
    Model = GeminiCallerV2(args)


def figure_out_convs(pack):
    """ Figure out the conversation format.
    :param pack: The pack to figure out.
    :return: The original query and answer."""
    # breakpoint()
    cand0 = pack.get('conversations', None)
    cand1 = pack.get('conversation', None)  # the question-answer pair is directly stored with keys - 'question' and 'answer'
    is_store_with_chat_mode: bool
    convs = None
    
    if cand0 is None:
        is_store_with_chat_mode = False
        convs = cand1
    elif cand1 is None:
        is_store_with_chat_mode = True
        convs = cand0

    if convs is not None:
        if is_store_with_chat_mode:
            org_query = convs[0].get('content', convs[0]['value'])
            org_answer = convs[1].get('content', convs[1]['value'])
            return org_query, org_answer
        else:
            return convs['question'], convs['answer']
        pass
    q0 = cand0[0].get('content', cand0[0]['value'])
    a0 = cand0[1].get('content', cand0[1]['value'])

    if isinstance(cand1, list):
        cand1 = cand1[0]
    q1 = cand1['question']
    a1 = cand1['answer']
    is_cand0_none = False
    is_cand1_none = False

    is_cand0_none = q0 is None or a0 is None
    is_cand1_none = a1 is None or q1 is None

    if not is_cand0_none:
        return q0, a0
    elif not is_cand1_none:
        return q1, a1
    return None


def jsonize_content(text: str, pack, org_query, org_answer, args):
    _text = text.split("```json")[-1]
    try:
        _text = _text.split("```json")[-1]
        # pack['Should drop'] = pack['Should drop']
        mpack = json.loads(_text.strip().strip("`").strip('json').strip())
        pack['Should drop'] = mpack['Should drop']
        if mpack['Should drop']:
            pack['Drop'] = True

        if mpack['Modified Answer Type'] is None:
            answer_type = mpack['Answer Type']
            answer = org_answer
        else:
            answer_type = mpack['Modified Answer Type']
            answer = mpack['Modified Answer']
        
        if mpack['Modified Question Type'] is None:
            qt = mpack['Question Topic']
            query = org_query
        else:
            qt = mpack['Modified Question Type']
            query = mpack['Modified Question']
        
        organ = mpack['Relate Organ digits']
        pack['Question Topic'] = qt
        pack['Answer Type'] = answer_type
        pack['organ'] = organ
        pack['oconvs'] = pack['conversations']
        pack['conversations'] = [
            {'from': 'human', 'value': query},
            {'from': 'gpt', 'value': answer}
        ]
        
    except Exception as e:  # Try to avoid potential json format error.
        # print(e.args)
        _text = text.split("```json")[-1]
        text = _text.strip().strip("`").strip('json').strip()
        pack['Instruction'] = text
    return pack

def _generate_response(pack, args):  
    # global ALL_GEMINI_TOKEN 
    try:
        tmp = figure_out_convs(pack)
    except Exception as e:
        print(f'ID: {pack["ID"]} | Error: {e}')
        print(f'{json.dumps(pack, indent=2)}')
        exit()
        return None
    if tmp is None:
        return None
    org_query, org_answer = tmp 
    # org_query = pack['conversations'][0]['value']
    org_query = re.sub(r'\n{0,1}<image>\n{0,1}', "", org_query)    

    ID = pack['ID']
    cache_path = join(args.cur_root, args.cache_dir, f'{ID}.json')
            
    response = Model(org_query, org_answer)
    pack = jsonize_content(response.text, pack, org_query, org_answer, args)
    
    with open(cache_path, 'w+') as cacher:
        json.dump(pack, cacher, indent=2)

    return pack


def drop_cache(full_ds, args):
    cache_path = join(args.cur_root, args.cache_dir)
    if not exists(cache_path):
        return full_ds
    all_cache = os.listdir(cache_path)
    all_cache = [name.split('.')[0] for name in all_cache]
    return list(filter(lambda _pack: str(_pack['ID']) not in all_cache, full_ds))

def concur_main(args):
    declare_client(args)
    
    dataset = load_json(args.dataset_path)
    dataset = drop_cache(dataset, args)
    final_train = list()
    
    for pack in tqdm(dataset, total=len(dataset)):
        new_pack = _generate_response(pack, args)
        if new_pack is None:
            continue
        final_train.append(
            new_pack
        )
    
    datasetname = os.path.basename(args.dataset_path)

    with open(join(args.cur_root, args.integrate_postfix.format(datasetname)), 'w+') as jout:
        json.dump(final_train, jout, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file.')
    parser.add_argument('--gemini_model_name', type=str, default="gemini-1.5-flash", help='Name of the Gemini model to use.')    
    parser.add_argument('--system_message_file', type=str, required=True, help='Path to the system message file.')
    parser.add_argument('--token_path', type=str, required=True, help='Path to the token file.')

    parser.add_argument('--output_dir', type=str, default='output_dir', help='Directory to save the output files.')
    parser.add_argument('--output_mid_dir', type=str, default='stage2', help='Intermediate output directory.')    
    parser.add_argument('--cache_dir', type=str, default='tmp_nv', help='Directory for caching responses.')
    parser.add_argument('--integrate_postfix', type=str, default='clean_{}', help='Name of the integrated dataset file.')
    
    args = parser.parse_args()
    
    args.cur_root = join(args.output_dir, args.output_mid_dir)
    os.makedirs(args.cur_root, exist_ok=True)
    os.makedirs(join(args.cur_root, args.cache_dir), exist_ok=True)

    concur_main(args)