import re
import os
import jsonlines as jsonl
import json
import datetime as dt

import numpy as np
import pandas as pd




def pandas_timestamp_to_datetime(pandas_timestamp, timezone=dt.timezone.utc):
    """Converts a pandas Timestamp object to a datetime.datetime object.

    Args:
        pandas_timestamp: The pandas Timestamp object.
        timezone: The timezone to use for the resulting datetime object.
                  Defaults to UTC. Crucial for correct conversions.

    Returns:
        A datetime.datetime object, or None if the conversion fails.
    """
    try:
        # pandas Timestamps can have timezone information, so handle that first.
        if pandas_timestamp.tz is not None:  # If the pandas Timestamp has a timezone
            if timezone != pandas_timestamp.tz:  # Convert to the given timezone if necessary.
                datetime_object = pandas_timestamp.tz_convert(timezone).to_pydatetime()
            else:
                datetime_object = pandas_timestamp.to_pydatetime() # No conversion needed
        else: # If the pandas Timestamp is naive (no timezone info)
            datetime_object = pandas_timestamp.to_pydatetime().replace(tzinfo=timezone) # Add the timezone

        return datetime_object
    except (ValueError, AttributeError, TypeError) as e: # Catch potential errors
        print(f"Error converting pandas Timestamp: {e}") # Print the specific error
        return None


def load_json(path):
    with open(path, 'r') as jout:
        return json.load(jout)


def load_table(branch_name) -> dict[str, dt.datetime]:
    """
    contains two keys, `patient_id`, `date`
    """
    taipei_table = '/home/hsu/PycharmProjects/scripts/res/Total_CT_Receive.xlsx'
    hsinchu_table = '/mnt/cardiac_usb_b/total.xlsx'
    if branch_name.lower() == 'taipei':
        table = pd.read_excel(taipei_table, sheet_name='台大病歷資料')
        to_dt_func = pandas_timestamp_to_datetime
    else:
        table = pd.read_excel(hsinchu_table)
        to_dt_func = lambda _str_dt: dt.datetime.strptime(_str_dt, "%Y/%m/%d %H:%M:%S")
    final_table = dict()
    for row in range(table.shape[0]):
        row_series = table.iloc[row]
        pid = row_series.get('病歷號碼', table.get('編號'))
        final_table[pid.lower()] = to_dt_func(row_series.get('檢查日期'))

    return final_table
    

def get_convs_loader(branch) -> jsonl.Reader:
    branch = 'hsinchu' if branch.lower() == 'hsinchu' else 'taipei'
    root = '/mnt/cardiac_usb_b/text_dataset'
    return jsonl.Reader(os.listdir(root, f'{branch}_instruction.jsonl'), 'r')


def make_patient2image_pool_mapper() -> dict[str, list[str | dict[str, str | int | float | list[dict]]]]:  
    """
        Using patient id to get all possible pack pool.
    """  
    if os.path.exists('/home/hsu/PycharmProjects/ConvertScripts/res/full_pool.json'):
        return load_json('/home/hsu/PycharmProjects/ConvertScripts/res/full_pool.json')
    all_ds_folder = ['/mnt/cardiac_usb_a/Taipei_502/meta', '/mnt/cardiac_usb_a/Taipei_2897/meta']
    p2ip = dict()

    for cur_folder in all_ds_folder:    # Iteration 502 and 2897
        depack = load_json(os.path.join(cur_folder, 'Deduplicate_PLQ_Table.json'))

        for pack in depack: # Iteration whole Deduplicate json file.
            cur_pid = pack['pid']
            if p2ip.get(cur_pid) is None:
                p2ip[cur_pid] = list()
            if pack.get('date') is None:
                pack['date'] = 
            p2ip[cur_pid].append(pack)
    # Taipei-Main Branch is DONE
    p2ip.update(load_json('/home/hsu/PycharmProjects/ConvertScripts/res/normal_image.json'))  
    with open('/home/hsu/PycharmProjects/ConvertScripts/res/full_pool.json', 'w+') as jwriter:
        json.dump(p2ip, jwriter, indent=2)
    return p2ip


def _refind_date_image_path(nii_path):
    dt_fmt = "%Y%m%d%H%M%S"
    nii_path = nii_path.replace('\\', '/').lstrip('/')
    all_possible = nii_path.split('/')

    try:
        date = dt.datetime.strptime(all_possible[-2], dt_fmt)
    except Exception as e:
        date = dt.datetime.strptime(all_possible[-3], dt_fmt)
    series = all_possible[-1].rstrip('.nii.gz')
    pack = {
        'image': nii_path,
        'date': date,
        'snum': series
    }
    return pack


def _find_pack_by_date(pack_pool, target_date):
    # target_date = ...
    for pack in pack_pool:
        if pack['date'] == target_date:
            return pack
    return None


def main():
    dst_path = '...'
    p2ip = make_patient2image_pool_mapper()
    phase = 'taipei'
    ccta_date_pid_table = load_table(phase)
    JsonlReader = get_convs_loader(phase)    
    applyed_image_path_convs = list()

    for one_convs in JsonlReader.read():
        pid = one_convs['image']
        if len((pid_list := pid.split('-'))) > 1:
            pid = pid_list[-1]        
        path_pool: list[str | os.PathLike] = p2ip[pid.lower()]
        possible_date = ccta_date_pid_table[pid.lower()]
        if re.fullmatch(r'[0-9]+', pid) is None:    # Make Sure Hsinchu branch data can fix Taipei Main branch data format.
            path_pool = list(map(_refind_date_image_path, path_pool))
        if possible_date is None:
            one_convs.update(path_pool[np.random.choice(len(path_pool))])
        else:
            one_convs.update(path_pool)
        applyed_image_path_convs.append(one_convs['image'])
    
    JsonlWriter = jsonl.open(f'./res/{phase}_inst_real_image.jsonl', 'a+')

    JsonlWriter.write_all(applyed_image_path_convs)
    # for one_convs in applyed_image_path_convs:
    #     JsonlWriter.

    

    
    pass


if __name__ == '__main__':
    main()