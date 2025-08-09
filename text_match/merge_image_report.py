"""
Before your run this scripts, please make sure you already process `make_taipei_image_table.py` or `make_hsinchu_image_table.py`
"""
import argparse
import os
from os.path import join, exists

import pandas as pd

def load_image_data_as_df(args) -> pd.DataFrame:
    df = pd.read_csv(args.image_table_path)
    if args.mode == 'taipei':
        df['pid'] = df['pid'].apply(lambda _pid: f'CVAI-{_pid:04}')
    return df
    # return pd.read_csv('/home/user/workspace/hsu/ListHsinchu/all_hsinchu.csv')


def load_report_data_as_df(args) -> pd.DataFrame:
    # df = pd.read_excel('./HsinchuCCTA.xlsx')
    if (rtp := args.report_table_path).endswith('.csv'):
        df = pd.read_csv(rtp)
    elif not rtp.endswith('.csv') and args.mode == 'hsinchu':
        df = pd.read_excel(rtp)
    elif not rtp.endswith('.csv') and args.mode == 'taipei':
        df = pd.read_excel(rtp, sheet_name='台大病歷資料')

    check_date = pd.to_datetime(df['檢查日期'])
    df['year'] = check_date.dt.year
    df['month'] = check_date.dt.month    
    df['day'] = check_date.dt.day

    if args.mode == 'hsinchu':
        df.drop(columns=['病人姓名'], inplace=True)
    else:
        print("Setting pid for report table")
        df['pid'] = df['編號'].copy(deep=True)
    return df


def get_matched_df(image_pool_df, report_pool_df):
    image_pool_df['key'] = 1
    report_pool_df['key'] = 1
    for key in ['year', 'month', 'day']:
        image_pool_df[key] = pd.to_numeric(image_pool_df[key], errors='coerce')
    image_pool_df.rename(
        columns={
            key: f'{key}_x' for key in ['pid', 'year', 'month', 'day']
        }, inplace=True
    )
    report_pool_df.rename(
        columns={
            key: f'{key}_y' for key in ['pid', 'year', 'month', 'day']
        }, inplace=True
    )

    cross_joined_df = pd.merge(image_pool_df, report_pool_df, on='key').drop('key', axis=1)
    merged_df = cross_joined_df[
        (cross_joined_df['pid_x'] == cross_joined_df['pid_y']) 
        #   & (
        #     (
        #         cross_joined_df['year_x'] > cross_joined_df['year_y']
        #     ) | (   # Year is equal, but Month is different
        #         (cross_joined_df['year_x'] == cross_joined_df['year_y']) & 
        #         (cross_joined_df['month_x'] > cross_joined_df['month_y'])
        #     ) | (   # Year is equal, Month also equal
        #         (cross_joined_df['year_x'] == cross_joined_df['year_y']) & 
        #         (cross_joined_df['month_x'] == cross_joined_df['month_y']) & 
        #         (cross_joined_df['day_x'] >= cross_joined_df['day_y'])
        #     )
        # )
    ]

    # Clean up the column names for a clearer final output.
    merged_df.rename(columns={
        'year_x': 'image_year', 'month_x': 'image_month', 'day_x': 'image_day',
        'year_y': 'report_year', 'month_y': 'report_month', 'day_y': 'report_day'
    }, inplace=True)
    merged_df.drop(columns=['pid_y'], inplace=True)
    merged_df.rename(columns={'pid_x': 'pid'}, inplace=True)
    return merged_df

def main(args):
    image_pool_df = load_image_data_as_df(args)
    report_pool_df = load_report_data_as_df(args)

    # merged_df = pd.merge(image_pool_df, report_pool_df, on=['year', 'month', 'day'], how='inner')
    merged_df = get_matched_df(image_pool_df, report_pool_df)
    os.makedirs(args.output_dir, exist_ok=True)
    if not (oname := args.output_name).endswith('.csv'):
        oname = f'{oname}.csv'
    
    merged_df.to_csv(join(args.output_dir, oname))
    # breakpoint()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['hsinchu', 'taipei'], type=str, default='taipei')
    parser.add_argument('--image_table_path', type=str, required=True)
    parser.add_argument('--report_table_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output_dir')
    parser.add_argument('--output_name', type=str, default='merge.csv')
    main(parser.parse_args())


