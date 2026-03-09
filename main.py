import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from pyarrow import fs
import pyarrow.parquet as pq
from tqdm import tqdm
from calculators import (get_hdfs_files, finalize_symbol_parsing_v5, 
                        make_features_logic, get_pure_calendar, asymmetric_binary_logloss)

PARQUET_PATH = 'contract_volume.parquet'
OLD_FOLDER = '/user/zli/comm/chinese_future/depth_v0.4'
NEW_FOLDER = '/user/zli/comm/data/depth_data/hlsh01'
CUTOFF_DATE = pd.Timestamp('2025-09-30')
START_DATE_DEFAULT = pd.Timestamp('2022-01-01')

def run_daily_inference():
    # --- 1. 数据同步逻辑 ---
    hdfs = fs.HadoopFileSystem('hdfs://ftxz-hadoop', user='zli')
    if not os.path.exists(PARQUET_PATH):
        latest_date = START_DATE_DEFAULT - pd.Timedelta(days=1)
        hist_df = pd.DataFrame()
    else:
        hist_df = pd.read_parquet(PARQUET_PATH)
        hist_df['date_dt'] = pd.to_datetime(hist_df['date'])
        latest_date = hist_df['date_dt'].max()

    new_files = get_hdfs_files(hdfs, OLD_FOLDER, latest_date + pd.Timedelta(days=1), CUTOFF_DATE)
    new_files += get_hdfs_files(hdfs, NEW_FOLDER, max(latest_date + pd.Timedelta(days=1), CUTOFF_DATE + pd.Timedelta(days=1)))
    new_files.sort(key=lambda x: x[1])

    if new_files:
        new_data_list = []
        for path, f_date in tqdm(new_files, desc="Syncing Data"):
            try:
                with hdfs.open_input_file(path) as f:
                    tmp = pq.read_table(f, columns=['symbol_str', 'volume']).to_pandas()
                day_sum = tmp.groupby('symbol_str').agg({'volume': 'last'}).reset_index()
                day_sum['date'] = f_date.strftime('%Y-%m-%d')
                new_data_list.append(day_sum)
            except:
                continue
        if new_data_list:
            inc_df = pd.concat(new_data_list, ignore_index=True)
            inc_df = finalize_symbol_parsing_v5(inc_df)
            inc_df['date_dt'] = pd.to_datetime(inc_df['date'])
            hist_df = pd.concat([hist_df, inc_df], ignore_index=True).drop_duplicates(subset=['symbol_code', 'delivery_code', 'date'], keep='last')
            hist_df.to_parquet(PARQUET_PATH, index=False)
    cal = get_pure_calendar()
    df = make_features_logic(hist_df)
    df = df.merge(cal[['date_dt', 'is_eve_of_long_holiday', 'is_before_holiday']], on='date_dt', how='left')
    
    df['rank'] = df.groupby(['symbol_code', 'date_dt'])['volume'].rank(ascending=False, method='first')
    rank_t5 = df.groupby(['symbol_code', 'delivery_code'])['rank'].shift(-5)
    target_val_t5 = df.groupby(['symbol_code', 'delivery_code'])['volume_share'].shift(-5)
    
    df['label'] = np.where(
        (rank_t5.isna()) | (rank_t5 <= 0),
        np.nan,
        ((rank_t5 == 1) & (target_val_t5 > 0.1)).astype(float)
    )
    
    feature_cols = ['volume_share', 'is_eve_of_long_holiday', 'is_before_holiday'] + \
                   [f'share_ma_{k}' for k in [3,5,10]] + [f'share_lag_{k}' for k in [3,5,10]] + \
                   [f'share_v_{k}' for k in [3,5,10]] + [f'share_a_{k}' for k in [3,5,10]]
    
    today_dt = df['date_dt'].max()
    train_df = df.dropna(subset=['label'] + feature_cols).copy()
    train_df = train_df[train_df['date_dt'] < today_dt]
    
    clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, verbosity=-1)
    clf.set_params(objective=asymmetric_binary_logloss)
    clf.fit(train_df[feature_cols], train_df['label'])

    curr = df[df['date_dt'] == today_dt].copy()
    if curr.empty:
        print(f"Warning: No data found for today ({today_dt.date()})")
        return

    curr['prob'] = 1.0 / (1.0 + np.exp(-clf.predict(curr[feature_cols], raw_score=True)))
    
    curr['curr_rank'] = curr.groupby('symbol_code')['volume'].rank(ascending=False, method='first')
    curr['pred_rank'] = curr.groupby('symbol_code')['prob'].rank(ascending=False, method='first')
    
    curr['is_alert'] = (curr['pred_rank'] == 1) & (curr['curr_rank'] != 1) & (curr['prob'] > 0.5)

    signals = curr[curr['is_alert']].copy()
    majors = curr[curr['curr_rank'] == 1][['symbol_code', 'delivery_code', 'volume']].rename(
        columns={'delivery_code': 'Current_Major', 'volume': 'Current_Vol'}
    )
    
    res = signals.merge(majors, on='symbol_code')
    res = res.rename(columns={
        'delivery_code': 'Target_Major', 
        'volume': 'Target_Vol', 
        'prob': 'Confidence'
    })
    
    res = res[['symbol_code', 'Current_Major', 'Target_Major', 'Confidence', 'Current_Vol', 'Target_Vol']]
    res = res.sort_values('Confidence', ascending=False)

    print(f"\n[DAILY SIGNAL REPORT] | {today_dt.date()}")
    if res.empty:
        print("No switch signals triggered today (Confidence <= 0.5 or no candidate).")
    else:
        print(res.to_string(index=False))

if __name__ == "__main__":
    run_daily_inference()