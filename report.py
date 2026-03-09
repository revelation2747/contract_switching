import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
import os
import numpy as np
from calculators import (
    get_pure_calendar, 
    asymmetric_binary_logloss, 
    make_features_logic, 
    calculate_full_performance, 
    plot_results
)

warnings.filterwarnings('ignore')

def run_main_pipeline(data_path="contract_volume.parquet", split_date_str='2025-06-30'):
    print(">>> Step 1: Loading and Preprocessing Data...")
    df_raw = pd.read_parquet(data_path)
    df_raw['date_dt'] = pd.to_datetime(df_raw['date'])
    
    # 基础排名
    df_raw['rank'] = df_raw.groupby(['symbol_code', 'date_dt'])['volume'].rank(ascending=False, method='first')
    
    cal = get_pure_calendar()
    trade_dates_list = cal['date_dt'].tolist()
    df = df_raw.merge(cal[['date_dt', 'is_eve_of_long_holiday', 'is_before_holiday']], on='date_dt', how='inner')
    
    print(">>> Step 2: Feature Engineering...")
    df = make_features_logic(df)
    rank_t5 = df.groupby(['symbol_code', 'delivery_code'])['rank'].shift(-5)
    df['is_main_t5'] = np.where(
        (rank_t5.isna()) | (rank_t5 <= 0), 
        np.nan, 
        (rank_t5 == 1).astype(float)
    )
    drop_idx = []
    for sym in df['symbol_code'].unique():
        sym_df = df[df['symbol_code'] == sym].sort_values('date_dt')
        if not sym_df.empty:
            drop_idx.extend(df[(df['symbol_code'] == sym) & (df['delivery_code'] == sym_df['delivery_code'].iloc[0])].index)
    df = df.drop(index=drop_idx)

    feature_cols = ['is_eve_of_long_holiday', 'is_before_holiday', 'volume_share']
    for k in [3, 5, 10]:
        feature_cols += [f'share_ma_{k}', f'share_lag_{k}', f'share_v_{k}', f'share_a_{k}']

    split_date = pd.to_datetime(split_date_str)
    
    train_df = df[df['date_dt'] < split_date].dropna(subset=['is_main_t5'] + feature_cols).copy()

    test_df = df[df['date_dt'] >= split_date].copy()

    print(f">>> Step 3: Training Model (Data before {split_date.date()})...")
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05, 
        objective=asymmetric_binary_logloss, 
        verbose=-1
    )
    clf.fit(train_df[feature_cols], train_df['is_main_t5'])

    print(">>> Step 4: Running Inference...")
    raw_scores = clf.predict(test_df[feature_cols], raw_score=True)
    test_df['score'] = raw_scores
    test_df['prob'] = 1.0 / (1.0 + np.exp(-raw_scores))
    
    test_df['pred_rank'] = test_df.groupby(['symbol_code', 'date_dt'])['score'].rank(ascending=False, method='first')
    
    test_df['is_alert'] = (test_df['pred_rank'] == 1) & (test_df['rank'] != 1) & (test_df['prob'] > 0.4)

    print(f">>> Step 5: Auditing Performance (Total Alerts: {test_df['is_alert'].sum()})...")
    stats_df, all_gaps, audit_df, matched_dates = calculate_full_performance(
        test_df, split_date, trade_dates_list
    )

    print(">>> Step 6: Visualizing Results...")
    save_dir = "output_results" 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plot_results(test_df, stats_df, all_gaps, matched_dates, split_date, trade_dates_list, n=80, save_path=save_dir)

    return stats_df, audit_df

if __name__ == "__main__":
    final_stats, final_audit = run_main_pipeline()