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
    # 标签：T+5 是否为主力
    df['is_main_t5'] = df.groupby(['symbol_code', 'delivery_code'])['rank'].shift(-5).apply(lambda x: 1 if x == 1 else 0)

    # 排除初始合约干扰
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
    
    # 训练集：严格 split_date 之前
    train_df = df[df['date_dt'] < split_date].dropna(subset=['is_main_t5', 'share_ma_10']).copy()
    
    # 测试集：为了绘图和统计纯净，直接从 split_date 开始
    # 如果需要计算特征的滞后项，make_features_logic 已经在全量 df 上做过了，这里可以直接截断
    test_df = df[df['date_dt'] >= split_date].copy()

    print(f">>> Step 3: Training Model (Data before {split_date.date()})...")
    clf = lgb.LGBMClassifier(
        n_estimators=150, # 配合不对称损失，建议不需要太深
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