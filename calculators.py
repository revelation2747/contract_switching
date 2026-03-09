import pandas as pd
import numpy as np
import os
from pyarrow import fs
import pyarrow.parquet as pq
from tqdm import tqdm
import lightgbm as lgb
from chinese_calendar import is_workday
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import warnings

warnings.filterwarnings('ignore')

def get_pure_calendar(start='2022-01-01', end='2026-12-31'):
    all_days = pd.date_range(start, end)
    trade_days = [d for d in all_days if is_workday(d) and d.weekday() < 5]
    cal = pd.DataFrame({'date_dt': trade_days}).sort_values('date_dt')
    cal['next_date'] = cal['date_dt'].shift(-1)
    cal['is_eve_of_long_holiday'] = ((cal['next_date'] - cal['date_dt']).dt.days > 3).astype(int)
    cal['is_before_holiday'] = ((cal['next_date'] - cal['date_dt']).dt.days > 1).astype(int)
    return cal

def asymmetric_binary_logloss(y_true, y_pred):
    probs = 1.0 / (1.0 + np.exp(-y_pred))
    grad = probs - y_true
    hess = probs * (1.0 - probs)
    weight = np.where((y_true == 1) & (probs < 0.6), 500.0, 
                      np.where((y_true == 0) & (probs > 0.3), 50.0, 1.0))
    return grad * weight, hess * weight

def finalize_symbol_parsing_v5(df):
    def get_delivery_map(series):
        valid_series = series.dropna().astype(str)
        if valid_series.empty: return {}
        digits = valid_series.str.extract(r'(\d+)')[0].fillna('')
        def transform(x):
            if not x or len(x) < 3: return x
            return x[-4:] if (len(x) >= 4 and x[-4].isdigit()) else '2' + x[-3:]
        return {c: transform(c) for c in digits.unique() if c}
    s_raw = df['symbol_str'].str.replace(r'[Ff]$', '', regex=True)
    df['symbol_code'] = s_raw.str.replace(r'\d+', '', regex=True).str.upper()
    d_map = get_delivery_map(s_raw)
    df['delivery_code'] = s_raw.str.extract(r'(\d+)')[0].fillna('').map(d_map).fillna('')
    return df

def make_features_logic(df):
    df = df.sort_values(['symbol_code', 'delivery_code', 'date_dt'])
    group_contract = df.groupby(['symbol_code', 'delivery_code'])
    group_daily = df.groupby(['symbol_code', 'date_dt'])
    df['volume_share'] = df['volume'] / (group_daily['volume'].transform('sum') + 1e-9)
    for k in [3, 5, 10]:
        df[f'share_ma_{k}'] = group_contract['volume_share'].transform(lambda x: x.rolling(k).mean())
        df[f'share_lag_{k}'] = group_contract['volume_share'].shift(k)
        df[f'share_v_{k}'] = group_contract['volume_share'].diff(k)
        df[f'share_a_{k}'] = group_contract[f'share_v_{k}'].diff(1)
    return df

def get_hdfs_files(hdfs, folder_path, start_date, end_date=None):
    try:
        selector = fs.FileSelector(folder_path, recursive=True)
        infos = hdfs.get_file_info(selector)
        valid_files = []
        for info in [i for i in infos if i.type == fs.FileType.File]:
            file_name = os.path.basename(info.path)
            try:
                f_date = pd.to_datetime(file_name.split('.')[0], format='%Y%m%d', errors='coerce')
                if pd.isna(f_date) or f_date < start_date: continue
                if end_date and f_date > end_date: continue
                valid_files.append((info.path, f_date))
            except: continue
        return valid_files
    except Exception as e:
        print(f"HDFS Scan Error: {e}")
        return []

def calculate_full_performance(full_test_df, split_date, trade_dates_list):
    summary_list, global_gaps, audit_records, matched_pred_dates = [], [], [], {}
    symbols = full_test_df['symbol_code'].unique()
    for sym in symbols:
        # --- 修复 1: 添加缺失的 sym_sub 定义 ---
        sym_sub = full_test_df[full_test_df['symbol_code'] == sym].sort_values('date_dt').copy()
        
        daily_main = sym_sub[sym_sub['rank'] == 1].drop_duplicates('date_dt').set_index('date_dt')['delivery_code']
        
        # --- 修复 2: 使用 .iloc[1:] 避开 06-30 的伪换月 ---
        diff_mask = (daily_main != daily_main.shift(1))
        real_sw_series = daily_main[diff_mask].iloc[1:] # 丢弃第一行记录
        
        real_events = []
        for d, to_c in real_sw_series.items():
            if d <= split_date: continue # 双重保险
            try:
                idx = daily_main.index.get_loc(d)
                real_events.append({'date': d, 'from': daily_main.iloc[idx-1], 'to': to_c, 'matched': False})
            except: continue
        
        pred_signals = sym_sub[(sym_sub['is_alert']) & (sym_sub['date_dt'] >= split_date)]
        pred_events = []
        for _, row in pred_signals.iterrows():
            curr_m = sym_sub[(sym_sub['date_dt'] == row['date_dt']) & (sym_sub['rank'] == 1)]['delivery_code'].values
            if len(curr_m) > 0:
                pred_events.append({'raw_date': row['date_dt'], 'from': curr_m[0], 'to': row['delivery_code'], 'matched': False})

        local_matched_gaps, sym_matched_raw_dates = [], []
        for p_ev in pred_events:
            targets = [r for r in real_events if r['from'] == p_ev['from'] and r['to'] == p_ev['to']]
            if targets:
                gaps = [(trade_dates_list.index(p_ev['raw_date']) - trade_dates_list.index(r['date'])) for r in targets]
                best_gap = min(gaps, key=abs)
                global_gaps.append(best_gap); p_ev['matched'] = True
                local_matched_gaps.append(best_gap); sym_matched_raw_dates.append(p_ev['raw_date'])
                for r in targets: r['matched'] = True
        
        matched_pred_dates[sym] = sym_matched_raw_dates
        n_real, n_pred = len(real_events), len(pred_events)
        n_matched_real = sum(r['matched'] for r in real_events)
        n_matched_pred = sum(p['matched'] for p in pred_events)

        for r in real_events: audit_records.append([sym, r['date'], r['from'], r['to'], 'REAL', r['matched']])
        for p in pred_events: audit_records.append([sym, p['raw_date'], p['from'], p['to'], 'PREDICT', p['matched']])

        summary_list.append({
            'symbol': sym, 'Real_N': n_real, 'Pred_N': n_pred, 'Matched': n_matched_real,
            'Recall': n_matched_real / n_real if n_real > 0 else 1.0,
            'Precision': n_matched_pred / n_pred if n_pred > 0 else 1.0,
            'False_Alarm': n_pred - n_matched_pred,
            'Avg_Gap': np.mean(local_matched_gaps) if local_matched_gaps else 0
        })

    summary_df = pd.DataFrame(summary_list)
    audit_df = pd.DataFrame(audit_records, columns=['Symbol', 'Date', 'From', 'To', 'Type', 'Is_Matched'])
    
    total_real = summary_df['Real_N'].sum()
    total_pred = summary_df['Pred_N'].sum()
    total_matched = summary_df['Matched'].sum()
    
    print("\n" + "="*45)
    print("CONTRACT-BASED ANALYSIS (NO TIME WINDOW)")
    print(f"Total Real Events: {total_real}")
    print(f"Total Pred Signals: {total_pred}")
    print(f"Recall: {total_matched/total_real:.2%}")
    total_matched_pred = sum(summary_df['Pred_N'] * summary_df['Precision'])
    print(f"Precision: {total_matched_pred/total_pred:.2%}")
    print("="*45 + "\n")
    
    return summary_df, pd.Series(global_gaps), audit_df, matched_pred_dates

def plot_results(full_test_df, stats_df, all_gaps, matched_dates_dict, split_date, trade_dates_list, n=80, save_path=None):
    first_signal_gaps = []
    symbols = full_test_df['symbol_code'].unique()
    for sym in symbols:
        sym_sub = full_test_df[full_test_df['symbol_code'] == sym].sort_values('date_dt').copy()
        daily_main = sym_sub[sym_sub['rank'] == 1].drop_duplicates('date_dt').set_index('date_dt')['delivery_code']
        real_sw = daily_main[daily_main != daily_main.shift(1)].iloc[1:] 
        real_sw = real_sw[real_sw.index > split_date]
        for r_date, to_c in real_sw.items():
            try:
                matched_alerts = sym_sub[(sym_sub['is_alert']) & (sym_sub['delivery_code'] == to_c) & (sym_sub['date_dt'] >= split_date)]
                if not matched_alerts.empty:
                    first_sig_date = matched_alerts['date_dt'].min()
                    gap = trade_dates_list.index(first_sig_date) - (trade_dates_list.index(r_date) - 5)
                    first_signal_gaps.append(gap)
            except: continue

    if save_path:
        fig_dist, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        combined_all_gaps = all_gaps + 5
        bins = np.arange(-15.5, 15.5, 1)
        ax1.hist(combined_all_gaps, bins=bins, color='steelblue', edgecolor='white', alpha=0.7, density=True)
        ax1.axvline(0, color='red', linestyle='--', lw=2, label='T-5 Target')
        ax1.set_title("Distribution of ALL Signals (Zero = T-5 Target Day)")
        ax1.legend()

        ax2.hist(first_signal_gaps, bins=bins, color='seagreen', edgecolor='white', alpha=0.7, density=True)
        ax2.axvline(0, color='red', linestyle='--', lw=2, label='T-5 Target')
        ax2.set_title("Distribution of FIRST Signal per Event (Earliest Alert)")
        ax2.set_xlabel("Days Gap from T-5 (Negative = Earlier, Positive = Later)")
        ax2.legend()
        plt.tight_layout(); plt.savefig(os.path.join(save_path, "overall_gap_distribution.png"), dpi=120); plt.close()

    # 绘制明细长图
    plot_symbols = symbols[:n]
    num_rows = (len(plot_symbols) + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(30, 8 * num_rows), constrained_layout=True)
    axes = axes.flatten()
    date_fmt = mdates.DateFormatter('%m-%d')
    cmap_bg, cmap_line = mpl.colormaps.get_cmap('Pastel1'), mpl.colormaps.get_cmap('tab10')

    for i, sym in enumerate(plot_symbols):
        ax, ax2 = axes[i], axes[i].twinx()
        sub_full = full_test_df[full_test_df['symbol_code'] == sym].sort_values('date_dt').copy()
        sub_plot = sub_full[sub_full['date_dt'] >= split_date]
        if sub_plot.empty: continue
        daily_main = sub_full[sub_full['rank'] == 1].drop_duplicates('date_dt').set_index('date_dt')
        for idx, m_code in enumerate(daily_main['delivery_code'].unique()):
            m_dates = daily_main[daily_main['delivery_code'] == m_code].index
            if len(m_dates) > 0: ax.axvspan(m_dates.min(), m_dates.max(), color=cmap_bg(idx % 9), alpha=0.1)
        
        y_max, y_range = sub_plot['prob'].max(), max(sub_plot['prob'].max() - sub_plot['prob'].min(), 0.1)
        involved = sub_plot[sub_plot['rank'] <= 5]['delivery_code'].unique()
        for idx, code in enumerate(involved):
            c_data = sub_plot[sub_plot['delivery_code'] == code].sort_values('date_dt')
            ax.plot(c_data['date_dt'], c_data['prob'], label=f'{code}', color=cmap_line(idx % 10), lw=2)
            real_sw = daily_main[(daily_main['delivery_code'] == code) & (daily_main['delivery_code'] != daily_main['delivery_code'].shift(1))]
            for d in real_sw[real_sw.index >= split_date].index:
                ax.scatter(d, y_max + 0.1 * y_range, color='red', marker='*', s=400, edgecolors='black', zorder=30)
                label_str = d.strftime('%m-%d(%a)') 
                ax.text(d, y_max + 0.15 * y_range, label_str, color='red', fontsize=11, fontweight='bold', ha='center', rotation=0)
                try: ax.axvline(trade_dates_list[trade_dates_list.index(d) - 5], color='red', alpha=0.3, linestyle=':', lw=1.5)
                except: pass
        sym_matches = matched_dates_dict.get(sym, [])
        for t_raw in sym_matches:
            if t_raw >= split_date: ax.scatter(t_raw, y_max + 0.3 * y_range, color='blue', marker='x', s=120, lw=2, zorder=25)
        res = stats_df[stats_df['symbol'] == sym].iloc[0]
        ax.set_title(f"[{sym}] Recall: {res['Recall']:.0%} | Prec: {res['Precision']:.1%} | Avg_Gap(vs T-5): {res['Avg_Gap']+5:.1f}d", fontsize=18, fontweight='bold')
        ax.xaxis.set_major_formatter(date_fmt); ax2.set_yticks([])
    for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
    if save_path: plt.savefig(os.path.join(save_path, "full_symbols_analysis.png"), dpi=100, bbox_inches='tight')
    plt.close()