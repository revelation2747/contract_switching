import json
import math
import os
import subprocess
import warnings

import chinese_calendar
import lightgbm as lgb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
import streamlit.components.v1 as components
from pyarrow import fs
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==========================================
# 全部可调参数
# ==========================================
OLD_FOLDER = "/user/zli/comm/chinese_future/depth_v0.4"
NEW_FOLDER = "/user/zli/comm/data/depth_data/hlsh01"
CUTOFF_DATE = pd.Timestamp("2025-09-30")
START_DATE_DEFAULT = pd.Timestamp("2022-01-01")
PARQUET_FILE = "daily_volume_oi_close.parquet"

RANDOM_SEED = 42
LOOKBACK_TRAIN_DAYS = 800
VALIDATION_DAYS = 90
VALIDATION_MONTH_SHIFT = 3
PLOT_ROLLING_DAYS = 90
SCHEDULE_HOUR = 18
SCHEDULE_MINUTE = 0
AUTO_REFRESH_SECONDS = 60
AUTO_REFRESH_ENABLED = False
DEFER_HEAVY_RENDER = False
AUTO_RUN_SCHEDULED_ON_PAGE_LOAD = False
AUTO_BOOTSTRAP_IF_EMPTY = False
BOOTSTRAP_DAYS_AGO = 30
SAVE_SNAPSHOT_TO_HDFS = True
HDFS_SNAPSHOT_DIR = "/ext1/user/zli/hxr/project/contract_switching/snapshots"
HDFS_UPLOAD_USER = "zli"
RUN_LOCK_STALE_MINUTES = 120

OUTPUT_DIR = "output_results"
SNAPSHOT_META_FILE = os.path.join(OUTPUT_DIR, "latest_snapshot_meta.json")
VALID_RESULTS_FILE = os.path.join(OUTPUT_DIR, "validation_results.parquet")
MONTHLY_SIGNAL_DIR = os.path.join(OUTPUT_DIR, "monthly_signals")
RUN_LOCK_FILE = os.path.join(OUTPUT_DIR, "run.lock")

EXCLUDED_SYMBOLS = ["PM", "JR", "LR", "RI", "ZC", "WH", "BB", "RS", "FB", "RR", "PP_F", "WR", "L_F", "V_F"]


CONFIG_T5 = {
    "family": "three_model",
    "s_rank": {"window_before": 8, "window_after": 8, "noise_frac": 0.1},
    "s_vshare": {"window_before": 20, "window_after": 20, "noise_frac": 0.0},
    "s_days": {"window_before": 20, "window_after": 20, "noise_frac": 0.0},
    "rank_cfg": {"learning_rate": 0.02, "num_leaves": 63, "feature_fraction": 0.85, "min_data_in_leaf": 20, "num_boost_round": 900},
    "v_cfg": {"learning_rate": 0.02, "num_leaves": 63, "feature_fraction": 0.85, "min_data_in_leaf": 20, "num_boost_round": 900},
    "d_cfg": {"learning_rate": 0.02, "num_leaves": 63, "feature_fraction": 0.85, "min_data_in_leaf": 20, "num_boost_round": 900},
    "mlp_cfg": {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "learning_rate_init": 1e-3, "max_iter": 1200},
    "ens_cfg": {"method": "exponential", "decay": 0.05, "gamma": 2.0}
}
CONFIG_T3 = {
    "family": "three_model",  
    "s_rank": {"window_before": 5, "window_after": 5, "noise_frac": 0.1},
    "s_vshare": {"window_before": 20, "window_after": 20, "noise_frac": 0.0},
    "s_days": {"window_before": 20, "window_after": 20, "noise_frac": 0.0},
    "rank_cfg": {"learning_rate": 0.02, "num_leaves": 127, "feature_fraction": 0.60, "min_data_in_leaf": 20, "num_boost_round": 900},
    "v_cfg": {"learning_rate": 0.02, "num_leaves": 63, "feature_fraction": 0.85, "min_data_in_leaf": 20, "num_boost_round": 900},
    "d_cfg": {"learning_rate": 0.02, "num_leaves": 63, "feature_fraction": 0.85, "min_data_in_leaf": 20, "num_boost_round": 900},
    "mlp_cfg": {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "learning_rate_init": 1e-3, "max_iter": 1200},
    "ens_cfg": {"method": "exponential", "decay": 0.05, "gamma": 2.0}
}

# Backward-compat aliases (used by backup.py and older code paths).
ENSEMBLE_CFG = CONFIG_T5
HORIZON_CONFIGS = {
    5: CONFIG_T5,
    3: CONFIG_T3,
}
HORIZON_ORDER = [5, 3]


def get_snapshot_paths(horizon):
    hz = int(horizon)
    suffix = f"t{hz}"
    return {
        "meta_file": os.path.join(OUTPUT_DIR, f"latest_snapshot_meta_{suffix}.json"),
        "valid_results_file": os.path.join(OUTPUT_DIR, f"validation_results_{suffix}.parquet"),
        "monthly_signal_dir": os.path.join(OUTPUT_DIR, f"monthly_signals_{suffix}"),
    }


FEATURE_COLS = [
    "near_holiday_3d",
    # "v_log_ratio",
    "oi_v_eff",
    "current_rank2_days",
    "v_share_diff_1",
    "v_share_diff_3",
    "v_share_diff_5",
    "v_share_accel",
    "v_share_ma3_slope_1",
    "v_share_ma3_slope_3",
    "v_share_ma3_slope_5",
    "v_share_ma3_accel",
    "v_share_ma5_slope_1",
    "v_share_ma5_slope_3",
    "v_share_ma5_slope_5",
    "v_share_ma5_accel",
    "o_share_diff_1",
    "o_share_diff_3",
    "o_share_diff_5",
    "o_share_accel",
    "o_share_ma3_slope_1",
    "o_share_ma3_slope_3",
    "o_share_ma3_slope_5",
    "o_share_ma3_accel",
    "o_share_ma5_slope_1",
    "o_share_ma5_slope_3",
    "o_share_ma5_slope_5",
    "o_share_ma5_accel",
    "v_share_hwm_5",
    "v_share_hwm_diff",
    # "v_share_ma_3",
    # "v_share_ma_5",
    "v_bias_3",
    "v_bias_5",
    "price_chg_1",
    "price_chg_3",
    "price_chg_5",
    "price_vs_v1",
    "vol_vs_v1",
    "basis_slope_3",
    "basis_accel",
    "basis_diff_1",
    "basis_convergence",
    "amihud_vs_v1",
    "vol_adj_flow",
    "vol_adj_flow_sum_3",
    "vol_adj_flow_sum_5",
    "rel_oi_inflow",
    "oi_handover_ratio",
    "v_skew_5",
    "div_oi_v_3",
    "div_oi_v_5",
    "o_accel_vs_v1",
    "days_since_alert",
    "day_of_week",
    "month",
    "day_of_month",
    "days_to_month_end",
    "is_month_end_surge",
    "months_to_maturity",
    "v_co_movement",
    "in_chaos_zone",
    "symbol_cat",
]

@st.cache_resource(show_spinner=False)
def get_hdfs_fs():
    return fs.HadoopFileSystem("hdfs://ftxz-hadoop", user=HDFS_UPLOAD_USER)

st.set_page_config(page_title="Contract Switching", layout="wide")
st.title("主力合约切换信号")


# ==========================================
# 增量数据提取
# ==========================================
def get_hdfs_files(folder, start_date, end_date=None):
    hdfs_fs = get_hdfs_fs()
    selector = fs.FileSelector(folder, recursive=True)
    infos = hdfs_fs.get_file_info(selector)
    files = []
    for info in infos:
        if info.type != fs.FileType.File:
            continue
        fname = os.path.basename(info.path)
        try:
            date_str = fname.split(".")[0]
            file_date = pd.to_datetime(date_str, format="%Y%m%d")
            if file_date < start_date:
                continue
            if end_date is not None and file_date > end_date:
                continue
            files.append((info.path, file_date))
        except Exception:
            continue
    files.sort(key=lambda x: x[1])
    return files


def get_delivery_code_vectorized(series):
    has_F = series.str.endswith(("F", "f"))
    clean_series = series.str.replace(r"[Ff]$", "", regex=True)
    s = clean_series.str.replace(r"\D+$", "", regex=True)
    suffix = s.str[-4:]
    is_digit = suffix.str[0].str.isdigit().fillna(False)
    delivery_code = np.where((s.str.len() >= 4) & is_digit, suffix, "2" + s.str[-3:])
    symbol_code = clean_series.str.replace(r"\d+", "", regex=True).str.upper()
    return symbol_code, delivery_code, has_F


def process_file_fast(file_path, file_date):
    try:
        hdfs_fs = get_hdfs_fs()
        cols = ["server_ts", "symbol_str", "price", "volume", "open_interest"]
        table = pq.read_table(file_path, filesystem=hdfs_fs, columns=cols)
        df = table.to_pandas()
        if df.empty:
            return None

        for col in ["price", "volume", "open_interest"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] > 1e8, col] = np.nan
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        df = df.sort_values("server_ts")
        res = df.groupby("symbol_str", observed=True).tail(1).copy()
        res.rename(columns={"price": "close", "open_interest": "oi"}, inplace=True)
        res["date"] = file_date
        return res[["symbol_str", "close", "volume", "oi", "date"]]
    except Exception:
        return None


def update_volume_oi():
    if os.path.exists(PARQUET_FILE):
        existing_df = pd.read_parquet(PARQUET_FILE)
        existing_df["date"] = pd.to_datetime(existing_df["date"])
        start_search = existing_df["date"].max() + pd.Timedelta(days=1)
    else:
        existing_df = pd.DataFrame()
        start_search = START_DATE_DEFAULT

    files_to_process = []
    if start_search <= CUTOFF_DATE:
        files_to_process += get_hdfs_files(OLD_FOLDER, start_search, CUTOFF_DATE)

    start_new = max(start_search, CUTOFF_DATE + pd.Timedelta(days=1))
    files_to_process += get_hdfs_files(NEW_FOLDER, start_new, None)

    if not files_to_process:
        return "[INFO] No new files to process."

    batch_results = []
    for file_path, file_date in tqdm(files_to_process, desc="Syncing Data"):
        df_res = process_file_fast(file_path, file_date)
        if df_res is not None:
            batch_results.append(df_res)

    if not batch_results:
        return "[WARN] No valid rows fetched."

    final_df = pd.concat(batch_results, ignore_index=True)
    s_code, d_code, has_F = get_delivery_code_vectorized(final_df["symbol_str"].astype(str))
    final_df["symbol_code"] = s_code
    final_df.loc[has_F, "symbol_code"] += "_F"
    final_df["delivery_code"] = d_code
    final_df.drop(columns=["symbol_str"], inplace=True)

    if not existing_df.empty:
        combined = pd.concat([existing_df, final_df], ignore_index=True)
    else:
        combined = final_df

    combined.drop_duplicates(subset=["symbol_code", "delivery_code", "date"], keep="last", inplace=True)
    combined.sort_values(["date", "symbol_code", "delivery_code"], inplace=True)
    combined.to_parquet(PARQUET_FILE, index=False)

    return f"[SUCCESS] Processed {len(files_to_process)} files. Latest date: {combined['date'].max().date()}"


# ==========================================
# 特征与模型逻辑
# ==========================================
def define_main_adaptive_robust(df):
    df = df.sort_values(["symbol_code", "date_dt", "delivery_code"])
    df["volume"] = df["volume"].replace(0, np.nan)
    df["oi"] = df["oi"].replace(0, np.nan)

    v_ranks = df.groupby(["symbol_code", "date_dt"])["volume"].rank(ascending=False, method="first", na_option="bottom")
    o_ranks = df.groupby(["symbol_code", "date_dt"])["oi"].rank(ascending=False, method="first", na_option="bottom")

    c1 = df["volume"].notna() & (v_ranks == 1) & df["oi"].notna() & (o_ranks <= 2)
    c2 = df["oi"].notna() & (o_ranks == 1)

    df["tmp_score"] = np.where(c1, 2, np.where(c2, 1, 0))
    max_score = df.groupby(["symbol_code", "date_dt"])["tmp_score"].transform("max")

    df["is_candidate"] = (df["tmp_score"] == max_score) & (df["tmp_score"] > 0)
    candidates = df[df["is_candidate"]][["symbol_code", "date_dt", "delivery_code"]].drop_duplicates(subset=["symbol_code", "date_dt"])
    candidates = candidates.rename(columns={"delivery_code": "main_code_cand"})

    sym_dates = df[["symbol_code", "date_dt"]].drop_duplicates().sort_values(["symbol_code", "date_dt"])
    sym_dates = sym_dates.merge(candidates, on=["symbol_code", "date_dt"], how="left")
    sym_dates["main_code_final"] = sym_dates.groupby("symbol_code")["main_code_cand"].ffill()

    df = df.merge(sym_dates[["symbol_code", "date_dt", "main_code_final"]], on=["symbol_code", "date_dt"], how="left")
    
    # Fallback to v_rank == 1 if main_code_final is still NaN
    fallback = df[v_ranks == 1][["symbol_code", "date_dt", "delivery_code"]].drop_duplicates(subset=["symbol_code", "date_dt"])
    fallback = fallback.rename(columns={"delivery_code": "fallback_main"})
    df = df.merge(fallback, on=["symbol_code", "date_dt"], how="left")
    df["main_code_final"] = df["main_code_final"].fillna(df["fallback_main"])

    df["daily_rank"] = np.where(df["delivery_code"] == df["main_code_final"], 1, 99)
    df["v_rank"] = v_ranks

    df.rename(columns={"main_code_final": "actual_main_code"}, inplace=True)
    df.drop(columns=["tmp_score", "is_candidate", "fallback_main"], inplace=True)
    return df


def make_refined_features(df):
    df = df.sort_values(["symbol_code", "delivery_code", "date_dt"])
    df["volume_clean"] = df["volume"].replace(0, np.nan)
    df["oi_clean"] = df["oi"].replace(0, np.nan)

    group_con = df.groupby(["symbol_code", "delivery_code"])
    group_daily = df.groupby(["symbol_code", "date_dt"])

    daily_v_sum = group_daily["volume"].transform("sum").replace(0, np.nan)
    daily_o_sum = group_daily["oi"].transform("sum").replace(0, np.nan)

    df["v_share"] = df["volume"] / daily_v_sum
    df["o_share"] = df["oi"] / daily_o_sum
    df["v_log_ratio"] = np.log(df["v_share"] / (1 - df["v_share"] + 1e-9) + 1e-9)
    df["oi_v_eff"] = df["v_share"] / df["o_share"].replace(0, np.nan)

    df["is_rank2"] = (df["v_rank"] == 2).astype(np.int8)
    state_block = (df["is_rank2"] != group_con["is_rank2"].shift(1)).cumsum()
    df["current_rank2_days"] = df.groupby(["symbol_code", "delivery_code", state_block]).cumcount() * df["is_rank2"]

    unique_dates = df["date_dt"].dropna().unique()
    holiday_flags = {}
    for d in unique_dates:
        dt = pd.to_datetime(d)
        is_near = False
        for k in range(-3, 4):
            check_date = dt + pd.Timedelta(days=k)
            try:
                if chinese_calendar.is_holiday(check_date.date()) and not chinese_calendar.is_workday(check_date.date()):
                    is_near = True
                    break
            except NotImplementedError:
                pass
        holiday_flags[d] = 1 if is_near else 0
    df["near_holiday_3d"] = df["date_dt"].map(holiday_flags)

    for col in ["v_share", "o_share"]:
        df[f"{col}_ma_3"] = group_con[col].transform(lambda x: x.rolling(3).mean())
        df[f"{col}_ma_5"] = group_con[col].transform(lambda x: x.rolling(5).mean())

        for k in [1, 3, 5]:
            df[f"{col}_diff_{k}"] = group_con[col].diff(k)
            df[f"{col}_ma3_slope_{k}"] = group_con[f"{col}_ma_3"].diff(k)
            df[f"{col}_ma5_slope_{k}"] = group_con[f"{col}_ma_5"].diff(k)

            if k == 1:
                df[f"{col}_accel"] = group_con[f"{col}_diff_{k}"].diff(1)
                df[f"{col}_ma3_accel"] = group_con[f"{col}_ma3_slope_1"].diff(1)
                df[f"{col}_ma5_accel"] = group_con[f"{col}_ma5_slope_1"].diff(1)

    df["v_share_hwm_5"] = group_con["v_share"].transform(lambda x: x.rolling(5).max())
    df["v_share_hwm_diff"] = df["v_share_hwm_5"] - df["v_share"]
    for k in [3, 5]:
        df[f"v_bias_{k}"] = df["v_share"] / df[f"v_share_ma_{k}"].replace(0, np.nan)

    for k in [1, 3, 5]:
        df[f"price_chg_{k}"] = group_con["close"].pct_change(k)

    v1_price = df[df["v_rank"] == 1].groupby(["symbol_code", "date_dt"])["close"].first().replace(0, np.nan)
    df["price_vs_v1"] = df["close"] / df.set_index(["symbol_code", "date_dt"]).index.map(v1_price)

    df["vol_5d"] = group_con["close"].transform(lambda x: x.rolling(5).std())
    v1_vol = df[df["v_rank"] == 1].groupby(["symbol_code", "date_dt"])["vol_5d"].first().replace(0, np.nan)
    df["vol_vs_v1"] = df["vol_5d"] / df.set_index(["symbol_code", "date_dt"]).index.map(v1_vol)

    df["basis_v1"] = df["price_vs_v1"] - 1
    df["basis_slope_3"] = group_con["basis_v1"].diff(3)
    df["basis_accel"] = group_con["basis_slope_3"].diff(1)

    df["basis_diff_1"] = group_con["basis_v1"].diff(1)
    df["basis_convergence"] = group_con["basis_diff_1"].diff(1)

    df["returns_abs"] = group_con["close"].pct_change().abs()
    amount = (df["volume"] * df["close"]).replace(0, np.nan)
    df["amihud"] = df["returns_abs"] / amount
    v1_amihud = df[df["v_rank"] == 1].groupby(["symbol_code", "date_dt"])["amihud"].first().replace(0, np.nan)
    df["amihud_vs_v1"] = df["amihud"] / df.set_index(["symbol_code", "date_dt"]).index.map(v1_amihud)

    money_flow = (df["volume"] * df["close"]).diff(1)
    df["vol_adj_flow"] = money_flow / df["vol_5d"].replace(0, np.nan)
    df["vol_adj_flow_sum_3"] = group_con["vol_adj_flow"].transform(lambda x: x.rolling(3).sum())
    df["vol_adj_flow_sum_5"] = group_con["vol_adj_flow"].transform(lambda x: x.rolling(5).sum())

    df["oi_diff"] = group_con["oi"].diff(1)
    neg_oi_diff = df["oi_diff"].clip(upper=0)
    total_oi_drop = neg_oi_diff.groupby([df["date_dt"], df["symbol_code"]]).transform("sum").abs().replace(0, np.nan)
    df["rel_oi_inflow"] = df["oi_diff"] / total_oi_drop

    v1_oi_diff = df[df["v_rank"] == 1].groupby(["symbol_code", "date_dt"])["oi_diff"].first()
    df["v1_oi_diff_ref"] = df.set_index(["symbol_code", "date_dt"]).index.map(v1_oi_diff)
    df["oi_handover_ratio"] = df["oi_diff"] / (df["v1_oi_diff_ref"].abs() + 1e-9)

    df["v_skew_5"] = group_con["volume"].transform(lambda x: x.rolling(5).skew())

    df["div_oi_v_3"] = df["o_share_diff_3"] - df["v_share_diff_3"]
    df["div_oi_v_5"] = df["o_share_diff_5"] - df["v_share_diff_5"]
    v1_o_accel = df[df["v_rank"] == 1].groupby(["symbol_code", "date_dt"])["o_share_accel"].first()
    df["o_accel_vs_v1"] = df["o_share_accel"] - df.set_index(["symbol_code", "date_dt"]).index.map(v1_o_accel)

    df["day_of_week"] = df["date_dt"].dt.dayofweek
    df["month"] = df["date_dt"].dt.month
    df["day_of_month"] = df["date_dt"].dt.day
    df["days_to_month_end"] = df["date_dt"].dt.days_in_month - df["day_of_month"]
    df["is_month_end_surge"] = (df["days_to_month_end"] <= 3).astype(int)
    yr = df["delivery_code"].astype(int) // 100
    yr_real = np.where(yr < 100, yr + 2000, yr)
    month_real = df["delivery_code"].astype(int) % 100
    df["months_to_maturity"] = (yr_real * 12 + month_real) - (df["date_dt"].dt.year * 12 + df["date_dt"].dt.month)

    df["is_alert"] = (df["v_share"] > 0.10).astype(int)
    df["days_since_alert"] = group_con["is_alert"].cumsum() * df["is_alert"]

    r1_diff = df[df["v_rank"] == 1].groupby(["symbol_code", "date_dt"])["v_share_diff_1"].first()
    df["r1_diff_ref"] = df.set_index(["symbol_code", "date_dt"]).index.map(r1_diff)
    df["v_co_movement"] = df["v_share_diff_1"] * df["r1_diff_ref"]
    df["in_chaos_zone"] = ((df["v_share"] > 0.3) & (df["v_share"] < 0.7)).astype(np.int8)

    df["symbol_cat"] = df["symbol_code"].astype("category")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def prepare_data_and_targets(path, horizon):
    df = pd.read_parquet(path)
    df["date_dt"] = pd.to_datetime(df["date"])
    df["is_excluded"] = df["symbol_code"].isin(EXCLUDED_SYMBOLS)

    df = define_main_adaptive_robust(df)
    mains = df[df["daily_rank"] == 1][["symbol_code", "date_dt", "delivery_code"]].sort_values(["symbol_code", "date_dt"])

    future_col = f"future_main_t{horizon}"
    is_col = f"is_t{horizon}_main"
    not_main_col = f"not_main_t0_t{horizon-1}"
    vshare_col = f"v_share_t{horizon}"

    mains[future_col] = mains.groupby("symbol_code")["delivery_code"].shift(-horizon)
    df = pd.merge(df, mains[["symbol_code", "date_dt", future_col]], on=["symbol_code", "date_dt"], how="left")

    main_seq = mains[["symbol_code", "date_dt", "delivery_code"]].rename(columns={"delivery_code": "main_t0"})
    for k in range(1, horizon):
        main_seq[f"main_t{k}"] = main_seq.groupby("symbol_code")["main_t0"].shift(-k)
    df = df.merge(main_seq, on=["symbol_code", "date_dt"], how="left")

    is_tn_main = df["delivery_code"] == df[future_col]
    cols_window = ["main_t0"] + [f"main_t{k}" for k in range(1, horizon)]
    has_full = df[cols_window].notna().all(axis=1)
    neq_all = np.ones(len(df), dtype=bool)
    for c in cols_window:
        neq_all &= (df["delivery_code"] != df[c])
    df[is_col] = is_tn_main
    df[not_main_col] = has_full & neq_all
    df = df.drop(columns=cols_window)

    df = make_refined_features(df)

    LABEL_SWITCH = 2
    LABEL_MAIN = 1

    df_sorted = df.sort_values(["symbol_code", "delivery_code", "date_dt"])
    df[vshare_col] = df_sorted.groupby(["symbol_code", "delivery_code"])["v_share"].shift(-horizon)

    unique_dates = np.sort(df["date_dt"].unique())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    df["date_idx"] = df["date_dt"].map(date_to_idx)

    mains_first = df[df["daily_rank"] == 1].groupby(["symbol_code", "delivery_code"])["date_idx"].min().reset_index()
    mains_first.rename(columns={"date_idx": "first_main_idx"}, inplace=True)
    df = df.merge(mains_first, on=["symbol_code", "delivery_code"], how="left")

    df["remaining_days"] = df["first_main_idx"] - df["date_idx"]
    df.loc[df["remaining_days"] < 0, "remaining_days"] = 0
    df["remaining_days"] = df["remaining_days"].fillna(30)
    df.loc[df["remaining_days"] > 30, "remaining_days"] = 30

    df["target_rank"] = np.where(df[is_col] & df[not_main_col], LABEL_SWITCH, np.where(df[is_col], LABEL_MAIN, 0)).astype(np.int32)
    df["target_vshare"] = df[vshare_col].fillna(0).astype(np.float32)
    df["target_days"] = df["remaining_days"].astype(np.float32)
    return df.sort_index()



# ==========================================
# 建模与评估
# ==========================================
def get_training_samples_aggressive(df_train, window_before, window_after, noise_frac, random_state=42):
    mains = df_train[df_train["daily_rank"] == 1].sort_values(["symbol_code", "date_dt"])
    mains["is_switch"] = (mains["delivery_code"] != mains.groupby("symbol_code")["delivery_code"].shift(1)).astype(int)

    switch_points = mains[mains["is_switch"] == 1][["symbol_code", "date_dt"]]
    indices = []
    for _, row in switch_points.iterrows():
        mask = (
            (df_train["symbol_code"] == row["symbol_code"])
            & (df_train["date_dt"] >= row["date_dt"] - pd.Timedelta(days=window_before))
            & (df_train["date_dt"] <= row["date_dt"] + pd.Timedelta(days=window_after))
        )
        indices.extend(df_train[mask].index.tolist())

    focus_idx = sorted(set(indices))
    focus_df = df_train.loc[focus_idx] if len(focus_idx) > 0 else df_train.iloc[0:0].copy()

    remaining = df_train.drop(index=focus_idx)
    if len(remaining) > 0 and noise_frac > 0:
        noise_df = remaining.sample(frac=min(noise_frac, 1.0), random_state=random_state)
    else:
        noise_df = remaining.iloc[0:0].copy()

    sampled = pd.concat([focus_df, noise_df], axis=0)
    return sampled.sort_values(["date_dt", "symbol_code", "delivery_code"])


def min_max_scale(x):
    margin = x.max() - x.min()
    return (x - x.min()) / margin if margin > 1e-9 else x * 0.0


def monthly_validation_start(latest_date, floor_date, month_shift=VALIDATION_MONTH_SHIFT):
    latest_month_start = pd.Timestamp(latest_date).to_period("M").to_timestamp()
    val_start = latest_month_start - pd.DateOffset(months=int(month_shift))
    floor_ts = pd.Timestamp(floor_date)

    # Keep validation window monthly while ensuring train/val are both non-empty.
    if val_start <= floor_ts:
        val_start = floor_ts + pd.Timedelta(days=1)
    if val_start >= pd.Timestamp(latest_date):
        val_start = pd.Timestamp(latest_date) - pd.Timedelta(days=1)
    return val_start


def build_final_eval(pred_df, ens_cfg, horizon, apply_window_filter=False):
    target_col = f"is_t{horizon}_main"
    tmp = pred_df.copy()
    tmp = tmp.sort_values(["date_dt", "symbol_code", "delivery_code"]).drop_duplicates(["date_dt", "symbol_code", "delivery_code"], keep="last")
    tmp["norm_rank"] = tmp.groupby(["date_dt", "symbol_code"])["pred_rank_raw"].transform(min_max_scale)

    method = ens_cfg.get("method", "multiplicative")
    if method == "multiplicative":
        base_pow = ens_cfg.get("base_pow", 1.0)
        alpha = ens_cfg.get("alpha", 5.0)
        beta = ens_cfg.get("beta", 1.0)
        rank_score = tmp["norm_rank"] ** base_pow
        inf_days = alpha / (np.clip(tmp["pred_days"], 0, None) + alpha)
        inf_vshare = np.clip(tmp["pred_vshare"], 0, None) * beta
        tmp["ensemble_score"] = rank_score * (1.0 + inf_days + inf_vshare)
    elif method == "additive":
        w_rank = ens_cfg.get("w_rank", 1.0)
        w_days = ens_cfg.get("w_days", 1.0)
        w_vshare = ens_cfg.get("w_vshare", 1.0)
        days_score = 1.0 / (np.clip(tmp["pred_days"], 0, None) + 1.0)
        tmp["ensemble_score"] = w_rank * tmp["norm_rank"] + w_days * days_score + w_vshare * np.clip(tmp["pred_vshare"], 0, None)
    else:
        decay = ens_cfg.get("decay", 0.1)
        gamma = ens_cfg.get("gamma", 1.0)
        day_penalty = np.exp(-decay * np.clip(tmp["pred_days"], 0, None))
        vshare_boost = np.clip(tmp["pred_vshare"], 0.001, None) ** gamma
        tmp["ensemble_score"] = tmp["norm_rank"] * day_penalty * vshare_boost

    tmp["ensemble_pred_rank"] = tmp.groupby(["date_dt", "symbol_code"])["ensemble_score"].rank(ascending=False, method="first")
    
    if apply_window_filter:
        min_date = tmp["date_dt"].min() + pd.Timedelta(days=5)
        max_date = tmp["date_dt"].max() - pd.Timedelta(days=5)
        final_eval = tmp[(tmp["date_dt"] >= min_date) & (tmp["date_dt"] <= max_date)].copy()
    else:
        final_eval = tmp

    main_map = (
        tmp[tmp["daily_rank"] == 1]
        .sort_values(["date_dt", "symbol_code", "delivery_code"])
        .drop_duplicates(["date_dt", "symbol_code"], keep="first")
        .set_index(["date_dt", "symbol_code"])["delivery_code"]
    )
    final_eval["actual_main_code"] = final_eval.set_index(["date_dt", "symbol_code"]).index.map(main_map)

    liq_mask = final_eval["volume"].fillna(0).gt(0) | final_eval["oi"].fillna(0).gt(0)
    has_main_ref = final_eval["actual_main_code"].notna()
    final_eval["is_action"] = (final_eval["ensemble_pred_rank"] == 1) & has_main_ref & (final_eval["delivery_code"] != final_eval["actual_main_code"]) & liq_mask
    final_eval["is_real_switch_target"] = has_main_ref & final_eval[target_col] & (final_eval["delivery_code"] != final_eval["actual_main_code"])
    return final_eval


def evaluate_trial(final_eval):
    y_true = final_eval["is_real_switch_target"].astype(int)
    y_pred = final_eval["is_action"].astype(int)
    metrics = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    switch_groups = final_eval[final_eval["is_real_switch_target"] == 1].copy()
    switch_groups = switch_groups.sort_values(["symbol_code", "delivery_code", "date_dt"])
    switch_groups["signal_day"] = switch_groups.groupby(["symbol_code", "delivery_code"]).cumcount() + 1
    for d in range(1, 6):
        day_df = switch_groups[switch_groups["signal_day"] == d]
        hit_rate = float(day_df["is_action"].mean()) if len(day_df) > 0 else 0.0
        metrics[f"day{d}_hit_rate"] = hit_rate
        metrics[f"day{d}_count"] = int(len(day_df))

    return metrics






def train_and_infer_models(train_df, eval_df, feat_cols, cfg, is_today=False):
    family = cfg.get("family", "three_model")
    pred_df = eval_df.copy()
    
    feat_cols_rank = [c for c in feat_cols if c != "symbol_cat"] + ["symbol_cat"]
    feat_cols_mlp = [c for c in feat_cols if c != "symbol_cat"]
    feat_cols_days = [c for c in feat_cols if c != "symbol_cat"]
    feat_cols_vshare = [c for c in feat_cols if c not in {"symbol_cat", "v_share"}]
    
    m_dict = {"family": family}

    if family == "lgbm_rank":
        tr_rank = get_training_samples_aggressive(train_df, **cfg["s_rank"], random_state=RANDOM_SEED)
        q_train = tr_rank.groupby(["date_dt", "symbol_code"]).size().values
        ds_train = lgb.Dataset(tr_rank[feat_cols_rank], label=tr_rank["target_rank"], group=q_train, categorical_feature=["symbol_cat"])
        m_rank = lgb.train({"objective": "lambdarank", "metric": "ndcg", "ndcg_at": [1, 2], "seed": RANDOM_SEED, "verbose": -1, "n_jobs": 1, **cfg["rank_cfg"]}, ds_train, num_boost_round=cfg["rank_cfg"]["num_boost_round"])
        pred_df["pred_rank_raw"] = m_rank.predict(pred_df[feat_cols_rank])
        pred_df["pred_vshare"] = 0.0
        pred_df["pred_days"] = 0.0
        m_dict["m_rank"] = m_rank
        
    elif family == "mlp_rank":
        tr_rank = get_training_samples_aggressive(train_df, **cfg["s_rank"], random_state=RANDOM_SEED)
        x = tr_rank[feat_cols_mlp].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        y = tr_rank["target_rank"].astype(float).values
        m_mlp = MLPRegressor(random_state=RANDOM_SEED, **cfg["mlp_cfg"])
        m_mlp.fit(x_scaled, y)
        x_eval = pred_df[feat_cols_mlp].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pred_df["pred_rank_raw"] = m_mlp.predict(scaler.transform(x_eval))
        pred_df["pred_vshare"] = 0.0
        pred_df["pred_days"] = 0.0
        m_dict["m_mlp"] = m_mlp
        m_dict["scaler"] = scaler

    elif family == "days_reg":
        tr_d = get_training_samples_aggressive(train_df, **cfg["s_days"], random_state=RANDOM_SEED)
        ds_train = lgb.Dataset(tr_d[feat_cols_days], label=tr_d["target_days"])
        m_days = lgb.train({"objective": "regression", "metric": "rmse", "seed": RANDOM_SEED, "verbose": -1, "n_jobs": 1, **cfg["d_cfg"]}, ds_train, num_boost_round=cfg["d_cfg"]["num_boost_round"])
        pred_df["pred_days"] = m_days.predict(pred_df[feat_cols_days])
        pred_df["pred_vshare"] = 0.0
        pred_df["pred_rank_raw"] = -pred_df["pred_days"]
        m_dict["m_days"] = m_days

    elif family == "vshare_reg":
        tr_v = get_training_samples_aggressive(train_df, **cfg["s_vshare"], random_state=RANDOM_SEED)
        ds_train = lgb.Dataset(tr_v[feat_cols_vshare], label=tr_v["target_vshare"])
        m_vshare = lgb.train({"objective": "regression", "metric": "rmse", "seed": RANDOM_SEED, "verbose": -1, "n_jobs": 1, **cfg["v_cfg"]}, ds_train, num_boost_round=cfg["v_cfg"]["num_boost_round"])
        pred_df["pred_vshare"] = m_vshare.predict(pred_df[feat_cols_vshare])
        pred_df["pred_days"] = 0.0
        pred_df["pred_rank_raw"] = pred_df["pred_vshare"]
        m_dict["m_vshare"] = m_vshare

    else:
        # three_model
        tr_rank = get_training_samples_aggressive(train_df, **cfg["s_rank"], random_state=RANDOM_SEED)
        tr_v = get_training_samples_aggressive(train_df, **cfg["s_vshare"], random_state=RANDOM_SEED)
        tr_d = get_training_samples_aggressive(train_df, **cfg["s_days"], random_state=RANDOM_SEED)
        
        q_train = tr_rank.groupby(["date_dt", "symbol_code"]).size().values
        ds_rank = lgb.Dataset(tr_rank[feat_cols_rank], label=tr_rank["target_rank"], group=q_train, categorical_feature=["symbol_cat"])
        ds_vshare = lgb.Dataset(tr_v[feat_cols_vshare], label=tr_v["target_vshare"])
        ds_days = lgb.Dataset(tr_d[feat_cols_days], label=tr_d["target_days"])
        
        m_rank = lgb.train({"objective": "lambdarank", "metric": "ndcg", "ndcg_at": [1, 2], "seed": RANDOM_SEED, "verbose": -1, "n_jobs": 1, **cfg["rank_cfg"]}, ds_rank, num_boost_round=cfg["rank_cfg"]["num_boost_round"])
        m_vshare = lgb.train({"objective": "regression", "metric": "rmse", "seed": RANDOM_SEED, "verbose": -1, "n_jobs": 1, **cfg["v_cfg"]}, ds_vshare, num_boost_round=cfg["v_cfg"]["num_boost_round"])
        m_days = lgb.train({"objective": "regression", "metric": "rmse", "seed": RANDOM_SEED, "verbose": -1, "n_jobs": 1, **cfg["d_cfg"]}, ds_days, num_boost_round=cfg["d_cfg"]["num_boost_round"])
        
        pred_df["pred_rank_raw"] = m_rank.predict(pred_df[feat_cols_rank])
        pred_df["pred_vshare"] = m_vshare.predict(pred_df[feat_cols_vshare])
        pred_df["pred_days"] = m_days.predict(pred_df[feat_cols_days])
        m_dict["m_rank"] = m_rank
        m_dict["m_vshare"] = m_vshare
        m_dict["m_days"] = m_days

    return pred_df, m_dict

def infer_with_models_using_dict(df_slice, feat_cols, m_dict, cfg, horizon, apply_window_filter=False):
    family = m_dict["family"]
    pred_df = df_slice.copy()
    
    feat_cols_rank = [c for c in feat_cols if c != "symbol_cat"] + ["symbol_cat"]
    feat_cols_mlp = [c for c in feat_cols if c != "symbol_cat"]
    feat_cols_days = [c for c in feat_cols if c != "symbol_cat"]
    feat_cols_vshare = [c for c in feat_cols if c not in {"symbol_cat", "v_share"}]
    
    if family == "lgbm_rank":
        pred_df["pred_rank_raw"] = m_dict["m_rank"].predict(pred_df[feat_cols_rank])
        pred_df["pred_vshare"] = 0.0
        pred_df["pred_days"] = 0.0
    elif family == "mlp_rank":
        x_eval = pred_df[feat_cols_mlp].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pred_df["pred_rank_raw"] = m_dict["m_mlp"].predict(m_dict["scaler"].transform(x_eval))
        pred_df["pred_vshare"] = 0.0
        pred_df["pred_days"] = 0.0
    elif family == "days_reg":
        pred_df["pred_days"] = m_dict["m_days"].predict(pred_df[feat_cols_days])
        pred_df["pred_vshare"] = 0.0
        pred_df["pred_rank_raw"] = -pred_df["pred_days"]
    elif family == "vshare_reg":
        pred_df["pred_vshare"] = m_dict["m_vshare"].predict(pred_df[feat_cols_vshare])
        pred_df["pred_days"] = 0.0
        pred_df["pred_rank_raw"] = pred_df["pred_vshare"]
    else:
        pred_df["pred_rank_raw"] = m_dict["m_rank"].predict(pred_df[feat_cols_rank])
        pred_df["pred_vshare"] = m_dict["m_vshare"].predict(pred_df[feat_cols_vshare])
        pred_df["pred_days"] = m_dict["m_days"].predict(pred_df[feat_cols_days])

    return build_final_eval(pred_df, cfg["ens_cfg"], horizon=horizon, apply_window_filter=apply_window_filter)

def filter_latest_signals_by_second_main_liquidity(signals_df, second_main_df):
    if signals_df.empty:
        return signals_df
    second_cols = [c for c in ["symbol_code", "second_main_vol", "second_main_oi"] if c in second_main_df.columns]
    if "symbol_code" not in second_cols:
        return signals_df

    out = signals_df.merge(second_main_df[second_cols], on="symbol_code", how="left")
    pred_has_liq = out["volume"].fillna(0).gt(0) | out["oi"].fillna(0).gt(0)
    second_has_liq = out["second_main_vol"].fillna(0).gt(0) | out["second_main_oi"].fillna(0).gt(0)
    out = out[pred_has_liq | second_has_liq].copy()
    drop_cols = [c for c in ["second_main_vol", "second_main_oi"] if c in out.columns and c not in signals_df.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def symbol_confidence_from_eval(eval_df):
    rows = []
    for sym, sdf in eval_df.groupby("symbol_code"):
        y_true = sdf["is_real_switch_target"].astype(int)
        y_pred = sdf["is_action"].astype(int)
        rows.append(
            {
                "symbol_code": sym,
                "confidence_f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "confidence_precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "confidence_recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "n_samples": int(len(sdf)),
            }
        )
    return pd.DataFrame(rows)


def get_importance_df(model, features, model_name):
    if model is None:
        return pd.DataFrame(
            {
                "feature": [],
                "gain": [],
                "split": [],
                "model": [],
            }
        )
    return pd.DataFrame(
        {
            "feature": features,
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
            "model": model_name,
        }
    ).sort_values("gain", ascending=False)


def run_pipeline(horizon, cfg, as_of_date=None):
    df = prepare_data_and_targets(PARQUET_FILE, horizon)
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    max_available_date = df["date_dt"].max()
    if as_of_date is None:
        latest_date = max_available_date
    else:
        as_of_ts = pd.to_datetime(as_of_date)
        candidate = df.loc[df["date_dt"] <= as_of_ts, "date_dt"]
        if candidate.empty:
            raise RuntimeError("as_of_date 早于数据起始日期，无法回放。")
        latest_date = candidate.max()

    train_start = latest_date - pd.Timedelta(days=LOOKBACK_TRAIN_DAYS)
    target_future_col = f"future_main_t{horizon}"

    train_pool_all = df[(~df["is_excluded"]) & (df[target_future_col].notna())].copy()
    train_pool = train_pool_all[(train_pool_all["date_dt"] >= train_start) & (train_pool_all["date_dt"] < latest_date)].copy()
    
    if train_pool.empty:
        raise RuntimeError("lookback 窗口内没有可训练样本。")

    val_start = monthly_validation_start(latest_date, train_pool["date_dt"].min(), VALIDATION_MONTH_SHIFT)
    
    val_train = train_pool[train_pool["date_dt"] < val_start].copy()
    val_eval = train_pool[train_pool["date_dt"] >= val_start].copy()

    if val_train.empty or val_eval.empty:
        raise RuntimeError("验证窗口为空，请调大 LOOKBACK_TRAIN_DAYS 或调小 VALIDATION_DAYS。")

    pred_val, m_dict = train_and_infer_models(val_train, val_eval, feat_cols, cfg)
    val_final_eval = build_final_eval(pred_val, cfg["ens_cfg"], horizon=horizon, apply_window_filter=False)
    val_metrics = evaluate_trial(val_final_eval)
    conf_by_symbol = symbol_confidence_from_eval(val_final_eval)

    infer_today = df[df["date_dt"] == latest_date].copy()
    today_final = infer_with_models_using_dict(infer_today, feat_cols, m_dict, cfg, horizon=horizon, apply_window_filter=False)

    current_main = (
        infer_today[infer_today["daily_rank"] == 1][["symbol_code", "delivery_code", "volume", "oi"]]
        .drop_duplicates(["symbol_code"])
        .rename(
            columns={
                "delivery_code": "current_main_code",
                "volume": "current_main_vol",
                "oi": "current_main_oi",
            }
        )
        .copy()
    )
    second_main = (
        infer_today[infer_today["daily_rank"] == 99][["symbol_code", "delivery_code", "volume", "oi"]]
        .sort_values(["symbol_code", "volume"], ascending=[True, False])
        .drop_duplicates(["symbol_code"])
        .rename(
            columns={
                "delivery_code": "second_main_code",
                "volume": "second_main_vol",
                "oi": "second_main_oi",
            }
        )
        .copy()
    )
    main_second = current_main.merge(second_main, on="symbol_code", how="left")
    main_second["date_dt"] = latest_date

    signals = today_final[today_final["is_action"]].copy()
    signals = filter_latest_signals_by_second_main_liquidity(signals, main_second)
    if not signals.empty:
        signals = signals.merge(conf_by_symbol[["symbol_code", "confidence_f1"]], on="symbol_code", how="left")
        signals["global_confidence_f1"] = val_metrics["f1"]

    return {
        "df": df,
        "max_available_date": max_available_date,
        "latest_date": latest_date,
        "train_start": train_start,
        "val_start": val_start,
        "feat_cols": feat_cols,
        "signals": signals,
        "val_metrics": val_metrics,
        "val_final_eval": val_final_eval,
        "conf_by_symbol": conf_by_symbol,
        "main_second": main_second,
        "today_final": today_final,
        "m_dict": m_dict,
    }


def render_history_plot(plot_df, symbols=None):
    if symbols is None:
        symbols = sorted(plot_df["symbol_code"].dropna().unique().tolist())
    else:
        symbols = sorted(pd.Series(symbols).dropna().astype(str).unique().tolist())

    if len(symbols) == 0:
        st.warning("历史窗口内无可展示品种。")
        return

    ncols = 4
    nrows = math.ceil(len(symbols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 3.8 * nrows), sharex=False, sharey=True)
    axes = np.array(axes).reshape(-1)

    for i, sym in enumerate(symbols):
        ax = axes[i]
        sdf = plot_df[plot_df["symbol_code"] == sym].copy()
        if sdf.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=12, color="gray", transform=ax.transAxes)
            ax.set_title(f"{sym} | no data")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.2)
            if i % ncols == 0:
                ax.set_ylabel("volume_share")
            if i >= (nrows - 1) * ncols:
                ax.set_xlabel("date")
            continue

        pivot = sdf.pivot_table(index="date_dt", columns="delivery_code", values="v_share", aggfunc="mean")
        if pivot.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=12, color="gray", transform=ax.transAxes)
            ax.set_title(f"{sym} | no data")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.2)
            if i % ncols == 0:
                ax.set_ylabel("volume_share")
            if i >= (nrows - 1) * ncols:
                ax.set_xlabel("date")
            continue

        for col in pivot.columns:
            ax.plot(pivot.index, pivot[col], linewidth=1.1, alpha=0.45)

        sdf = sdf.sort_values("date_dt")
        real_switch_dates = sdf.loc[sdf["is_real_switch_target"] == True, "date_dt"]
        if not real_switch_dates.empty:
            ax.scatter(real_switch_dates, np.full(len(real_switch_dates), 0.95), marker="*", s=120, c="red", alpha=0.9, label="Real Switch Day")

        pred_switch_dates = sdf.loc[sdf["is_action"], "date_dt"]
        if not pred_switch_dates.empty:
            ax.scatter(pred_switch_dates, np.full(len(pred_switch_dates), 0.90), marker="x", s=80, c="blue", alpha=0.85, label="Pred Alert Day")

        y_true_sym = sdf["is_real_switch_target"].astype(int) if "is_real_switch_target" in sdf.columns else pd.Series(0, index=sdf.index)
        y_pred_sym = sdf["is_action"].astype(int) if "is_action" in sdf.columns else pd.Series(0, index=sdf.index)
        p_sym = precision_score(y_true_sym, y_pred_sym, zero_division=0)
        r_sym = recall_score(y_true_sym, y_pred_sym, zero_division=0)
        ax.set_title(f"{sym} | Prec={p_sym:.3f}, Rec={r_sym:.3f}")

        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", labelrotation=45)

        if i % ncols == 0:
            ax.set_ylabel("volume_share")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("date")

    for j in range(len(symbols), len(axes)):
        axes[j].axis("off")

    handles, labels = [], []
    for ax in axes[: len(symbols)]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    if labels:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2)

    fig.suptitle("Per-Symbol Volume Share with Real/Pred Signals", y=1.03, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)


def trigger_auto_refresh(interval_seconds):
    components.html(
        f"""
        <script>
            setTimeout(function() {{
                window.parent.location.reload();
            }}, {int(interval_seconds) * 1000});
        </script>
        """,
        height=0,
    )


def load_meta(horizon=5):
    paths = get_snapshot_paths(horizon)
    meta_file = paths["meta_file"]
    if not os.path.exists(meta_file):
        return {}
    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_meta(meta, horizon=5):
    paths = get_snapshot_paths(horizon)
    meta_file = paths["meta_file"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def is_task_running():
    return os.path.exists(RUN_LOCK_FILE)


def mark_task_running(mode, as_of_date=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lock_data = {
        "started_at": str(pd.Timestamp.now()),
        "mode": mode,
        "as_of_date": None if as_of_date is None else str(pd.to_datetime(as_of_date).date()),
        "pid": os.getpid(),
        "stage": "starting",
        "progress": 0.0,
    }
    with open(RUN_LOCK_FILE, "w", encoding="utf-8") as f:
        json.dump(lock_data, f, ensure_ascii=False)


def clear_task_running():
    if os.path.exists(RUN_LOCK_FILE):
        os.remove(RUN_LOCK_FILE)


def load_task_lock():
    if not os.path.exists(RUN_LOCK_FILE):
        return {}
    with open(RUN_LOCK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def process_exists(pid):
    try:
        pid_int = int(pid)
    except Exception:
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
        return True
    except OSError:
        return False


def recover_stale_lock_if_needed():
    if not os.path.exists(RUN_LOCK_FILE):
        return None

    lock_info = load_task_lock()
    started_at = pd.to_datetime(lock_info.get("started_at"), errors="coerce")
    lock_pid = lock_info.get("pid")

    stale_by_pid = lock_pid is not None and (not process_exists(lock_pid))
    stale_by_time = False
    if pd.notna(started_at):
        stale_by_time = (pd.Timestamp.now() - started_at) > pd.Timedelta(minutes=RUN_LOCK_STALE_MINUTES)

    if stale_by_pid or stale_by_time:
        clear_task_running()
        reasons = []
        if stale_by_pid:
            reasons.append(f"pid={lock_pid} not alive")
        if stale_by_time:
            reasons.append(f"age>{RUN_LOCK_STALE_MINUTES}min")
        return "; ".join(reasons)

    return None


def sync_snapshot_to_hdfs(file_paths):
    if not SAVE_SNAPSHOT_TO_HDFS:
        return {"enabled": False, "uploaded": [], "failed": []}

    uploaded = []
    failed = []

    def cli_mkdir_p(path):
        env = os.environ.copy()
        env["HADOOP_USER_NAME"] = HDFS_UPLOAD_USER
        cmd = ["hadoop", "fs", "-mkdir", "-p", path]
        ret = subprocess.run(cmd, env=env, capture_output=True, text=True)
        return ret.returncode == 0, ret.stderr.strip() or ret.stdout.strip()

    def cli_put(local_path, hdfs_path):
        env = os.environ.copy()
        env["HADOOP_USER_NAME"] = HDFS_UPLOAD_USER
        cmd = ["hadoop", "fs", "-put", "-f", local_path, hdfs_path]
        ret = subprocess.run(cmd, env=env, capture_output=True, text=True)
        return ret.returncode == 0, ret.stderr.strip() or ret.stdout.strip()

    hdfs_fs = None
    ok, msg = cli_mkdir_p(HDFS_SNAPSHOT_DIR)
    if not ok:
        try:
            hdfs_fs = get_hdfs_fs()
            hdfs_fs.create_dir(HDFS_SNAPSHOT_DIR, recursive=True)
        except Exception as e:
            return {
                "enabled": True,
                "uploaded": [],
                "failed": [{"path": HDFS_SNAPSHOT_DIR, "error": f"cli={msg}; pyarrow={str(e)}"}],
            }

    for local_path in file_paths:
        hdfs_path = f"{HDFS_SNAPSHOT_DIR.rstrip('/')}/{os.path.basename(local_path)}"
        ok_put, msg_put = cli_put(local_path, hdfs_path)
        if ok_put:
            uploaded.append(hdfs_path)
            continue

        try:
            with open(local_path, "rb") as f_in:
                payload = f_in.read()
            if hdfs_fs is None:
                hdfs_fs = get_hdfs_fs()
            with hdfs_fs.open_output_stream(hdfs_path) as f_out:
                f_out.write(payload)
            uploaded.append(hdfs_path)
        except Exception as e:
            failed.append({"path": hdfs_path, "error": f"cli={msg_put}; pyarrow={str(e)}"})

    return {"enabled": True, "uploaded": uploaded, "failed": failed}


def sync_daily_signal_to_hdfs(daily_df, signal_date):
    if not SAVE_SNAPSHOT_TO_HDFS:
        return {"enabled": False, "uploaded": [], "failed": []}

    date_tag = pd.to_datetime(signal_date).strftime("%Y%m%d")
    local_daily_file = os.path.join(OUTPUT_DIR, f"{date_tag}.parquet")
    daily_df.to_parquet(local_daily_file, index=False)

    def cli_put(local_path, hdfs_path):
        env = os.environ.copy()
        env["HADOOP_USER_NAME"] = HDFS_UPLOAD_USER
        cmd = ["hadoop", "fs", "-put", "-f", local_path, hdfs_path]
        ret = subprocess.run(cmd, env=env, capture_output=True, text=True)
        return ret.returncode == 0, ret.stderr.strip() or ret.stdout.strip()

    def cli_mkdir_p(path):
        env = os.environ.copy()
        env["HADOOP_USER_NAME"] = HDFS_UPLOAD_USER
        cmd = ["hadoop", "fs", "-mkdir", "-p", path]
        ret = subprocess.run(cmd, env=env, capture_output=True, text=True)
        return ret.returncode == 0, ret.stderr.strip() or ret.stdout.strip()

    hdfs_target = f"{HDFS_SNAPSHOT_DIR.rstrip('/')}/{date_tag}.parquet"
    ok_dir, msg_dir = cli_mkdir_p(HDFS_SNAPSHOT_DIR)
    if ok_dir:
        ok_put, msg_put = cli_put(local_daily_file, hdfs_target)
        if ok_put:
            return {"enabled": True, "uploaded": [hdfs_target], "failed": []}

    try:
        hdfs_fs = get_hdfs_fs()
        hdfs_fs.create_dir(HDFS_SNAPSHOT_DIR, recursive=True)
        with open(local_daily_file, "rb") as f_in:
            payload = f_in.read()
        with hdfs_fs.open_output_stream(hdfs_target) as f_out:
            f_out.write(payload)
        return {"enabled": True, "uploaded": [hdfs_target], "failed": []}
    except Exception as e:
        return {
            "enabled": True,
            "uploaded": [],
            "failed": [{"path": hdfs_target, "error": f"cli_mkdir={msg_dir}; pyarrow={str(e)}"}],
        }


def sync_single_file_to_hdfs(local_file_path):
    if not SAVE_SNAPSHOT_TO_HDFS:
        return {"enabled": False, "uploaded": [], "failed": []}

    hdfs_target = f"{HDFS_SNAPSHOT_DIR.rstrip('/')}/{os.path.basename(local_file_path)}"

    env = os.environ.copy()
    env["HADOOP_USER_NAME"] = HDFS_UPLOAD_USER

    def cli_mkdir_p(path):
        cmd = ["hadoop", "fs", "-mkdir", "-p", path]
        ret = subprocess.run(cmd, env=env, capture_output=True, text=True)
        return ret.returncode == 0, ret.stderr.strip() or ret.stdout.strip()

    def cli_put(local_path, hdfs_path):
        cmd = ["hadoop", "fs", "-put", "-f", local_path, hdfs_path]
        ret = subprocess.run(cmd, env=env, capture_output=True, text=True)
        return ret.returncode == 0, ret.stderr.strip() or ret.stdout.strip()

    ok_dir, msg_dir = cli_mkdir_p(HDFS_SNAPSHOT_DIR)
    if ok_dir:
        ok_put, msg_put = cli_put(local_file_path, hdfs_target)
        if ok_put:
            return {"enabled": True, "uploaded": [hdfs_target], "failed": []}

    try:
        hdfs_fs = get_hdfs_fs()
        hdfs_fs.create_dir(HDFS_SNAPSHOT_DIR, recursive=True)
        with open(local_file_path, "rb") as f_in:
            payload = f_in.read()
        with hdfs_fs.open_output_stream(hdfs_target) as f_out:
            f_out.write(payload)
        return {"enabled": True, "uploaded": [hdfs_target], "failed": []}
    except Exception as e:
        return {
            "enabled": True,
            "uploaded": [],
            "failed": [{"path": hdfs_target, "error": f"cli_mkdir={msg_dir}; pyarrow={str(e)}"}],
        }


def upsert_monthly_signal_files(monthly_df, monthly_signal_dir=None):
    target_dir = MONTHLY_SIGNAL_DIR if monthly_signal_dir is None else monthly_signal_dir
    os.makedirs(target_dir, exist_ok=True)
    uploaded = []
    failed = []

    if monthly_df.empty:
        return {"enabled": SAVE_SNAPSHOT_TO_HDFS, "uploaded": uploaded, "failed": failed, "files": []}

    df = monthly_df.copy()
    df["date_dt"] = pd.to_datetime(df["date_dt"])
    df["month_key"] = df["date_dt"].dt.strftime("%Y%m")
    touched = []

    for mk, sdf in df.groupby("month_key"):
        local_file = os.path.join(target_dir, f"{mk}.parquet")
        new_part = sdf.drop(columns=["month_key"]).copy()

        if os.path.exists(local_file):
            old = pd.read_parquet(local_file)
            merged = pd.concat([old, new_part], ignore_index=True)
        else:
            merged = new_part

        key_cols = ["date_dt", "symbol_code", "delivery_code"]
        merged.drop_duplicates(subset=key_cols, keep="last", inplace=True)
        merged.sort_values(key_cols, inplace=True)
        merged.to_parquet(local_file, index=False)
        touched.append(local_file)

        hdfs_ret = sync_single_file_to_hdfs(local_file)
        uploaded.extend(hdfs_ret.get("uploaded", []))
        failed.extend(hdfs_ret.get("failed", []))

    return {
        "enabled": SAVE_SNAPSHOT_TO_HDFS,
        "uploaded": uploaded,
        "failed": failed,
        "files": touched,
    }


def update_task_lock(stage, progress):
    if not os.path.exists(RUN_LOCK_FILE):
        return
    lock_data = load_task_lock()
    lock_data["stage"] = stage
    lock_data["progress"] = float(progress)
    with open(RUN_LOCK_FILE, "w", encoding="utf-8") as f:
        json.dump(lock_data, f, ensure_ascii=False)


def should_run_today(now_ts, meta):
    schedule_ts = now_ts.replace(hour=SCHEDULE_HOUR, minute=SCHEDULE_MINUTE, second=0, microsecond=0)
    last_success_date = meta.get("last_success_date")
    today = str(now_ts.date())
    return (now_ts >= schedule_ts) and (last_success_date != today)


def save_snapshot(result, ensemble_cfg, mode="scheduled", run_as_of_date=None, sync_to_hdfs=True, horizon=5):
    paths = get_snapshot_paths(horizon)
    valid_results_file = paths["valid_results_file"]
    monthly_signal_dir = paths["monthly_signal_dir"]
    meta_file = paths["meta_file"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    latest_date = result["latest_date"]
    train_start = result["train_start"]
    val_start = result["val_start"]
    signals = result["signals"].copy()
    val_metrics = result["val_metrics"]
    val_final_eval = result["val_final_eval"].copy()
    monthly_full_eval = result.get("monthly_full_eval")
    conf_by_symbol = result["conf_by_symbol"].copy()
    today_final = result["today_final"].copy()

    val_df = conf_by_symbol.copy()
    val_df["latest_signal_date"] = str(latest_date.date())
    val_df["max_available_date"] = str(result["max_available_date"].date())
    val_df["train_start"] = str(train_start.date())
    val_df["val_start"] = str(val_start.date())
    val_df["global_val_f1"] = float(val_metrics.get("f1", 0.0))
    val_df["global_val_precision"] = float(val_metrics.get("precision", 0.0))
    val_df["global_val_recall"] = float(val_metrics.get("recall", 0.0))
    val_df.to_parquet(valid_results_file, index=False)

    signal_cols = [
        "date_dt",
        "symbol_code",
        "delivery_code",
        "v_rank",
        "volume",
        "oi",
        "v_share",
        "ensemble_score",
        "is_action",
        "is_real_switch_target",
        "actual_main_code",
    ]
    monthly_source = monthly_full_eval if isinstance(monthly_full_eval, pd.DataFrame) else val_final_eval
    monthly_core = monthly_source[[c for c in signal_cols if c in monthly_source.columns]].copy()

    if "symbol_code" in monthly_core.columns:
        monthly_core = monthly_core.merge(
            conf_by_symbol[["symbol_code", "confidence_f1"]],
            on="symbol_code",
            how="left",
        )
    monthly_core["global_confidence_f1"] = float(val_metrics.get("f1", 0.0))

    latest_signals = signals[[c for c in ["symbol_code", "confidence_f1", "global_confidence_f1"] if c in signals.columns]].copy()
    today_core = today_final[[c for c in signal_cols if c in today_final.columns]].copy()
    if not latest_signals.empty:
        today_core = today_core.merge(latest_signals, on="symbol_code", how="left")

    monthly_payload = pd.concat([monthly_core, today_core], ignore_index=True)
    monthly_sync = upsert_monthly_signal_files(monthly_payload, monthly_signal_dir=monthly_signal_dir)

    if not monthly_payload.empty and ("date_dt" in monthly_payload.columns):
        _m = pd.to_datetime(monthly_payload["date_dt"], errors="coerce").dt.strftime("%Y%m")
        months_written = sorted(_m.dropna().unique().tolist())
    else:
        months_written = []

    meta = {
        "last_update_ts": str(pd.Timestamp.now()),
        "last_success_date": str(pd.Timestamp.now().date()),
        "latest_signal_date": str(latest_date.date()),
        "max_available_date": str(result["max_available_date"].date()),
        "train_start": str(train_start.date()),
        "val_start": str(val_start.date()),
        "val_metrics": val_metrics,
        "horizon": int(horizon),
        "ensemble_cfg": ensemble_cfg,
        "mode": mode,
        "run_as_of_date": None if run_as_of_date is None else str(pd.to_datetime(run_as_of_date).date()),
        "local_snapshot_dir": OUTPUT_DIR,
        "hdfs_snapshot_dir": HDFS_SNAPSHOT_DIR if SAVE_SNAPSHOT_TO_HDFS else None,
        "meta_file": meta_file,
        "valid_results_file": valid_results_file,
        "monthly_signal_dir": monthly_signal_dir,
        "monthly_signal_months": months_written,
    }
    save_meta(meta, horizon=horizon)

    if sync_to_hdfs:
        sync_status = sync_snapshot_to_hdfs([meta_file, valid_results_file])
        daily_status = monthly_sync
    else:
        sync_status = {"enabled": False, "uploaded": [], "failed": []}
        daily_status = {"enabled": False, "uploaded": [], "failed": []}

    meta["hdfs_sync_status"] = sync_status
    meta["hdfs_daily_signal_status"] = daily_status
    save_meta(meta, horizon=horizon)


def load_snapshot_tables(horizon=5):
    paths = get_snapshot_paths(horizon)
    valid_results_file = paths["valid_results_file"]
    monthly_signal_dir = paths["monthly_signal_dir"]

    if not os.path.exists(valid_results_file) or (not os.path.isdir(monthly_signal_dir)):
        return None

    files = [
        os.path.join(monthly_signal_dir, f)
        for f in os.listdir(monthly_signal_dir)
        if f.endswith(".parquet")
    ]
    if len(files) == 0:
        return None

    parts = [pd.read_parquet(p) for p in sorted(files)]
    monthly_all = pd.concat(parts, ignore_index=True)
    monthly_all["date_dt"] = pd.to_datetime(monthly_all["date_dt"])

    return {
        "valid_results": pd.read_parquet(valid_results_file),
        "monthly_all": monthly_all,
    }


def build_meta_from_local_tables(tables, horizon=5):
    paths = get_snapshot_paths(horizon)
    valid_results_file = paths["valid_results_file"]
    monthly_signal_dir = paths["monthly_signal_dir"]
    cfg = HORIZON_CONFIGS.get(int(horizon), ENSEMBLE_CFG)
    latest_signal_date = pd.NaT

    monthly_all = tables.get("monthly_all")
    if monthly_all is not None and (not monthly_all.empty) and ("date_dt" in monthly_all.columns):
        latest_signal_date = pd.to_datetime(monthly_all["date_dt"], errors="coerce").max()

    latest_date_str = "N/A" if pd.isna(latest_signal_date) else str(pd.to_datetime(latest_signal_date).date())
    now_ts = pd.Timestamp.now()
    return {
        "last_update_ts": str(now_ts),
        "last_success_date": str(now_ts.date()),
        "latest_signal_date": latest_date_str,
        "max_available_date": latest_date_str,
        "train_start": "N/A",
        "val_start": "N/A",
        "val_metrics": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
        "horizon": int(horizon),
        "ensemble_cfg": cfg,
        "mode": "local_snapshot_recover",
        "run_as_of_date": None,
        "local_snapshot_dir": OUTPUT_DIR,
        "hdfs_snapshot_dir": HDFS_SNAPSHOT_DIR if SAVE_SNAPSHOT_TO_HDFS else None,
        "valid_results_file": valid_results_file,
        "monthly_signal_dir": monthly_signal_dir,
    }


def render_snapshot(meta, tables, horizon=5):
    cfg = HORIZON_CONFIGS.get(int(horizon), ENSEMBLE_CFG)
    if not meta:
        st.warning("暂无历史结果，请等待首次 18:00 自动刷新。")
        return

    now_ts = pd.Timestamp.now()
    schedule_ts = now_ts.replace(hour=SCHEDULE_HOUR, minute=SCHEDULE_MINUTE, second=0, microsecond=0)
    if now_ts > schedule_ts:
        next_schedule = schedule_ts + pd.Timedelta(days=1)
    else:
        next_schedule = schedule_ts

    st.info(
        f"上次更新时间: {meta.get('last_update_ts', 'N/A')} | "
        f"上次信号日期: {meta.get('latest_signal_date', 'N/A')} | "
        f"下次计划刷新: {next_schedule.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    st.caption(f"本地快照目录: {meta.get('local_snapshot_dir', OUTPUT_DIR)}")
    if meta.get("hdfs_snapshot_dir"):
        st.caption(f"HDFS快照目录: {meta.get('hdfs_snapshot_dir')}")
    if meta.get("mode") == "bootstrap_backfill":
        st.caption(
            f"最大可用数据日期={meta.get('max_available_date', 'N/A')}。"
        )

    vm = meta.get("val_metrics", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Validation F1", f"{vm.get('f1', 0.0):.4f}")
    c2.metric("Validation Precision", f"{vm.get('precision', 0.0):.4f}")
    c3.metric("Validation Recall", f"{vm.get('recall', 0.0):.4f}")
    max_day = max(1, min(int(horizon), 5))
    day_cols = st.columns(max_day)
    for d, col in enumerate(day_cols, start=1):
        col.metric(
            f"Day{d} Hit",
            f"{vm.get(f'day{d}_hit_rate', 0.0):.4f}",
            f"n={int(vm.get(f'day{d}_count', 0))}",
        )

    st.write(f"当前 Ensemble 参数: {meta.get('ensemble_cfg', cfg)}")

    if tables is None:
        st.warning("历史结果文件不完整。请先运行 backup_bootstrap_to_hdfs.py 生成快照。")
        return

    monthly_all = tables["monthly_all"].copy()
    monthly_all["date_dt"] = pd.to_datetime(monthly_all["date_dt"])
    latest_date = monthly_all["date_dt"].max()
    latest_slice = monthly_all[monthly_all["date_dt"] == latest_date].copy()
    second_main_for_filter = (
        latest_slice[latest_slice["v_rank"] == 2][["symbol_code", "volume", "oi"]]
        .drop_duplicates("symbol_code")
        .rename(columns={"volume": "second_main_vol", "oi": "second_main_oi"})
    )

    st.subheader("发出信号的品种")
    signals = latest_slice[latest_slice["is_action"] == True].copy()
    signals = filter_latest_signals_by_second_main_liquidity(signals, second_main_for_filter)
    if signals.empty:
        st.warning("最新一天未触发切换信号。")
    else:
        current_main = latest_slice[latest_slice["delivery_code"] == latest_slice["actual_main_code"]][["symbol_code", "actual_main_code", "volume", "oi"]].drop_duplicates("symbol_code").copy()
        current_main.columns = ["symbol_code", "actual_main_code", "cur_main_vol", "cur_main_oi"]
        out = signals.merge(current_main, on=["symbol_code", "actual_main_code"], how="left")
        out_cols = [
            "date_dt",
            "symbol_code",
            "actual_main_code",
            "delivery_code",
            "cur_main_vol",
            "cur_main_oi",
            "volume",
            "oi",
            "confidence_f1",
        ]
        out = out[[c for c in out_cols if c in out.columns]].copy()
        out["date_dt"] = out["date_dt"].dt.strftime("%Y-%m-%d")
        out = out.rename(
            columns={
                "date_dt": "日期",
                "symbol_code": "品种",
                "actual_main_code": "当前主力",
                "delivery_code": "将成主力",
                "cur_main_vol": "当前主力volume",
                "cur_main_oi": "当前主力oi",
                "volume": "将成主力volume",
                "oi": "将成主力oi",
                "confidence_f1": "品种F1置信度",
            }
        )
        st.dataframe(out, use_container_width=True)

    st.subheader("每个品种当前主力与次主力")
    st.caption("说明：本板块仅按当日 volume>0 的合约排序；主力=Top1、次主力=Top2，并按二者 volume gap 排序。若某品种不足两个 volume>0 合约，则该品种不参与 gap 排名并以 NaN 占位。")

    ranked = latest_slice[["symbol_code", "delivery_code", "volume", "oi"]].copy()
    symbols = ranked[["symbol_code"]].drop_duplicates().copy()
    positive = ranked[ranked["volume"].fillna(0) > 0].copy()
    positive = positive.sort_values(["symbol_code", "volume"], ascending=[True, False])
    positive["vol_pos_rank"] = positive.groupby("symbol_code").cumcount() + 1

    current_main = positive[positive["vol_pos_rank"] == 1][["symbol_code", "delivery_code", "volume", "oi"]].drop_duplicates("symbol_code").copy()
    current_main.columns = ["symbol_code", "current_main_code", "current_main_vol", "current_main_oi"]
    second_main = positive[positive["vol_pos_rank"] == 2][["symbol_code", "delivery_code", "volume", "oi"]].drop_duplicates("symbol_code").copy()
    second_main.columns = ["symbol_code", "second_main_code", "second_main_vol", "second_main_oi"]

    main_second = symbols.merge(current_main, on="symbol_code", how="left").merge(second_main, on="symbol_code", how="left")
    pos_cnt = positive.groupby("symbol_code").size().rename("positive_contract_cnt").reset_index()
    main_second = main_second.merge(pos_cnt, on="symbol_code", how="left")
    main_second["positive_contract_cnt"] = main_second["positive_contract_cnt"].fillna(0).astype(int)
    main_second["volume_gap"] = main_second["current_main_vol"] - main_second["second_main_vol"]
    valid_mask = main_second["positive_contract_cnt"] >= 2
    main_second["gap_rank"] = np.nan
    if valid_mask.any():
        valid_rank = main_second.loc[valid_mask, "volume_gap"].rank(ascending=False, method="dense")
        main_second.loc[valid_mask, "gap_rank"] = valid_rank

    # For symbols with <2 positive-volume contracts, keep placeholders as NaN.
    fill_nan_cols = [
        "current_main_code",
        "current_main_vol",
        "current_main_oi",
        "second_main_code",
        "second_main_vol",
        "second_main_oi",
        "volume_gap",
    ]
    main_second.loc[~valid_mask, fill_nan_cols] = np.nan
    show_cols = [
        "symbol_code",
        "current_main_code",
        "current_main_vol",
        "current_main_oi",
        "second_main_code",
        "second_main_vol",
        "second_main_oi",
        "volume_gap",
        "gap_rank",
    ]
    main_second = main_second[[c for c in show_cols if c in main_second.columns]].copy()
    main_second = main_second.rename(
        columns={
            "symbol_code": "品种",
            "current_main_code": "当前主力",
            "current_main_vol": "当前主力volume",
            "current_main_oi": "当前主力oi",
            "second_main_code": "次主力",
            "second_main_vol": "次主力volume",
            "second_main_oi": "次主力oi",
            "volume_gap": "volume_gap",
            "gap_rank": "gap_rank",
        }
    )
    st.dataframe(main_second.sort_values(["gap_rank", "品种"], na_position="last"), use_container_width=True)

    st.subheader(f"最近 {PLOT_ROLLING_DAYS} 天分品种图")
    plot_start = latest_date - pd.Timedelta(days=PLOT_ROLLING_DAYS)
    plot_df = monthly_all[monthly_all["date_dt"] >= plot_start].copy()
    plot_df["date_dt"] = pd.to_datetime(plot_df["date_dt"])
    identified_symbols = sorted(latest_slice["symbol_code"].dropna().astype(str).unique().tolist())
    render_history_plot(plot_df, symbols=identified_symbols)


meta = load_meta(5)
now_ts = pd.Timestamp.now()
stale_reason = recover_stale_lock_if_needed()

task_running = is_task_running()

def run_scheduled_job():
    with st.spinner("触发 18:00 定时任务：增量更新 + 训练 + 验证 + 推理..."):
        mark_task_running(mode="scheduled", as_of_date=None)
        try:
            update_task_lock("updating_data", 0.10)
            update_msg = update_volume_oi()

            for idx, horizon in enumerate(HORIZON_ORDER, start=1):
                cfg = HORIZON_CONFIGS[horizon]
                progress_base = 0.25 + 0.30 * (idx - 1)
                update_task_lock(f"training_and_inference_t{horizon}", progress_base)
                result = run_pipeline(horizon, cfg)

                update_task_lock(f"saving_snapshot_t{horizon}", progress_base + 0.20)
                save_snapshot(result, cfg, mode="scheduled", run_as_of_date=None, horizon=horizon)

            update_task_lock("completed", 1.0)
            meta = load_meta(5)
        finally:
            clear_task_running()

if (not task_running) and AUTO_RUN_SCHEDULED_ON_PAGE_LOAD and should_run_today(now_ts, meta):
    run_scheduled_job()

if "load_snapshot_now" not in st.session_state:
    st.session_state["load_snapshot_now"] = (not DEFER_HEAVY_RENDER)

if DEFER_HEAVY_RENDER:
    if st.button("加载快照与图表"):
        st.session_state["load_snapshot_now"] = True

if st.session_state.get("load_snapshot_now", False):
    tab_t5, tab_t3 = st.tabs(["T+5", "T+3"])

    with tab_t5:
        meta_t5 = load_meta(5)
        tables_t5 = load_snapshot_tables(5)
        if (not meta_t5) and (tables_t5 is not None):
            meta_t5 = build_meta_from_local_tables(tables_t5, horizon=5)
            save_meta(meta_t5, horizon=5)

        if tables_t5 is None:
            st.warning("未检测到 T+5 快照结果。请先运行：python backup_bootstrap_to_hdfs.py 30")
        else:
            render_snapshot(meta_t5, tables_t5, horizon=5)

    with tab_t3:
        meta_t3 = load_meta(3)
        tables_t3 = load_snapshot_tables(3)
        if (not meta_t3) and (tables_t3 is not None):
            meta_t3 = build_meta_from_local_tables(tables_t3, horizon=3)
            save_meta(meta_t3, horizon=3)

        if tables_t3 is None:
            st.warning("未检测到 T+3 快照结果。请先运行一次自动任务或手动触发。")
        else:
            render_snapshot(meta_t3, tables_t3, horizon=3)
else:
    st.caption("当前为轻量模式：未加载快照数据与图表。")
