import ast
import json
import os
import subprocess
import sys
import time
import warnings

import chinese_calendar
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyarrow import fs
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

BACKFILL_SIGNAL_DAYS = 90

# Constants loaded from main.py to keep behavior aligned.
CONST_NAMES = {
    "OLD_FOLDER",
    "NEW_FOLDER",
    "CUTOFF_DATE",
    "START_DATE_DEFAULT",
    "PARQUET_FILE",
    "RANDOM_SEED",
    "LOOKBACK_TRAIN_DAYS",
    "VALIDATION_DAYS",
    "VALIDATION_MONTH_SHIFT",
    "PLOT_ROLLING_DAYS",
    "SCHEDULE_HOUR",
    "SCHEDULE_MINUTE",
    "AUTO_REFRESH_SECONDS",
    "AUTO_REFRESH_ENABLED",
    "AUTO_BOOTSTRAP_IF_EMPTY",
    "BOOTSTRAP_DAYS_AGO",
    "SAVE_SNAPSHOT_TO_HDFS",
    "HDFS_SNAPSHOT_DIR",
    "HDFS_UPLOAD_USER",
    "RUN_LOCK_STALE_MINUTES",
    "OUTPUT_DIR",
    "SNAPSHOT_META_FILE",
    "VALID_RESULTS_FILE",
    "MONTHLY_SIGNAL_DIR",
    "SNAPSHOT_SIGNALS_FILE",
    "SNAPSHOT_MAIN_SECOND_FILE",
    "SNAPSHOT_CONF_FILE",
    "SNAPSHOT_HIST_FILE",
    "SNAPSHOT_IMP_RANK_FILE",
    "SNAPSHOT_IMP_VSHARE_FILE",
    "SNAPSHOT_IMP_DAYS_FILE",
    "RUN_LOCK_FILE",
    "SAMPLE_CFG_RANK",
    "SAMPLE_CFG_VSHARE",
    "SAMPLE_CFG_DAYS",
    "RANK_CFG",
    "VSHARE_CFG",
    "DAYS_CFG",
    "CONFIG_T5",
    "CONFIG_T3",
    "HORIZON_CONFIGS",
    "HORIZON_ORDER",
    "ENSEMBLE_CFG",
    "ENSEMBLE_EXAMPLES",
    "FEATURE_COLS",
    "EXCLUDED_SYMBOLS",
    "hdfs_fs",
}


def _extract_target_names(node):
    names = []
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name):
                names.append(t.id)
    elif isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return names


def load_main_functions_and_constants(main_path):
    with open(main_path, "r", encoding="utf-8") as f:
        src = f.read()

    mod = ast.parse(src, filename=main_path)
    picked_nodes = []

    for node in mod.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target_names = _extract_target_names(node)
            if any(n in CONST_NAMES for n in target_names):
                picked_nodes.append(node)
        elif isinstance(node, ast.FunctionDef):
            picked_nodes.append(node)

    runtime_mod = ast.Module(body=picked_nodes, type_ignores=[])
    ast.fix_missing_locations(runtime_mod)

    import streamlit as st

    runtime_ns = {
        "__builtins__": __builtins__,
        "os": os,
        "subprocess": subprocess,
        "json": json,
        "warnings": warnings,
        "np": np,
        "pd": pd,
        "pq": pq,
        "fs": fs,
        "lgb": lgb,
        "tqdm": tqdm,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "chinese_calendar": chinese_calendar,
        "st": st,
    }

    code = compile(runtime_mod, main_path, "exec")
    exec(code, runtime_ns)
    return runtime_ns


def _fmt_seconds(seconds):
    return f"{seconds:.2f}s"


def _print_step(step_idx, step_total, title, step_start, global_start):
    elapsed = time.perf_counter() - step_start
    global_elapsed = time.perf_counter() - global_start
    pct = int(step_idx * 100 / step_total)
    bar_len = 24
    fill = int(bar_len * pct / 100)
    bar = "#" * fill + "-" * (bar_len - fill)
    print(
        f"[PROGRESS] [{bar}] {pct:>3}% | step {step_idx}/{step_total} {title} | "
        f"step_time={_fmt_seconds(elapsed)} | total_time={_fmt_seconds(global_elapsed)}"
    )


def _monthly_validation_start(latest_date, floor_date, month_shift):
    latest_month_start = pd.Timestamp(latest_date).to_period("M").to_timestamp()
    val_start = latest_month_start - pd.DateOffset(months=int(month_shift))
    floor_ts = pd.Timestamp(floor_date)
    if val_start <= floor_ts:
        val_start = floor_ts + pd.Timedelta(days=1)
    if val_start >= pd.Timestamp(latest_date):
        val_start = pd.Timestamp(latest_date) - pd.Timedelta(days=1)
    return val_start


def run_pipeline_with_timing(ns, horizon, model_cfg, as_of_date=None):
    # New main.py path: run_pipeline(horizon, cfg, as_of_date)
    if "run_pipeline" in ns:
        return ns["run_pipeline"](horizon, model_cfg, as_of_date=as_of_date)

    # Legacy fallback for older main.py implementations.
    prepare_data_and_targets = ns["prepare_data_and_targets"]
    train_models = ns["train_models"]
    infer_with_models = ns["infer_with_models"]
    evaluate_trial = ns["evaluate_trial"]
    symbol_confidence_from_eval = ns["symbol_confidence_from_eval"]

    lookback_train_days = ns["LOOKBACK_TRAIN_DAYS"]
    validation_month_shift = int(ns.get("VALIDATION_MONTH_SHIFT", 3))
    feature_cols_all = ns["FEATURE_COLS"]
    parquet_file = ns["PARQUET_FILE"]

    stage_total = 7
    global_start = time.perf_counter()

    stage_start = time.perf_counter()
    print("[STEP 1/7] 准备数据与特征...")
    try:
        df = prepare_data_and_targets(parquet_file, horizon)
    except TypeError:
        df = prepare_data_and_targets(parquet_file)
    feat_cols = [c for c in feature_cols_all if c in df.columns]
    _print_step(1, stage_total, "准备数据与特征", stage_start, global_start)

    stage_start = time.perf_counter()
    print("[STEP 2/7] 切分窗口（lookback/monthly validation）...")
    max_available_date = df["date_dt"].max()
    if as_of_date is None:
        latest_date = max_available_date
    else:
        as_of_ts = pd.to_datetime(as_of_date)
        candidate = df.loc[df["date_dt"] <= as_of_ts, "date_dt"]
        if candidate.empty:
            raise RuntimeError("as_of_date 早于数据起始日期，无法回放。")
        latest_date = candidate.max()

    train_start = latest_date - pd.Timedelta(days=lookback_train_days)

    target_future_col = f"future_main_t{horizon}"
    if target_future_col not in df.columns:
        target_future_col = "future_main_t5"
    train_pool_all = df[(~df["is_excluded"]) & (df[target_future_col].notna())].copy()
    lookback_pool = train_pool_all[(train_pool_all["date_dt"] >= train_start) & (train_pool_all["date_dt"] < latest_date)].copy()
    if lookback_pool.empty:
        raise RuntimeError("lookback 窗口内没有可训练样本。")

    val_start = _monthly_validation_start(latest_date, lookback_pool["date_dt"].min(), validation_month_shift)
    val_train = lookback_pool[lookback_pool["date_dt"] < val_start].copy()
    val_eval = lookback_pool[lookback_pool["date_dt"] >= val_start].copy()
    if val_train.empty or val_eval.empty:
        raise RuntimeError("验证窗口为空，请调大 LOOKBACK_TRAIN_DAYS 或调小 VALIDATION_MONTH_SHIFT。")

    print(
        f"  - latest_date={latest_date.date()} | train_start={train_start.date()} | "
        f"val_start={val_start.date()} | lookback_pool={len(lookback_pool)}"
    )
    _print_step(2, stage_total, "切分窗口", stage_start, global_start)

    stage_start = time.perf_counter()
    print("[STEP 3/7] 第一次训练（训练集）...")
    m_rank_val, m_vshare_val, m_days_val = train_models(val_train, feat_cols, model_cfg)
    _print_step(3, stage_total, "第一次训练（训练集）", stage_start, global_start)

    stage_start = time.perf_counter()
    print("[STEP 4/7] 跑验证集并计算 confidence/F1...")
    val_final_eval = infer_with_models(
        val_eval,
        feat_cols,
        m_rank_val,
        m_vshare_val,
        m_days_val,
        model_cfg,
        apply_window_filter=False,
    )
    val_metrics = evaluate_trial(val_final_eval)
    conf_by_symbol = symbol_confidence_from_eval(val_final_eval)
    print(
        f"  - val_f1={val_metrics['f1']:.4f}, val_precision={val_metrics['precision']:.4f}, "
        f"val_recall={val_metrics['recall']:.4f}"
    )
    print(
        "  - day_hit: "
        + ", ".join(
            [
                f"d{d}={val_metrics.get(f'day{d}_hit_rate', 0.0):.4f}"
                f"(n={int(val_metrics.get(f'day{d}_count', 0))})"
                for d in range(1, 6)
            ]
        )
    )
    _print_step(4, stage_total, "验证推理与评估", stage_start, global_start)

    stage_start = time.perf_counter()
    print("[STEP 5/7] 第二次训练（全量窗口一次）...")
    full_train_pool = train_pool_all[train_pool_all["date_dt"] < latest_date].copy()
    if full_train_pool.empty:
        raise RuntimeError("全量训练窗口为空，无法进行第二次训练。")
    m_rank, m_vshare, m_days = train_models(full_train_pool, feat_cols, model_cfg)
    _print_step(5, stage_total, "第二次训练（全量）", stage_start, global_start)

    stage_start = time.perf_counter()
    print("[STEP 6/7] 生成全日期信号（验证起点后所有天）...")
    backfill_start = latest_date - pd.Timedelta(days=BACKFILL_SIGNAL_DAYS - 1)
    infer_start = max(val_start, backfill_start)
    infer_all = df[(df["date_dt"] >= infer_start) & (df["date_dt"] <= latest_date)].copy()
    all_eval = infer_with_models(
        infer_all,
        feat_cols,
        m_rank,
        m_vshare,
        m_days,
        model_cfg,
        apply_window_filter=False,
    )

    main_stats = all_eval[all_eval["daily_rank"] == 1][
        ["date_dt", "symbol_code", "delivery_code", "volume", "oi"]
    ].drop_duplicates(["date_dt", "symbol_code", "delivery_code"]).copy()
    main_stats.columns = ["date_dt", "symbol_code", "actual_main_code", "cur_main_vol", "cur_main_oi"]
    all_eval = all_eval.merge(main_stats, on=["date_dt", "symbol_code", "actual_main_code"], how="left")
    if "current_main_code" not in all_eval.columns and "actual_main_code" in all_eval.columns:
        all_eval["current_main_code"] = all_eval["actual_main_code"]

    signals = all_eval[(all_eval["date_dt"] == latest_date) & (all_eval["is_action"])].copy()
    all_days_signals = all_eval[all_eval["is_action"]].copy()
    today_final = all_eval[all_eval["date_dt"] == latest_date].copy()

    # Keep output schema aligned with main.run_pipeline -> save_snapshot contract.
    latest_slice = all_eval[all_eval["date_dt"] == latest_date].copy()
    current_main = (
        latest_slice[latest_slice["daily_rank"] == 1][["symbol_code", "delivery_code", "volume", "oi"]]
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
        latest_slice[latest_slice["daily_rank"] == 99][["symbol_code", "delivery_code", "volume", "oi"]]
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

    if not signals.empty:
        signals = signals.merge(
            main_second[["symbol_code", "second_main_vol", "second_main_oi"]],
            on="symbol_code",
            how="left",
        )
        pred_has_liq = signals["volume"].fillna(0).gt(0) | signals["oi"].fillna(0).gt(0)
        second_has_liq = signals["second_main_vol"].fillna(0).gt(0) | signals["second_main_oi"].fillna(0).gt(0)
        signals = signals[pred_has_liq | second_has_liq].copy()
        signals = signals.drop(columns=["second_main_vol", "second_main_oi"], errors="ignore")

    if not signals.empty:
        signals = signals.merge(conf_by_symbol[["symbol_code", "confidence_f1"]], on="symbol_code", how="left")
        signals["global_confidence_f1"] = val_metrics["f1"]

    if not all_days_signals.empty:
        all_days_signals = all_days_signals.merge(conf_by_symbol[["symbol_code", "confidence_f1"]], on="symbol_code", how="left")
        all_days_signals["global_confidence_f1"] = val_metrics["f1"]

    monthly_rows = []
    for month, mdf in val_final_eval.groupby(val_final_eval["date_dt"].dt.to_period("M")):
        y_true = mdf["is_real_switch_target"].astype(int)
        y_pred = mdf["is_action"].astype(int)
        monthly_rows.append(
            {
                "month": str(month),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "n_samples": int(len(mdf)),
            }
        )
    val_metrics_monthly = pd.DataFrame(monthly_rows).sort_values("month") if monthly_rows else pd.DataFrame()

    print(
        f"  - latest_day_signals={len(signals)} | all_days_signals={len(all_days_signals)} "
        f"| backfill_start={infer_start.date()}"
    )
    all_months = sorted(all_eval["date_dt"].dt.strftime("%Y%m").dropna().unique().tolist())
    print(f"  - months_covered={all_months}")
    _print_step(6, stage_total, "生成全日期信号", stage_start, global_start)

    stage_start = time.perf_counter()
    print("[STEP 7/7] 打包输出结果...")
    result = {
        "df": df,
        "max_available_date": max_available_date,
        "latest_date": latest_date,
        "train_start": train_start,
        "val_start": val_start,
        "feat_cols": feat_cols,
        "signals": signals,
        "val_metrics": val_metrics,
        "val_final_eval": val_final_eval,
        "monthly_full_eval": all_eval,
        "conf_by_symbol": conf_by_symbol,
        "main_second": main_second,
        "today_final": today_final,
        "m_rank": m_rank,
        "m_vshare": m_vshare,
        "m_days": m_days,
        "all_days_signals": all_days_signals,
        "val_metrics_monthly": val_metrics_monthly,
    }
    _print_step(7, stage_total, "打包输出结果", stage_start, global_start)

    print(f"[TIMING] pipeline_total={_fmt_seconds(time.perf_counter() - global_start)}")
    return result


def main():
    days_ago = None
    if len(sys.argv) > 1:
        try:
            days_ago = int(sys.argv[1])
        except ValueError:
            print(f"[ERROR] Invalid days_ago argument: {sys.argv[1]}")
            print("Usage: python backup_bootstrap_to_hdfs.py [days_ago]")
            sys.exit(2)

    main_path = os.path.join(os.path.dirname(__file__), "main.py")
    ns = load_main_functions_and_constants(main_path)

    update_volume_oi = ns["update_volume_oi"]
    save_snapshot = ns["save_snapshot"]
    load_meta = ns["load_meta"]
    horizon_configs = ns.get("HORIZON_CONFIGS", {5: ns["ENSEMBLE_CFG"]})

    # Force HDFS upload in backup mode, as requested.
    ns["SAVE_SNAPSHOT_TO_HDFS"] = True

    global_start = time.perf_counter()

    step_start = time.perf_counter()
    print("[STAGE 1/3] Incremental data sync...")
    update_msg = update_volume_oi()
    print(update_msg)
    print(f"[TIMING] data_sync={_fmt_seconds(time.perf_counter() - step_start)}")

    bootstrap_as_of = None if days_ago is None else (pd.Timestamp.now() - pd.Timedelta(days=days_ago))
    as_of_desc = "latest_available" if bootstrap_as_of is None else str(bootstrap_as_of.date())
    print(f"[STAGE 2/3] Run bootstrap pipeline as_of={as_of_desc}...")
    all_horizons = [h for h in [5, 3] if h in horizon_configs]
    if not all_horizons:
        all_horizons = [5]

    step_start = time.perf_counter()
    for horizon in all_horizons:
        cfg = horizon_configs[horizon]
        print(f"  - Running T+{horizon} pipeline...")
        result = run_pipeline_with_timing(ns, horizon, cfg, as_of_date=bootstrap_as_of)

        print(f"  - Saving T+{horizon} snapshot...")
        save_snapshot(
            result,
            cfg,
            mode="bootstrap_backfill",
            run_as_of_date=bootstrap_as_of,
            sync_to_hdfs=True,
            horizon=horizon,
        )

    print(f"[TIMING] snapshot_save_and_hdfs={_fmt_seconds(time.perf_counter() - step_start)}")

    print("\n[DONE] Bootstrap snapshot completed.")
    any_failed = False
    for horizon in all_horizons:
        meta = load_meta(horizon)
        hdfs_sync = meta.get("hdfs_sync_status", {})
        hdfs_daily = meta.get("hdfs_daily_signal_status", {})
        print(f"[T+{horizon}] latest_signal_date={meta.get('latest_signal_date')}")
        print(f"[T+{horizon}] hdfs_snapshot_uploaded={len(hdfs_sync.get('uploaded', []))}")
        print(f"[T+{horizon}] hdfs_snapshot_failed={len(hdfs_sync.get('failed', []))}")
        print(f"[T+{horizon}] hdfs_daily_uploaded={len(hdfs_daily.get('uploaded', []))}")
        print(f"[T+{horizon}] hdfs_daily_failed={len(hdfs_daily.get('failed', []))}")
        if hdfs_sync.get("failed") or hdfs_daily.get("failed"):
            any_failed = True
    print(f"[TIMING] bootstrap_total={_fmt_seconds(time.perf_counter() - global_start)}")

    if any_failed:
        print("[WARN] Some HDFS uploads failed. See latest_snapshot_meta.json for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
