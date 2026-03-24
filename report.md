# 主力合约切换报告

数据来源：仅基于已跑出的 Notebook/日志结果整理，不重新训练。

## 0) 数据切分与特征工程说明

### 0.1 时间范围

| 训练集 | 测试/验证集 |
|---|---|
| `TEST_START_DATE - 500天` 到 `TEST_START_DATE-1天`（`TEST_START_DATE=2025-11-01`） | `date_dt >= 2025-11-01` |

### 0.2 特征工程（按 Cell2 归类）

| 特征组 | 代表特征 | 作用 |
|---|---|---|
| 量仓结构 | `v_share`, `o_share`, `v_log_ratio`, `oi_v_eff` | 判断合约在日内资金与持仓中的相对地位 |
| 动量与斜率 | `v_share_diff_1/3/5`, `v_share_accel`, `o_share_diff_*`, `o_share_accel` | 捕捉切换前资金迁移速度 |
| 平滑与偏离 | `v_share_ma_3/5`, `v_bias_3/5`, `v_share_hwm_5`, `v_share_hwm_diff` | 提升稳健性，减少尖刺影响 |
| 价格与波动 | `price_chg_*`, `price_vs_v1`, `vol_5d`, `vol_vs_v1` | 融合价格路径与波动状态 |
| 基差与流动性 | `basis_slope_3`, `basis_accel`, `basis_convergence`, `amihud_vs_v1` | 辅助识别临近切换的结构变化 |
| 资金流与持仓接力 | `vol_adj_flow`, `vol_adj_flow_sum_3/5`, `rel_oi_inflow`, `oi_handover_ratio` | 识别“旧主力流出、新主力承接” |
| 时序日历 | `day_of_week`, `month`, `day_of_month`, `days_to_month_end`, `is_month_end_surge`, `months_to_maturity` | 捕捉月末/到期等制度性规律 |
| 共振与状态 | `current_rank2_days`, `days_since_alert`, `v_co_movement`, `in_chaos_zone`, `symbol_cat` | 描述次主力状态与共振强度 |

## 1) 单模型：0/1 主力标签 LightGBM（Rank）

### 1.1 可调参数（你代码中的主参数）

| 参数组 | 参数 | 候选值 |
|---|---|---|
| 采样窗口 | `window_before` | 30, 20, 10, 7, 5 |
| 采样窗口 | `window_after` | 30, 20, 10, 7, 5 |
| 噪声采样 | `noise_frac` | 0.0, 0.1 |
| 树结构 | `num_leaves` | 31, 63, 127 |
| 学习率 | `learning_rate` | 0.05, 0.03, 0.01, 0.001 |
| 列采样 | `feature_fraction` | 0.6, 0.8 |
| boosting 轮数 | `num_boost_round` | 800 |

### 1.2 代表性结果（真实日志）

| 配置 | NDCG@1 | NDCG@2 | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| `L=63, LR=0.03, FF=0.6, WinB=10, WinA=20, Noise=0.0` | 0.9546 | 0.9816 | **0.8273** | 0.8177 | **0.8371** |
| `L=63, LR=0.03, FF=0.6, WinB=10, WinA=30, Noise=0.1` | 0.9540 | 0.9813 | 0.8239 | **0.8215** | 0.8264 |
| `L=63, LR=0.03, FF=0.6, WinB=10, WinA=20, Noise=0.1` | 0.9533 | 0.9814 | 0.8244 | 0.8144 | 0.8347 |

### 1.3 信号序列（最佳 F1 配置）

配置：`L=63, LR=0.03, FF=0.6, WinB=10, WinA=20, Noise=0.0`

| Signal Day | Hit Rate | Count |
|---|---:|---:|
| Day 1 | 0.5667 | 150 |
| Day 2 | 0.7483 | 143 |
| Day 3 | 0.8705 | 139 |
| Day 4 | 0.9394 | 132 |
| Day 5 | 0.9365 | 126 |

结论：信号 F1 约 `0.81~0.83`，Precision/Recall 随采样强度与噪声比例此消彼长。

## 2) 单模型：MLP

### 2.1 模型结构
- `SimpleImputer + StandardScaler + MLPClassifier`
- 通过 `pred_prob` 做横截面选优（每个品种每天概率最高且过阈值）

### 2.2 结果汇总

| 评估集合 | F1 | Precision | Recall |
|---|---:|---:|---:|
| All Symbols | 0.7666 | 0.8006 | 0.7354 |
| Excluded Symbols | 0.7718 | 0.8027 | 0.7431 |

结论：MLP 可作为轻量基线，但整体指标低于三模型融合与强配置 LGBM。

## 3) 三模型融合

### 3.1 三个子模型：输入、目标、输出

| 子模型 | 输入 | Target | 输出 |
|---|---|---|---|
| Rank (`lambdarank`) | `FEATURE_COLS`（含 `symbol_cat`） | `target_rank` | `pred_rank_raw` |
| VShare (`regression`) | 同上 | `target_vshare` | `pred_vshare` |
| Days (`regression`) | 同上 | `target_days` | `pred_days` |

### 3.2 融合方式与公式

先定义：
- $R=\text{Norm}(P_{rank})$，即对同一 `date_dt, symbol_code` 下的 `pred_rank_raw` 做 min-max 归一化。
- $V=\max(0, P_{vshare})$。
- $D=\max(0, P_{days})$。

1) Multiplicative

$$
	ext{Score} = R^{\text{base\_pow}} \cdot \left(1 + \frac{\alpha}{D + \alpha} + \beta \cdot V\right)
$$

2) Additive

$$
	ext{Score} = w_{rank}\cdot R + w_{days}\cdot\frac{1}{D+1} + w_{vshare}\cdot V
$$

3) Exponential

$$
	ext{Score} = R \cdot e^{-\text{decay}\cdot D} \cdot \left(\max(P_{vshare}, 0.001)\right)^{\gamma}
$$

最后在每个 `date_dt, symbol_code` 分组内按 `Score` 降序排名，得到 `ensemble_pred_rank`，并生成切换信号。

### 3.3 融合参数组合（14 trials）

| Trial | Method | 参数 |
|---:|---|---|
| 1 | multiplicative | `alpha=5.0, beta=1.0, base_pow=1.2` |
| 2 | multiplicative | `alpha=10.0, beta=1.5, base_pow=1.5` |
| 3 | multiplicative | `alpha=7.0, beta=1.2, base_pow=1.0` |
| 4 | multiplicative | `alpha=3.0, beta=0.8, base_pow=1.1` |
| 5 | multiplicative | `alpha=2.0, beta=2.0, base_pow=1.3` |
| 6 | multiplicative | `alpha=8.0, beta=0.5, base_pow=1.0` |
| 7 | additive | `w_rank=1.0, w_days=1.0, w_vshare=2.0` |
| 8 | additive | `w_rank=1.5, w_days=2.0, w_vshare=1.0` |
| 9 | additive | `w_rank=1.0, w_days=0.5, w_vshare=3.0` |
| 10 | additive | `w_rank=2.0, w_days=3.0, w_vshare=0.5` |
| 11 | exponential | `decay=0.1, gamma=1.0` |
| 12 | exponential | `decay=0.2, gamma=0.5` |
| 13 | exponential | `decay=0.05, gamma=2.0` |
| 14 | exponential | `decay=0.3, gamma=1.5` |

### 3.4 14组真实结果（你提供日志）

| Trial | Excluded F1 | All F1 | Excluded Precision | Excluded Recall |
|---:|---:|---:|---:|---:|
| 1 | 0.9396 | 0.9051 | 0.9391 | 0.9401 |
| 2 | 0.9342 | 0.8995 | 0.9341 | 0.9343 |
| 3 | 0.9399 | 0.9056 | 0.9394 | 0.9404 |
| 4 | 0.9427 | 0.9085 | 0.9423 | 0.9432 |
| 5 | 0.9406 | 0.9065 | 0.9400 | 0.9412 |
| 6 | 0.9390 | 0.9048 | 0.9385 | 0.9395 |
| 7 | 0.9425 | 0.9110 | 0.9426 | 0.9424 |
| 8 | 0.9165 | 0.8867 | 0.9162 | 0.9168 |
| 9 | 0.9505 | 0.9179 | 0.9508 | 0.9502 |
| 10 | 0.8927 | 0.8637 | 0.8921 | 0.8932 |
| 11 | 0.9464 | 0.9144 | 0.9466 | 0.9463 |
| 12 | 0.9376 | 0.9068 | 0.9375 | 0.9377 |
| 13 | **0.9511** | **0.9184** | **0.9511** | **0.9510** |
| 14 | 0.9393 | 0.9081 | 0.9392 | 0.9393 |

### 3.5 按融合方法聚合（便于看趋势）

| Method | Excluded F1 均值 | All F1 均值 |
|---|---:|---:|
| Multiplicative (Trial 1-6) | 0.9393 | 0.9050 |
| Additive (Trial 7-10) | 0.9256 | 0.8948 |
| Exponential (Trial 11-14) | **0.9436** | **0.9119** |

结论：
- 当前日志中最佳是 **Trial 13（exponential: decay=0.05, gamma=2.0）**。
- 三模型融合的信号指标明显高于单模型。
- Additive 组波动最大，参数不当时会显著拖低 F1（如 Trial 10）。

