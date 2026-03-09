# 未来五天换月预测说明

## 输入

- 数据文件：`contract_volume.parquet`
- 必要字段：
   - `date`：交易日期
   - `symbol_code`：品种代码
   - `delivery_code`：合约代码
   - `volume`：成交量

## 用到的特征

- 日历特征：
   - `is_eve_of_long_holiday`：是否长假前最后交易日
   - `is_before_holiday`：是否节假日前交易日
- 成交量占比特征：
   - `volume_share`：当日该合约成交量占同品种总成交量的比例
- 时序衍生特征（`k in [3, 5, 10]`）：
   - `share_ma_k`：`volume_share` 的 k 日均值
   - `share_lag_k`：`volume_share` 的 k 日滞后
   - `share_v_k`：`volume_share` 的 k 日变化量
   - `share_a_k`：`share_v_k` 的 1 日变化量

## 输出如何解读（未来五天内发生换月的预测结果）

- 模型输出 `prob`：表示某合约在未来 5 个交易日内成为主力合约的概率。
- 每个 `symbol_code + date` 内，`prob` 最高的合约记为 `pred_rank == 1`，即当天最可能发生换月的目标合约。
- 若该合约当天还不是主力（`rank != 1`），则视为一个换月信号
