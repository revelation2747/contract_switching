# 期货主力合约切换预测系统（Futures Major Contract Switch Prediction System）

1. depth数据更新读入
2. 训练模型（T-5之前）
  在合约维度，构造lag和移动平均的特征作为X_features输入，构造 T+5是否是主力合约（0/1） 作为y_targat
  但在这里修改了一下损失函数，给miss掉信号的情况赋予了更高的惩罚权重，所以classifier会退化成regressor
1. 信号输出
  在合约维度，预测该合约在T+5是主力合约的条件概率，将概率排序，判断概率最大的合约与当前主力合约是否一致，不一致则发出信号

output sample：

[DAILY SIGNAL REPORT] | 2026-03-12
symbol_code Current_Major Target_Major  Confidence  Current_Vol  Target_Vol
         BR          2604         2605    0.722689       329538      257735
         AD          2604         2605    0.685336         6862        4888
         AL          2604         2605    0.668363       319157      278849
         PF          2604         2606    0.661580       167015      148036
         CU          2604         2605    0.637586        87960       62188
          V          2605         2701    0.615281        37603       20895
         SC          2604         2605    0.613598       165535      111177
         SF          2605         2607    0.593299       117753       78086
         WR          2605         2607    0.584973          114          59
          L          2605         2701    0.551561         5750        2351
# contract_switching
