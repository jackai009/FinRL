# Stock_NeurIPS2018

我们展示了在算法交易中应用强化学习的工作流程，这是对[NeurIPS 2018论文](https://arxiv.org/abs/1811.07522)中过程的复现和改进。

# 使用方法

## 步骤一：数据

首先，运行笔记本：*Stock_NeurIPS2018_1_Data.ipynb*。

它会下载并预处理股票的OHLCV数据。

它会生成两个csv文件：*train.csv* 和 *trade.csv*。您可以查看提供的两个示例文件。

## 步骤二：训练交易智能体

其次，运行笔记本：*Stock_NeurIPS2018_2_Train.ipynb*。

它展示了如何将数据处理成OpenAI gym风格的环境，然后训练一个深度强化学习智能体。

它会生成一个训练好的RL模型.zip文件。此外，我们还提供了一个训练好的A2C模型.zip文件。

## 步骤三：回测

最后，运行笔记本：*Stock_NeurIPS2018_3_Backtest.ipynb*。

它对训练好的智能体进行回测，并分别与两个基准进行比较：均值-方差优化和市场DJIA指数。

最后，它会绘制回测过程中投资组合价值的图表。
