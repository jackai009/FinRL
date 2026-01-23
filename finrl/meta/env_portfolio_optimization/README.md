# 投资组合优化环境 (POE)

该环境模拟市场对投资组合的影响，该投资组合通过强化学习智能体定期重新平衡。在每个时间步$t$，智能体负责确定一个投资组合向量$W_{t}$，其中包含投资于每只股票的资金百分比。然后，环境利用用户提供的数据模拟时间步$t+1$时的新投资组合价值。

有关此问题公式化的更多详细信息，请查看以下论文：

[POE：FinRL的通用投资组合优化环境](https://doi.org/10.5753/bwaif.2023.231144)
```
@inproceedings{bwaif,
 author = {Caio Costa and Anna Costa},
 title = {POE: A General Portfolio Optimization Environment for FinRL},
 booktitle = {Anais do II Brazilian Workshop on Artificial Intelligence in Finance},
 location = {João Pessoa/PB},
 year = {2023},
 keywords = {},
 issn = {0000-0000},
 pages = {132--143},
 publisher = {SBC},
 address = {Porto Alegre, RS, Brasil},
 doi = {10.5753/bwaif.2023.231144},
 url = {https://sol.sbc.org.br/index.php/bwaif/article/view/24959}
}
```

## 输入
该环境基于数据框提供的数据模拟智能体与金融市场之间的交互。数据框包含用户定义的特征时间序列（例如收盘价、最高价和最低价），并且必须具有时间列和股票代码列，分别包含日期时间列表和股票代码符号列表。
数据框示例如下：
``````
    date        high            low             close           tic
0   2020-12-23  0.157414        0.127420        0.136394        ADA-USD
1   2020-12-23  34.381519       30.074295       31.097898       BNB-USD
2   2020-12-23  24024.490234    22802.646484    23241.345703    BTC-USD
3   2020-12-23  0.004735        0.003640        0.003768        DOGE-USD
4   2020-12-23  637.122803      560.364258      583.714600      ETH-USD
... ...         ...             ...             ...             ...
``````

## 动作

在每个时间步，环境期望一个形状为(n+1,)的一维Box动作，其中$n$是投资组合中的股票数量。这个动作称为*投资组合向量*，包含剩余现金和每只股票的分配资金百分比。

例如：给定一个包含三只股票的投资组合，一个有效的投资组合向量为$W_{t} = [0.25, 0.4, 0.2, 0.15]$。在此示例中，25%的资金未投资（剩余现金），40%投资于股票1，20%投资于股票2，15%投资于股票3。

**注意：** 投资组合向量中的值之和等于（或非常接近）1非常重要。如果不是，POE将应用softmax归一化。

## 观测

POE在模拟过程中可以返回两种类型的观测：Dict或Box。

- Box是一个形状为$(f, n, t)$的三维数组，其中$f$是特征数量，$n$是投资组合中的股票数量，$t$是时间序列时间窗口。该观测基本上只包含智能体的当前状态。

- 另一方面，字典表示是包含状态和上一个投资组合向量的字典，如下所示：

```json
{
"state": "three-dimensional Box (f, n, t representing the time series",
"last_action": "one-dimensional Box (n+1,) representing the portfolio weights"
}
```

## 奖励
给定时间步$t$的模拟，奖励由以下公式给出：$r_{t} = ln(V_{t}/V_{t-1})$，其中$V_{t}$是时间$t$时的投资组合价值。通过使用此公式，每当投资组合价值因重新平衡而减少时，奖励为负，否则为正。

## 示例
使用此环境的Jupyter笔记本可以在[此处](/examples/FinRL_PortfolioOptimizationEnv_Demo.ipynb)找到。
