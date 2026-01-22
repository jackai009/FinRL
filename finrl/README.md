这个文件夹有三个子文件夹：
+ applications: 交易任务，
+ agents: DRL算法，来自ElegantRL、RLlib或Stable Baselines 3 (SB3)。用户可以插入任何DRL库进行尝试。
+ meta: 市场环境，我们合并了来自活跃的[FinRL-Meta仓库](https://github.com/AI4Finance-Foundation/FinRL-Meta)的稳定版本。

然后，我们通过三个文件采用训练-测试-交易管道：train.py、test.py和trade.py。

```
FinRL
├── finrl (this folder)
│   ├── applications
│   	├── cryptocurrency_trading
│   	├── high_frequency_trading
│   	├── portfolio_allocation
│   	└── stock_trading
│   ├── agents
│   	├── elegantrl
│   	├── rllib
│   	└── stablebaseline3
│   ├── meta
│   	├── data_processors
│   	├── env_cryptocurrency_trading
│   	├── env_portfolio_allocation
│   	├── env_stock_trading
│   	├── preprocessor
│   	├── data_processor.py
│   	└── finrl_meta_config.py
│   ├── config.py
│   ├── config_tickers.py
│   ├── main.py
│   ├── train.py
│   ├── test.py
│   ├── trade.py
│   └── plot.py
```
