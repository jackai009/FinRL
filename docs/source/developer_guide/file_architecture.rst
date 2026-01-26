:github_url: https://github.com/AI4Finance-Foundation/FinRL

=================
文件架构
=================

FinRL的文件架构严格遵循:ref:`三层架构`。

.. code:: bash

    FinRL
    ├── finrl (主文件夹)
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
    └───└── plot.py
