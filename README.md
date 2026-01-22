<div align="center">
<img align="center" width="30%" alt="image" src="https://github.com/AI4Finance-Foundation/FinGPT/assets/31713746/e0371951-1ce1-488e-aa25-0992dafcc139">
</div>

# FinRL®: 金融强化学习 [![twitter][1.1]][1] [![facebook][1.2]][2] [![google+][1.3]][3] [![linkedin][1.4]][4]

[1.1]: http://www.tensorlet.org/wp-content/uploads/2021/01/button_twitter_22x22.png
[1.2]: http://www.tensorlet.org/wp-content/uploads/2021/01/facebook-button_22x22.png
[1.3]: http://www.tensorlet.org/wp-content/uploads/2021/01/button_google_22.xx_.png
[1.4]: http://www.tensorlet.org/wp-content/uploads/2021/01/button_linkedin_22x22.png

[1]: https://twitter.com/intent/tweet?text=FinRL-Financial-Deep-Reinforcement-Learning%20&url=https://github.com/AI4Finance-Foundation/FinRL&hashtags=DRL&hashtags=AI
[2]: https://www.facebook.com/sharer.php?u=http%3A%2F%2Fgithub.com%2FAI4Finance-Foundation%2FFinRL
[3]: https://plus.google.com/share?url=https://github.com/AI4Finance-Foundation/FinRL
[4]: https://www.linkedin.com/sharing/share-offsite/?url=http%3A%2F%2Fgithub.com%2FAI4Finance-Foundation%2FFinRL

<div align="center">
<img align="center" src=figs/logo_transparent_background.png width="55%"/>
</div>

[![Downloads](https://static.pepy.tech/badge/finrl)](https://pepy.tech/project/finrl)
[![Downloads](https://static.pepy.tech/badge/finrl/week)](https://pepy.tech/project/finrl)
[![Join Discord](https://img.shields.io/badge/Discord-Join-blue)](https://discord.gg/trsr8SXpW5)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/finrl.svg)](https://pypi.org/project/finrl/)
[![Documentation Status](https://readthedocs.org/projects/finrl/badge/?version=latest)](https://finrl.readthedocs.io/en/latest/?badge=latest)
![License](https://img.shields.io/github/license/AI4Finance-Foundation/finrl.svg?color=brightgreen)
![](https://img.shields.io/github/issues-raw/AI4Finance-Foundation/finrl?label=Issues)
![](https://img.shields.io/github/issues-closed-raw/AI4Finance-Foundation/finrl?label=Closed+Issues)
![](https://img.shields.io/github/issues-pr-raw/AI4Finance-Foundation/finrl?label=Open+PRs)
![](https://img.shields.io/github/issues-pr-closed-raw/AI4Finance-Foundation/finrl?label=Closed+PRs)

[FinGPT](https://github.com/AI4Finance-Foundation/ChatGPT-for-FinTech): Open-source for open-finance! Revolutionize FinTech.


[![](https://dcbadge.vercel.app/api/server/trsr8SXpW5)](https://discord.gg/trsr8SXpW5)

![Visitors](https://api.visitorbadge.io/api/VisitorHit?user=AI4Finance-Foundation&repo=FinRL&countColor=%23B17A)
[![](https://dcbadge.limes.pink/api/server/trsr8SXpW5?cb=1)](https://discord.gg/trsr8SXpW5)



**金融强化学习（FinRL®）** ([文档网站](https://finrl.readthedocs.io/en/latest/index.html)) 是**第一个金融强化学习开源框架**。FinRL已经发展成为一个**生态系统**
* [FinRL-DeepSeek](https://github.com/AI4Finance-Foundation/FinRL_DeepSeek): 面向交易智能体的LLM融合风险敏感强化学习

| 开发路线图  | 阶段 | 用户 | 项目 | 描述 |
|----|----|----|----|----|
| 0.0 (准备) | 入口 | 从业者 | [FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Meta)| gym风格的市场环境 |
| 1.0 (概念验证)| 全栈 | 开发者 | [本仓库](https://github.com/AI4Finance-Foundation/FinRL) | 自动化管道 |
| 2.0 (专业) | 专业 | 专家 | [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL) | 算法 |
| 3.0 (生产) | 服务 | 对冲基金 | [Podracer](https://github.com/AI4Finance-Foundation/FinRL_Podracer) | 云原生部署 |


## 大纲

  - [概述](#概述)
  - [文件结构](#文件结构)
  - [支持的数据源](#支持的数据源)
  - [安装](#安装)
  - [状态更新](#状态更新)
  - [教程](#教程)
  - [出版物](#出版物)
  - [新闻](#新闻)
  - [引用FinRL](#引用finrl)
  - [加入和贡献](#加入和贡献)
    - [贡献者](#贡献者)
    - [Sponsorship](#赞助)
  - [许可证](#许可证)

## 概述

FinRL有三个层次：市场环境、智能体和应用。对于交易任务（顶层），智能体（中间层）与市场环境（底层）交互，做出顺序决策。

<div align="center">
<img align="center" src=figs/finrl_framework.png>
</div>

快速开始：Stock_NeurIPS2018.ipynb。视频 [FinRL](http://www.youtube.com/watch?v=ZSGJjtM-5jA) 在 [AI4Finance Youtube频道](https://www.youtube.com/channel/UCrVri6k3KPBa3NhapVV4K5g)。


## 文件结构

主文件夹 **finrl** 有三个子文件夹 **applications, agents, meta**。我们采用 **训练-测试-交易** 管道，包含三个文件：train.py、test.py 和 trade.py。

```
FinRL
├── finrl (main folder)
│   ├── applications
│   	├── Stock_NeurIPS2018
│   	├── imitation_learning
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
│       ├── meta_config_tickers.py
│   	└── meta_config.py
│   ├── config.py
│   ├── config_tickers.py
│   ├── main.py
│   ├── plot.py
│   ├── train.py
│   ├── test.py
│   └── trade.py
│
├── examples
├── unit_tests (unit tests to verify codes on env & data)
│   ├── environments
│   	└── test_env_cashpenalty.py
│   └── downloaders
│   	├── test_yahoodownload.py
│   	└── test_alpaca_downloader.py
├── setup.py
├── requirements.txt
└── README.md
```

## 支持的数据源

|数据源 |类型 |范围与频率 |请求限制|原始数据|预处理数据|
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|[Akshare](https://alpaca.markets/docs/introduction/)| CN Securities| 2015-now, 1day| Account-specific| OHLCV| Prices&Indicators|
|[Alpaca](https://docs.alpaca.markets/docs/getting-started)| US Stocks, ETFs| 2015-now, 1min| Account-specific| OHLCV| Prices&Indicators|
|[Baostock](http://baostock.com/baostock/index.php/Python_API%E6%96%87%E6%A1%A3)| CN Securities| 1990-12-19-now, 5min| Account-specific| OHLCV| Prices&Indicators|
|[Binance](https://binance-docs.github.io/apidocs/spot/en/#public-api-definitions)| Cryptocurrency| API-specific, 1s, 1min| API-specific| Tick-level daily aggegrated trades, OHLCV| Prices&Indicators|
|[CCXT](https://docs.ccxt.com/en/latest/manual.html)| Cryptocurrency| API-specific, 1min| API-specific| OHLCV| Prices&Indicators|
|[EODhistoricaldata](https://eodhistoricaldata.com/financial-apis/)| US Securities| Frequency-specific, 1min| API-specific | OHLCV | Prices&Indicators|
|[IEXCloud](https://iexcloud.io/docs/api/)| NMS US securities|1970-now, 1 day|100 per second per IP|OHLCV| Prices&Indicators|
|[JoinQuant](https://www.joinquant.com/)| CN Securities| 2005-now, 1min| 3 requests each time| OHLCV| Prices&Indicators|
|[QuantConnect](https://www.quantconnect.com/docs/v2)| US Securities| 1998-now, 1s| NA| OHLCV| Prices&Indicators|
|[RiceQuant](https://www.ricequant.com/doc/rqdata/python/)| CN Securities| 2005-now, 1ms| Account-specific| OHLCV| Prices&Indicators|
[Sinopac](https://sinotrade.github.io/zh_TW/tutor/prepare/terms/) | Taiwan securities | 2023-04-13~now, 1min | Account-specific | OHLCV | Prices&Indicators|
|[Tushare](https://tushare.pro/document/1?doc_id=131)| CN Securities, A share| -now, 1 min| Account-specific| OHLCV| Prices&Indicators|
|[WRDS](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/nyse-trade-and-quote-taq/)| US Securities| 2003-now, 1ms| 5 requests each time| Intraday Trades|Prices&Indicators|
|[YahooFinance](https://pypi.org/project/yfinance/)| US Securities| Frequency-specific, 1min| 2,000/hour| OHLCV | Prices&Indicators|


<!-- |Data Source |Type |Max Frequency |Raw Data|Preprocessed Data|
|  ----  |  ----  |  ----  |  ----  |  ----  |
|    AkShare |  CN Securities | 1 day  |  OHLCV |  Prices, indicators |
|    Alpaca |  US Stocks, ETFs |  1 min |  OHLCV |  Prices, indicators |
|    Alpha Vantage | Stock, ETF, forex, crypto, technical indicators | 1 min |  OHLCV  & Prices, indicators |
|    Baostock |  CN Securities |  5 min |  OHLCV |  Prices, indicators |
|    Binance |  Cryptocurrency |  1 s |  OHLCV |  Prices, indicators |
|    CCXT |  Cryptocurrency |  1 min  |  OHLCV |  Prices, indicators |
|    currencyapi |  Exchange rate | 1 day |  Exchange rate | Exchange rate, indicators |
|    currencylayer |  Exchange rate | 1 day  |  Exchange rate | Exchange rate, indicators |
|    EOD Historical Data | US stocks, and ETFs |  1 day  |  OHLCV  | Prices, indicators |
|    Exchangerates |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    findatapy |  CN Securities | 1 day  |  OHLCV |  Prices, indicators |
|    Financial Modeling prep | US stocks, currencies, crypto |  1 min |  OHLCV  | Prices, indicators |
|    finnhub | US Stocks, currencies, crypto |   1 day |  OHLCV  | Prices, indicators |
|    Fixer |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    IEXCloud |  NMS US securities | 1 day  | OHLCV |  Prices, indicators |
|    JoinQuant |  CN Securities |  1 min  |  OHLCV |  Prices, indicators |
|    Marketstack | 50+ countries |  1 day  |  OHLCV | Prices, indicators |
|    Open Exchange Rates |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    pandas\_datareader |  US Securities |  1 day |  OHLCV | Prices, indicators |
|    pandas-finance |  US Securities |  1 day  |  OHLCV  & Prices, indicators |
|    Polygon |  US Securities |  1 day  |  OHLCV  | Prices, indicators |
|    Quandl | 250+ sources |  1 day  |  OHLCV  | Prices, indicators |
|    QuantConnect |  US Securities |  1 s |  OHLCV |  Prices, indicators |
|    RiceQuant |  CN Securities |  1 ms  |  OHLCV |  Prices, indicators |
|    Sinopac   | Taiwan securities | 1min | OHLCV |  Prices, indicators |
|    Tiingo | Stocks, crypto |  1 day  |  OHLCV  | Prices, indicators |
|    Tushare |  CN Securities | 1 min  |  OHLCV |  Prices, indicators |
|    WRDS |  US Securities |  1 ms  |  Intraday Trades | Prices, indicators |
|    XE |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    Xignite |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    YahooFinance |  US Securities | 1 min  |  OHLCV  |  Prices, indicators |
|    ystockquote |  US Securities |  1 day  |  OHLCV | Prices, indicators | -->



OHLCV: 开盘价、最高价、最低价和收盘价；成交量。adjusted_close: 调整后收盘价

技术指标：'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma'。用户也可以添加新特征。


## 安装
+ [所有操作系统的安装说明（MAC OS、Ubuntu、Windows 10）](./docs/source/start/installation.rst)
+ [面向量化金融的FinRL：初学者安装和设置教程](https://ai4finance.medium.com/finrl-for-quantitative-finance-install-and-setup-tutorial-for-beginners-1db80ad39159)

## 状态更新
<details><summary><b>版本历史</b> <i>[点击展开]</i></summary>
<div>

* 2022-06-25
	0.3.5: FinRL的正式版本，neo_finrl更名为FinRL-Meta，相关文件在目录：*meta*中。
* 2021-08-25
	0.3.1: 具有三层架构的pytorch版本，apps（金融任务）、drl_agents（深度强化学习算法）、neo_finrl（gym环境）
* 2020-12-14
  	升级到 **Pytorch** 并使用stable-baselines3；暂时移除tensorflow 1.0，正在开发支持tensorflow 2.0
* 2020-11-27
  	0.1: 使用tensorflow 1.5的测试版本
</div>
</details>


## 教程

+ [Towardsdatascience] [自动化股票交易的深度强化学习](https://towardsdatascience.com/deep-reinforcement-learning-for-automated-stock-trading-f1dad0126a02)


## 出版物

|标题 |会议/期刊 |链接|引用|年份|
|  ----  |  ----  |  ----  |  ----  |  ----  |
|金融强化学习的动态数据集和市场环境| Machine Learning - Springer Nature| [论文](https://arxiv.org/abs/2304.13174) [代码](https://github.com/AI4Finance-Foundation/FinRL-Meta) | 7 | 2024 |
|**FinRL-Meta**: 面向数据驱动的金融强化学习的市场环境和基准| NeurIPS 2022| [论文](https://arxiv.org/abs/2211.03107) [代码](https://github.com/AI4Finance-Foundation/FinRL-Meta) | 37 | 2022 |
|**FinRL**: 量化金融中自动化交易的深度强化学习框架| ACM国际金融AI会议 (ICAIF) | [论文](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3955949) | 49 | 2021 |
|**FinRL**: 量化金融中自动化股票交易的深度强化学习库| NeurIPS 2020 深度强化学习研讨会  | [论文](https://arxiv.org/abs/2011.09607) | 87 | 2020 |
|深度强化学习用于自动化股票交易：集成策略| ACM国际金融AI会议 (ICAIF) | [论文](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996) [代码](https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/tutorials/2-Advance/FinRL_Ensemble_StockTrading_ICAIF_2020/FinRL_Ensemble_StockTrading_ICAIF_2020.ipynb) | 154 | 2020 |
|股票交易的实用深度强化学习方法 | NeurIPS 2018 金融AI服务挑战与机遇研讨会| [论文](https://arxiv.org/abs/1811.07522) [代码](https://github.com/AI4Finance-Foundation/DQN-DDPG_Stock_Trading](https://github.com/AI4Finance-Foundation/FinRL/tree/master/examples))| 164 | 2018 |


## 新闻
+ [央广网] [2021 IDEA大会于福田圆满落幕：群英荟萃论道AI 多项目发布亮点纷呈](http://tech.cnr.cn/techph/20211123/t20211123_525669092.shtml)
+ [央广网] [2021 IDEA大会开启AI思想盛宴 沈向洋理事长发布六大前沿产品](https://baijiahao.baidu.com/s?id=1717101783873523790&wfr=spider&for=pc)
+ [IDEA新闻] [2021 IDEA大会发布产品FinRL-Meta——基于数据驱动的强化学习金融风险模拟系统](https://idea.edu.cn/news/20211213143128.html)
+ [知乎] [FinRL-Meta基于数据驱动的强化学习金融元宇宙](https://zhuanlan.zhihu.com/p/437804814)
+ [量化投资与机器学习] [基于深度强化学习的股票交易策略框架（代码+文档)](https://www.mdeditor.tw/pl/p5Gg)
+ [运筹OR帷幄] [领读计划NO.10 | 基于深度增强学习的量化交易机器人：从AlphaGo到FinRL的演变过程](https://zhuanlan.zhihu.com/p/353557417)
+ [深度强化实验室] [【重磅推荐】哥大开源“FinRL”: 一个用于量化金融自动交易的深度强化学习库](https://blog.csdn.net/deeprl/article/details/114828024)
+ [商业新知] [金融科技讲座回顾|AI4Finance: 从AlphaGo到FinRL](https://www.shangyexinzhi.com/article/4170766.html)
+ [Kaggle] [Jane Street Market Prediction](https://www.kaggle.com/c/jane-street-market-prediction/discussion/199313)
+ [矩池云Matpool] [在矩池云上如何运行FinRL股票交易策略框架](http://www.python88.com/topic/111918)
+ [财智无界] [金融学会常务理事陈学彬: 深度强化学习在金融资产管理中的应用](https://www.sohu.com/a/486837028_120929319)
+ [Neurohive] [FinRL: глубокое обучение с подкреплением для трейдинга](https://neurohive.io/ru/gotovye-prilozhenija/finrl-glubokoe-obuchenie-s-podkrepleniem-dlya-trejdinga/)
+ [ICHI.PRO] [양적 금융을위한 FinRL: 단일 주식 거래를위한 튜토리얼](https://ichi.pro/ko/yangjeog-geum-yung-eul-wihan-finrl-dan-il-jusig-geolaeleul-wihan-tyutolieol-61395882412716)
+ [知乎] [基于深度强化学习的金融交易策略（FinRL+Stable baselines3，以道琼斯30股票为例）](https://zhuanlan.zhihu.com/p/563238735)
+ [知乎] [动态数据驱动的金融强化学习](https://zhuanlan.zhihu.com/p/616799055)
+ [知乎] [FinRL的W&B化+超参数搜索和模型优化(基于Stable Baselines 3）](https://zhuanlan.zhihu.com/p/498115373)
+ [知乎] [FinRL-Meta: 未来金融强化学习的元宇宙](https://zhuanlan.zhihu.com/p/544621882)
+
## 引用FinRL

```
@article{dynamic_datasets,
    author = {Liu, Xiao-Yang and Xia, Ziyi and Yang, Hongyang and Gao, Jiechao and Zha, Daochen and Zhu, Ming and Wang, Christina Dan and Wang, Zhaoran and Guo, Jian},
    title = {Dynamic Datasets and Market Environments for Financial Reinforcement Learning},
    journal = {Machine Learning - Springer Nature},
    year = {2024}
}
```


```
@article{liu2022finrl_meta,
  title={FinRL-Meta: Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning},
  author={Liu, Xiao-Yang and Xia, Ziyi and Rui, Jingyang and Gao, Jiechao and Yang, Hongyang and Zhu, Ming and Wang, Christina Dan and Wang, Zhaoran and Guo, Jian},
  journal={NeurIPS},
  year={2022}
}
```

```
@article{liu2021finrl,
    author  = {Liu, Xiao-Yang and Yang, Hongyang and Gao, Jiechao and Wang, Christina Dan},
    title   = {{FinRL}: Deep reinforcement learning framework to automate trading in quantitative finance},
    journal = {ACM International Conference on AI in Finance (ICAIF)},
    year    = {2021}
}

```

```
@article{finrl2020,
    author  = {Liu, Xiao-Yang and Yang, Hongyang and Chen, Qian and Zhang, Runjia and Yang, Liuqing and Xiao, Bowen and Wang, Christina Dan},
    title   = {{FinRL}: A deep reinforcement learning library for automated stock trading in quantitative finance},
    journal = {Deep RL Workshop, NeurIPS 2020},
    year    = {2020}
}
```

```
@article{liu2018practical,
  title={Practical deep reinforcement learning approach for stock trading},
  author={Liu, Xiao-Yang and Xiong, Zhuoran and Zhong, Shan and Yang, Hongyang and Walid, Anwar},
  journal={NeurIPS Workshop on Deep Reinforcement Learning},
  year={2018}
}
```

我们发表了[FinRL论文](http://tensorlet.org/projects/ai-in-finance/)，这些论文列在[谷歌学术](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=XsdPXocAAAAJ)上。之前的论文在[列表](https://github.com/AI4Finance-Foundation/FinRL/blob/master/tutorials/FinRL_papers.md)中给出。


## 加入和贡献

欢迎来到 **AI4Finance** 社区！

请查看[贡献指南](https://github.com/AI4Finance-Foundation/FinRL-Tutorials/blob/master/Contributing.md)。

### 贡献者

谢谢！

<a href="https://github.com/AI4Finance-LLC/FinRL-Library/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AI4Finance-LLC/FinRL-Library" />
</a>


## 许可证

MIT License
```
Trademark Disclaimer

FinRL® is a registered trademark.
This license does not grant permission to use the FinRL name, logo, or related trademarks
without prior written consent, except as permitted by applicable trademark law.
For trademark inquiries or permissions, please contact: contact@finrl.ai

```

**免责声明：我们出于学术目的在MIT教育许可证下分享代码。本文中的任何内容都不构成财务建议，也不是交易真钱的推荐。请使用常识，并在交易或投资前始终首先咨询专业人士。**
