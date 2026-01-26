:github_url: https://github.com/AI4Finance-Foundation/FinRL

=================
第1节. 数据
=================

第1部分. 安装包
==================================
..  code-block:: python
    ## 安装所需包
    !pip install swig
    !pip install wrds
    !pip install pyportfolioopt
    ## 安装finrl库
    !pip install git+https://github.com/AI4Finance-Foundation/FinRL.git

..  code-block:: python
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS

import itertools

第2部分. 获取数据
==================================

`yfinance <https://github.com/ranaroussi/yfinance>`_ 是一个开源库,提供从Yahoo Finance获取历史数据的API。在FinRL中,我们有一个名为YahooDownloader的类,它使用yfinance从Yahoo Finance获取数据。

**OHLCV**:下载的数据形式为OHLCV,分别对应**开盘价、最高价、最低价、收盘价、成交量**。OHLCV很重要,因为它们包含股票时间序列中的大部分数值信息。从OHLCV,交易者可以进一步判断和预测,如动量、人们的兴趣、市场趋势等。

单个股票代码的数据
----------------------------------------

**使用yfinance**
..  code-block:: python
    aapl_df_yf = yf.download(tickers = "aapl", start='2020-01-01', end='2020-01-31')

**使用FinRL**

在FinRL的YahooDownloader中,我们将数据框修改为便于进一步数据处理的格式。我们使用调整后的收盘价而不是收盘价,并添加一个表示一周中某天的列(0-4对应星期一至星期五)。

..  code-block:: python
    aapl_df_finrl = YahooDownloader(start_date = '2020-01-01',
                                    end_date = '2020-01-31',
                                    ticker_list = ['aapl']).fetch_data()

选定股票代码的数据
----------------------------------------
..  code-block:: python
    TRAIN_START_DATE = '2009-01-01'
    TRAIN_END_DATE = '2020-07-01'
    TRADE_START_DATE = '2020-07-01'
    TRADE_END_DATE = '2021-10-29'
..  code-block:: python
    df_raw = YahooDownloader(start_date = TRAIN_START_DATE,
                             end_date = TRADE_END_DATE,
                             ticker_list = config_tickers.DOW_30_TICKER).fetch_data()

第3部分. 预处理数据
==================================

我们需要检查缺失的数据并进行特征工程,将数据点转换为状态。

- **添加技术指标**。在实际交易中,需要考虑各种信息,如历史价格、当前持有股份、技术指标等。在这里,我们演示两个趋势跟踪技术指标:MACD和RSI。
- **添加动荡指数**。风险厌恶反映了投资者是否更倾向于保护资本。它还影响一个人在面对不同市场波动水平时的交易策略。为了在最坏情况下控制风险,例如2007-2008年的金融危机,FinRL采用了衡量资产价格极端波动的动荡指数。

让我们以MACD为例。移动平均收敛/发散(MACD)是最常用的指标之一,显示牛市和熊市。其计算基于EMA(指数移动平均指标,衡量一段时间内的趋势方向)。
