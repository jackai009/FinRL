:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

多股票交易
===============================

从零开始的股票交易深度强化学习:多股票交易


.. tip::

    在`Google Colab`_上逐步运行代码。

    .. _Google Colab: https://colab.research.google.com/github/AI4Finance-Foundation/FinRL/blob/master/FinRL_StockTrading_NeurIPS_2018.ipynb




步骤1:准备工作
---------------------------------------

**步骤1.1:概述**


首先,我想解释一下使用深度强化学习进行多股票交易的逻辑。

我们在本文中以道琼斯30只成分股为例,因为它们是最受欢迎的股票。

很多人对"深度强化学习"这个词感到恐惧,实际上,你可以把它看作是一个"智能AI"或"智能股票交易员"或"R2-D2交易员",如果你愿意的话,可以直接使用它。

假设我们有一个训练良好的DRL代理"DRL交易员",我们想用它来交易投资组合中的多只股票。

    - 假设我们在时间t,在时间t的交易日结束时,我们将知道道琼斯30只成分股的开盘价-最高价-最低价-收盘价。我们可以使用这些信息来计算技术指标,如MACD、RSI、CCI、ADX。在强化学习中,我们称这些数据或特征为"状态"。

    - 我们知道我们的投资组合价值V(t) = 余额(t) + 股票的美元金额(t)。

    - 我们将状态输入到我们训练良好的DRL交易员中,交易员将输出一个动作列表,每只股票的动作是一个在[-1, 1]范围内的值,我们可以将此值视为交易信号,1表示强烈的买入信号,-1表示强烈的卖出信号。

    - 我们计算k = actions * h_max,h_max是一个预定义参数,设置为交易的最大股份数。所以我们将得到一个要交易的股份列表。

    - 股份的美元金额 = 要交易的股份 * 收盘价(t)。

    - 更新余额和股份。这些股份的美元金额是我们在时间t需要交易的资金。更新后的余额 = 余额(t) - 我们购买股份支付的资金金额 + 我们卖出股份收到的资金金额。更新后的股份 = 持有股份(t) - 要卖出的股份 + 要买入的股份。

    - 所以我们在时间t的交易日结束时(时间t的收盘价等于时间t+1的开盘价)根据DRL交易员的建议采取行动进行交易。我们希望我们将在时间t+1的交易日结束时从这些行动中受益。

    - 进入时间t+1,在交易日结束时,我们将知道t+1的收盘价,股票的美元金额(t+1) = sum(更新后的股份 * 收盘价(t+1))。投资组合价值V(t+1) = 余额(t+1) + 股票的美元金额(t+1)。

    - 所以从DRL交易员在时间t到t+1采取动作的步骤奖励是r = v(t+1) - v(t)。在训练阶段,奖励可以是正的也可以是负的。但当然,我们在交易中需要正的奖励来说明我们的DRL交易员是有效的。

    - 重复此过程直到终止。

以下是多股票交易的逻辑图表和一个用于演示目的的虚构示例:
.. image:: ../../image/multiple_1.jpeg
    :scale: 60%
.. image:: ../../image/multiple_2.png

多股票交易与单股票交易不同,因为随着股票数量的增加,数据的维度将增加,强化学习中的状态和动作空间将呈指数级增长。因此,稳定性和可重复性在这里非常关键。

我们介绍一个DRL库FinRL,它促进初学者接触量化金融并开发自己的股票交易策略。

FinRL的特点是其可重复性、可扩展性、简单性、适用性和可扩展性。

本文重点关注我们论文中的用例之一:多股票交易。我们使用一个Jupyter notebook来包含所有必要的步骤。
.. image:: ../../image/FinRL-Architecture.png

**步骤1.2:问题定义**:

这个问题是设计股票交易的自动化解决方案。我们将股票交易过程建模为马尔可夫决策过程(MDP)。然后我们将我们的交易目标制定为最大化问题。
该算法使用深度强化学习(DRL)算法进行训练,强化学习环境的组件包括:

- 动作:动作空间描述了代理与环境交互的允许动作。通常,a ∈ A包括三个动作:a ∈ {-1, 0, 1},其中-1、0、1分别代表卖出、持有和买入一只股票。此外,一个动作可以在多股上执行。我们使用动作空间{-k, ..., -1, 0, 1, ..., k},其中k表示股份数。例如,"买入10股AAPL"或"卖出10股AAPL"分别是10或-10

- 奖励函数:r(s, a, s′)是代理学习更好动作的激励机制。当在状态s下采取动作a并到达新状态s'时投资组合价值的变化,即r(s, a, s′) = v′ - v,其中v′和v分别代表状态s′和s时的投资组合价值

- 状态:状态空间描述了代理从环境接收的观察。正如人类交易员在执行交易之前需要分析各种信息一样,我们的交易代理观察许多不同的特征以便在交互环境中更好地学习。

- 环境:道琼斯30只成分股

本案例研究中股票的数据是从Yahoo Finance API获取的。数据包含开盘价-最高价-最低价-收盘价和成交量。

**步骤1.3:FinRL安装**:

.. code-block::
    :linenos:

    ## 安装finrl库
    !pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git

然后我们导入本演示所需的包。

**步骤1.4:导入包**:

.. code-block:: python
    :linenos:

    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    # matplotlib.use('Agg')
    import datetime

    %matplotlib inline
    from finrl import config
    from finrl import config_tickers
    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    from finrl.agents.stablebaselines3.models import DRLAgent

    from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
    from pprint import pprint

    import sys
    sys.path.append("../FinRL-Library")

    import itertools

最后,创建用于存储的文件夹。

**步骤1.5:创建文件夹**:
.. code-block:: python
    :linenos:

    import os
    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)

然后所有准备工作都完成了。我们可以开始了!

步骤2:下载数据
---------------------------------------
在训练我们的DRL代理之前,我们首先需要获取道琼斯30只股票的历史数据。这里我们使用来自Yahoo! Finance的数据。
Yahoo! Finance是一个提供股票数据、金融新闻、财务报告等的网站。Yahoo Finance提供的所有数据都是免费的。yfinance是一个开源库,提供从Yahoo! Finance下载数据的API。我们将使用这个包在这里下载数据。

FinRL使用YahooDownloader_类来提取数据。

.. _YahooDownloader: https://github.com/AI4Finance-LLC/FinRL-Library/blob/master/finrl/marketdata/yahoodownloader.py

.. code-block:: python

    class YahooDownloader:
        """
            提供从Yahoo Finance API检索每日股票数据的方法

            属性
            ----------
                start_date : str
                    数据的开始日期(从config.py修改)
                end_date : str
                    数据的结束日期(从config.py修改)
                ticker_list : list
                    股票代码列表(从config.py修改)

            方法
            -------
                fetch_data()
                    从yahoo API获取数据
        """

下载并将数据保存在pandas DataFrame中:

.. code-block:: python
   :linenos:

    # 下载并将数据保存在pandas DataFrame中:
    df = YahooDownloader(start_date = '2009-01-01',
                              end_date = '2020-09-30',
                              ticker_list = config_tickers.DOW_30_TICKER).fetch_data()

    print(df.sort_values(['date','tic'],ignore_index=True).head(30))


.. image:: ../../image/multiple_3.png

步骤3:预处理数据
---------------------------------------

数据预处理是训练高质量机器学习模型的关键步骤。我们需要检查缺失数据并进行特征工程,以便将数据转换为模型就绪的状态。

**步骤3.1:检查缺失数据**

.. code-block:: python
    :linenos:

    # 检查缺失数据
    dow_30.isnull().values.any()


**步骤3.2:添加技术指标**

在实际交易中,需要考虑各种信息,例如历史股票价格、当前持有股份、技术指标等。在本文中,我们演示两个趋势跟踪技术指标:MACD和RSI。

.. code-block:: python
    :linenos:

    def add_technical_indicator(df):
            """
            计算技术指标
            使用stockstats包添加技术指标
            :param data: (df) pandas dataframe
            :return: (df) pandas dataframe
            """
            stock = Sdf.retype(df.copy())
            stock['close'] = stock['adjcp']
            unique_ticker = stock.tic.unique()

            macd = pd.DataFrame()
            rsi = pd.DataFrame()

            #temp = stock[stock.tic == unique_ticker[0]]['macd']
            for i in range(len(unique_ticker)):
                ## macd
                temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
                temp_macd = pd.DataFrame(temp_macd)
                macd = macd.append(temp_macd, ignore_index=True)
                ## rsi
                temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
                temp_rsi = pd.DataFrame(temp_rsi)
                rsi = rsi.append(temp_rsi, ignore_index=True)

            df['macd'] = macd
            df['rsi'] = rsi
            return df


**步骤3.3:添加动荡指数**

风险厌恶反映了投资者是否将选择保护资本。它还影响一个人在面对不同市场波动水平时的交易策略。

为了在最坏情况下控制风险,例如2007-2008年的金融危机,FinRL采用了衡量极端资产价格波动的金融动荡指数。

.. code-block:: python
    :linenos:

    def add_turbulence(df):
        """
            从预先计算的数据框添加动荡指数
            :param data: (df) pandas dataframe
            :return: (df) pandas dataframe
            """
        turbulence_index = calcualte_turbulence(df)
        df = df.merge(turbulence_index, on='datadate')
        df = df.sort_values(['datadate','tic']).reset_index(drop=True)
        return df


    def calcualte_turbulence(df):
        """基于道琼斯30只计算动荡指数"""
        # 可以添加其他市场资产

        df_price_pivot=df.pivot(index='datadate', columns='tic', values='adjcp')
        unique_date = df.datadate.unique()
        # 一年后开始
        start = 252
        turbulence_index = [0]*start
        #turbulence_index = [0]
        count=0
        for i in range(start,len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
            cov_temp = hist_price.cov()
            current_temp=(current_price - np.mean(hist_price,axis=0))
            temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
            if temp>0:
                count+=1
                if count>2:
                    turbulence_temp = temp[0][0]
                else:
                    #避免因为计算刚刚开始而产生的大异常值
                    turbulence_temp=0
            else:
                turbulence_temp=0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame({'datadate':df_price_pivot.index,
                                         'turbulence':turbulence_index})
        return turbulence_index
**步骤3.4 特征工程**

FinRL使用FeatureEngineer_类来预处理数据。

.. _FeatureEngineer: https://github.com/AI4Finance-LLC/FinRL-Library/blob/master/finrl/preprocessing/preprocessors.py

.. code-block: python

    class FeatureEngineer:
        """
            提供预处理股票价格数据的方法

            属性
            ----------
                df: DataFrame
                    从Yahoo API下载的数据
                feature_number : int
                    我们使用的特征数量
                use_technical_indicator : boolean
                    使用技术指标或不使用
                use_turbulence : boolean
                    使用动荡指数或不使用
            方法
            -------
                preprocess_data()
                    进行特征工程的主要方法
        """

执行特征工程:

.. code-block:: python
   :linenos:

    # 执行特征工程:
    df = FeatureEngineer(df.copy(),
                         use_technical_indicator=True,
                         tech_indicator_list = config.INDICATORS,
                         use_turbulence=True,
                         user_defined_feature = False).preprocess_data()


.. image:: ../../image/multiple_4.png

步骤4:设计环境
---------------------------------------

考虑到自动化股票交易任务的随机性和交互性质,金融任务被建模为马尔可夫决策过程(MDP)问题。训练过程涉及观察股票价格变化、采取行动和奖励计算,以使代理相应地调整其策略。通过与环境的交互,交易代理将随着时间推移得出一个具有最大化奖励的交易策略。

我们的交易环境基于OpenAI Gym框架,根据时间驱动模拟的原则,使用真实市场数据模拟实时股票市场。

动作空间描述了代理与环境交互的允许动作。通常,动作a包括三个动作:{-1, 0, 1},其中-1、0、1分别代表卖出、持有和买入一股。此外,一个动作可以在多股上执行。我们使用动作空间{-k,…,-1, 0, 1, …, k},其中k表示要买入的股份数,-k表示要卖出的股份数。例如,"买入10股AAPL"或"卖出10股AAPL"分别是10或-10。连续动作空间需要归一化为[-1, 1],因为策略定义在高斯分布上,这需要归一化和对称。

**步骤4.1:训练环境**

.. code-block:: python
    :linenos:

    ## 训练环境
    import numpy as np
    import pandas as pd
    from gym.utils import seeding
    import gym
    from gym import spaces
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 股份归一化因子
    # 每次交易100股
    HMAX_NORMALIZE = 100
    # 我们账户中的初始资金金额
    INITIAL_ACCOUNT_BALANCE=1000000
    # 我们投资组合中的股票总数
    STOCK_DIM = 30
    # 交易费用:1/1000合理的百分比
    TRANSACTION_FEE_PERCENT = 0.001

    REWARD_SCALING = 1e-4

    class StockEnvTrain(gym.Env):
        """用于OpenAI gym的股票交易环境"""
        metadata = {'render.modes': ['human']}

        def __init__(self, df,day = 0):
            #super(StockEnv, self).__init__()
            self.day = day
            self.df = df

            # action_space归一化和形状是STOCK_DIM
            self.action_space = spaces.Box(low = -1, high = 1,shape = (STOCK_DIM,))
            # 形状 = 181:[当前余额]+[价格1-30]+[持有股份1-30]
            # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
            self.observation_space = spaces.Box(low=0, high=np.inf, shape = (121,))
            # 从pandas dataframe加载数据
            self.data = self.df.loc[self.day,:]
            self.terminal = False
            # 初始化状态
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                          self.data.adjcp.values.tolist() + \
                          [0]*STOCK_DIM + \
                          self.data.macd.values.tolist() + \
                          self.data.rsi.values.tolist()
                          #self.data.cci.values.tolist() + \
                          #self.data.adx.values.tolist()
            # 初始化奖励
            self.reward = 0
            self.cost = 0
            # 记住所有总余额变化
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.rewards_memory = []
            self.trades = 0
            self._seed()

        def _sell_stock(self, index, action):
            # 根据动作的符号执行卖出操作
            if self.state[index+STOCK_DIM+1] > 0:
                #更新余额
                self.state[0] += \
                self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \
                 (1- TRANSACTION_FEE_PERCENT)

                self.state[index+STOCK_DIM+1] -= min(abs(action), self.state[index+STOCK_DIM+1])
                self.cost +=self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \
                 TRANSACTION_FEE_PERCENT
                self.trades+=1
            else:
                pass

        def _buy_stock(self, index, action):
            # 根据动作的符号执行买入操作
            available_amount = self.state[0] // self.state[index+1]
            # print('available_amount:{}'.format(available_amount))

            #更新余额
            self.state[0] -= self.state[index+1]*min(available_amount, action)* \
                              (1+ TRANSACTION_FEE_PERCENT)

            self.state[index+STOCK_DIM+1] += min(available_amount, action)

            self.cost+=self.state[index+1]*min(available_amount, action)* \
                              TRANSACTION_FEE_PERCENT
            self.trades+=1

        def step(self, actions):
            # print(self.day)
            self.terminal = self.day >= len(self.df.index.unique())-1
            # print(actions)

            if self.terminal:
                plt.plot(self.asset_memory,'r')
                plt.savefig('account_value_train.png')
                plt.close()
                end_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
                print("previous_total_asset:{}".format(self.asset_memory[0]))

                print("end_total_asset:{}".format(end_total_asset))
                df_total_value = pd.DataFrame(self.asset_memory)
                df_total_value.to_csv('account_value_train.csv')
                print("total_reward:{}".format(self.state[0]+sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):61]))- INITIAL_ACCOUNT_BALANCE ))
                print("total_cost: ", self.cost)
                print("total_trades: ", self.trades)
                df_total_value.columns = ['account_value']
                df_total_value['daily_return']=df_total_value.pct_change(1)
                sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ \
                      df_total_value['daily_return'].std()
                print("Sharpe: ",sharpe)
                print("=================================")
                df_rewards = pd.DataFrame(self.rewards_memory)
                df_rewards.to_csv('account_rewards_train.csv')

                return self.state, self.reward, self.terminal,{}
            else:
                actions = actions * HMAX_NORMALIZE

                begin_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):61]))
                #print("begin_total_asset:{}".format(begin_total_asset))

                argsort_actions = np.argsort(actions)

                sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
                buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

                for index in sell_index:
                    # print('take sell action'.format(actions[index]))
                    self._sell_stock(index, actions[index])

                for index in buy_index:
                    # print('take buy action: {}'.format(actions[index]))
                    self._buy_stock(index, actions[index])

                self.day += 1
                self.data = self.df.loc[self.day,:]
                #加载下一个状态
                # print("stock_shares:{}".format(self.state[29:]))
                self.state =  [self.state[0]] + \
                        self.data.adjcp.values.tolist() + \
                        list(self.state[(STOCK_DIM+1):61]) + \
                        self.data.macd.values.tolist() + \
                        self.data.rsi.values.tolist()

                end_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):61]))

                #print("end_total_asset:{}".format(end_total_asset))

                self.reward = end_total_asset - begin_total_asset
                self.rewards_memory.append(self.reward)

                self.reward = self.reward * REWARD_SCALING
                # print("step_reward:{}".format(self.reward))

                self.asset_memory.append(end_total_asset)


            return self.state, self.reward, self.terminal, {}

        def reset(self):
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.day = 0
            self.data = self.df.loc[self.day,:]
            self.cost = 0
            self.trades = 0
            self.terminal = False
            self.rewards_memory = []
            #初始化状态
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                          self.data.adjcp.values.tolist() + \
                          [0]*STOCK_DIM + \
                          self.data.macd.values.tolist() + \
                          self.data.rsi.values.tolist()
            return self.state

        def render(self, mode='human'):
            return self.state

        def _seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]

**步骤4.2:交易环境**

.. code-block:: python
    :linenos:

    ## 交易环境
    import numpy as np
    import pandas as pd
    from gym.utils import seeding
    import gym
    from gym import spaces
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 股份归一化因子
    # 每次交易100股
    HMAX_NORMALIZE = 100
    # 我们账户中的初始资金金额
    INITIAL_ACCOUNT_BALANCE=1000000
    # 我们投资组合中的股票总数
    STOCK_DIM = 30
    # 交易费用:1/1000合理的百分比
    TRANSACTION_FEE_PERCENT = 0.001

    # 动荡指数:90-150合理的阈值
    #TURBULENCE_THRESHOLD = 140
    REWARD_SCALING = 1e-4

    class StockEnvTrade(gym.Env):
        """用于OpenAI gym的股票交易环境"""
        metadata = {'render.modes': ['human']}

        def __init__(self, df,day = 0,turbulence_threshold=140):
            #super(StockEnv, self).__init__()
            #money = 10 , scope = 1
            self.day = day
            self.df = df
            # action_space归一化和形状是STOCK_DIM
            self.action_space = spaces.Box(low = -1, high = 1,shape = (STOCK_DIM,))
            # 形状 = 181:[当前余额]+[价格1-30]+[持有股份1-30]
            # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
            self.observation_space = spaces.Box(low=0, high=np.inf, shape = (121,))
            # 从pandas dataframe加载数据
            self.data = self.df.loc[self.day,:]
            self.terminal = False
            self.turbulence_threshold = turbulence_threshold
            # 初始化状态
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                          self.data.adjcp.values.tolist() + \
                          [0]*STOCK_DIM + \
                          self.data.macd.values.tolist() + \
                          self.data.rsi.values.tolist()

            # 初始化奖励
            self.reward = 0
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            # 记住所有总余额变化
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.rewards_memory = []
            self.actions_memory=[]
            self.date_memory=[]
            self._seed()


        def _sell_stock(self, index, action):
            # 根据动作的符号执行卖出操作
            if self.turbulence<self.turbulence_threshold:
                if self.state[index+STOCK_DIM+1] > 0:
                    #更新余额
                    self.state[0] += \
                    self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \
                     (1- TRANSACTION_FEE_PERCENT)

                    self.state[index+STOCK_DIM+1] -= min(abs(action), self.state[index+STOCK_DIM+1])
                    self.cost +=self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \
                     TRANSACTION_FEE_PERCENT
                    self.trades+=1
                else:
                    pass
            else:
                # 如果动荡超过阈值,只需清空所有头寸
                if self.state[index+STOCK_DIM+1] > 0:
                    #更新余额
                    self.state[0] += self.state[index+1]*self.state[index+STOCK_DIM+1]* \
                                  (1- TRANSACTION_FEE_PERCENT)
                    self.state[index+STOCK_DIM+1] =0
                    self.cost += self.state[index+1]*self.state[index+STOCK_DIM+1]* \
                                  TRANSACTION_FEE_PERCENT
                    self.trades+=1
                else:
                    pass

        def _buy_stock(self, index, action):
            # 根据动作的符号执行买入操作
            if self.turbulence< self.turbulence_threshold:
                available_amount = self.state[0] // self.state[index+1]
                # print('available_amount:{}'.format(available_amount))

                #更新余额
                self.state[0] -= self.state[index+1]*min(available_amount, action)* \
                                  (1+ TRANSACTION_FEE_PERCENT)

                self.state[index+STOCK_DIM+1] += min(available_amount, action)

                self.cost+=self.state[index+1]*min(available_amount, action)* \
                                  TRANSACTION_FEE_PERCENT
                self.trades+=1
            else:
                # 如果动荡超过阈值,只需停止买入
                pass

        def step(self, actions):
            # print(self.day)
            self.terminal = self.day >= len(self.df.index.unique())-1
            # print(actions)

            if self.terminal:
                plt.plot(self.asset_memory,'r')
                plt.savefig('account_value_trade.png')
                plt.close()

                df_date = pd.DataFrame(self.date_memory)
                df_date.columns = ['datadate']
                df_date.to_csv('df_date.csv')

                df_actions = pd.DataFrame(self.actions_memory)
                df_actions.columns = self.data.tic.values
                df_actions.index = df_date.datadate
                df_actions.to_csv('df_actions.csv')

                df_total_value = pd.DataFrame(self.asset_memory)
                df_total_value.to_csv('account_value_trade.csv')
                end_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
                print("previous_total_asset:{}".format(self.asset_memory[0]))

                print("end_total_asset:{}".format(end_total_asset))
                print("total_reward:{}".format(self.state[0]+sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):61]))- self.asset_memory[0] ))
                print("total_cost: ", self.cost)
                print("total trades: ", self.trades)

                df_total_value.columns = ['account_value']
                df_total_value['daily_return']=df_total_value.pct_change(1)
                sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ \
                      df_total_value['daily_return'].std()
                print("Sharpe: ",sharpe)

                df_rewards = pd.DataFrame(self.rewards_memory)
                df_rewards.to_csv('account_rewards_trade.csv')

                # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
                #with open('obs.pkl', 'wb') as f:
                #    pickle.dump(self.state, f)

                return self.state, self.reward, self.terminal,{}
            else:
                # print(np.array(self.state[1:29]))
                self.date_memory.append(self.data.datadate.unique())

                #print(self.data)
                actions = actions * HMAX_NORMALIZE
                if self.turbulence>=self.turbulence_threshold:
                    actions=np.array([-HMAX_NORMALIZE]*STOCK_DIM)
                self.actions_memory.append(actions)

                #actions = (actions.astype(int))

                begin_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
                #print("begin_total_asset:{}".format(begin_total_asset))

                argsort_actions = np.argsort(actions)
                #print(argsort_actions)

                sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
                buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

                for index in sell_index:
                    # print('take sell action'.format(actions[index]))
                    self._sell_stock(index, actions[index])

                for index in buy_index:
                    # print('take buy action: {}'.format(actions[index]))
                    self._buy_stock(index, actions[index])

                self.day += 1
                self.data = self.df.loc[self.day,:]
                self.turbulence = self.data['turbulence'].values[0]
                #print(self.turbulence)
                #加载下一个状态
                # print("stock_shares:{}".format(self.state[29:]))
                self.state =  [self.state[0]] + \
                        self.data.adjcp.values.tolist() + \
                        list(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]) + \
                        self.data.macd.values.tolist() + \
                        self.data.rsi.values.tolist()

                end_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))

                #print("end_total_asset:{}".format(end_total_asset))

                self.reward = end_total_asset - begin_total_asset
                self.rewards_memory.append(self.reward)

                self.reward = self.reward * REWARD_SCALING

                self.asset_memory.append(end_total_asset)

            return self.state, self.reward, self.terminal, {}
        def reset(self):
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.day = 0
            self.data = self.df.loc[self.day,:]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            #self.iteration=self.iteration
            self.rewards_memory = []
            self.actions_memory=[]
            self.date_memory=[]
            #初始化状态
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                          self.data.adjcp.values.tolist() + \
                          [0]*STOCK_DIM + \
                          self.data.macd.values.tolist() + \
                          self.data.rsi.values.tolist()

            return self.state
        def render(self, mode='human',close=False):
            return self.state

        def _seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]

步骤5:实现DRL算法
-------------------------------------

DRL算法的实现基于OpenAI Baselines和Stable Baselines。Stable Baselines是OpenAI Baselines的一个分支,进行了主要的结构重构和代码清理。

**步骤5.1:训练数据拆分**:2009-01-01至2018-12-31

.. code-block:: python
    :linenos:

    def data_split(df,start,end):
        """
            使用日期将数据集拆分为训练或测试
            :param data: (df) pandas dataframe, start, end
            :return: (df) pandas dataframe
            """
        data = df[(df.datadate >= start) & (df.datadate < end)]
        data=data.sort_values(['datadate','tic'],ignore_index=True)
        data.index = data.datadate.factorize()[0]
        return data

**步骤5.2:模型训练**:DDPG

.. code-block:: python
    :linenos:

    ## tensorboard --logdir ./multiple_stock_tensorboard/
    # 在DDPG中向动作添加噪声有助于学习以便更好地探索
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    # 模型设置
    model_ddpg = DDPG('MlpPolicy',
                       env_train,
                       batch_size=64,
                       buffer_size=100000,
                       param_noise=param_noise,
                       action_noise=action_noise,
                       verbose=0,
                       tensorboard_log="./multiple_stock_tensorboard/")

    ## 250k timesteps:大约需要20分钟完成
    model_ddpg.learn(total_timesteps=250000, tb_log_name="DDPG_run_1")

**步骤5.3:交易**

假设我们在2019-01-01拥有1,000,000美元的初始资本。我们使用DDPG模型来交易道琼斯30只股票。

**步骤5.4:设置动荡阈值**

将动荡阈值设置为样本内动荡数据的99%分位数,如果当前动荡指数大于阈值,那么我们假设当前市场波动

.. code-block:: python
    :linenos:

    insample_turbulence = dow_30[(dow_30.datadate<'2019-01-01') & (dow_30.datadate>='2009-01-01')]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])

**步骤5.5:准备测试数据和环境**

.. code-block:: python
    :linenos:

    # 测试数据
    test = data_split(dow_30, start='2019-01-01', end='2020-10-30')
    # 测试环境
    env_test = DummyVecEnv([lambda: StockEnvTrade(test, turbulence_threshold=insample_turbulence_threshold)])
    obs_test = env_test.reset()

**步骤5.6:预测**

.. code-block:: python
    :linenos:

    def DRL_prediction(model, data, env, obs):
        print("==============Model Prediction===========")
        for i in range(len(data.index.unique())):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()

步骤6:回测我们的策略
---------------------------------

出于简化目的,在本文中,我们只是手动计算夏普比率和年化回报。

.. code-block:: python
    :linenos:

    def backtest_strat(df):
        strategy_ret= df.copy()
        strategy_ret['Date'] = pd.to_datetime(strategy_ret['Date'])
        strategy_ret.set_index('Date', drop = False, inplace = True)
        strategy_ret.index = strategy_ret.index.tz_localize('UTC')
        del strategy_ret['Date']
        ts = pd.Series(strategy_ret['daily_return'].values, index=strategy_ret.index)
        return ts

**步骤6.1:道琼斯工业平均指数**

.. code-block:: python
    :linenos:

    def get_buy_and_hold_sharpe(test):
        test['daily_return']=test['adjcp'].pct_change(1)
        sharpe = (252**0.5)*test['daily_return'].mean()/ \
        test['daily_return'].std()
        annual_return = ((test['daily_return'].mean()+1)**252-1)*100
        print("annual return: ", annual_return)

        print("sharpe ratio: ", sharpe)
        #return sharpe

**步骤6.2:我们的DRL交易策略**

.. code-block:: python
    :linenos:

    def get_daily_return(df):
        df['daily_return']=df.account_value.pct_change(1)
        #df=df.dropna()
        sharpe = (252**0.5)*df['daily_return'].mean()/ \
        df['daily_return'].std()

        annual_return = ((df['daily_return'].mean()+1)**252-1)*100
        print("annual return: ", annual_return)
        print("sharpe ratio: ", sharpe)
        return df

**步骤6.3:使用Quantopian pyfolio绘制结果**

回测在评估交易策略性能方面起着关键作用。首选自动化回测工具,因为它减少了人为错误。我们通常使用Quantopian pyfolio包来回测我们的交易策略。它易于使用,由各种单独的图表组成,提供交易策略性能的全面图像。

.. code-block:: python
    :linenos:

    %matplotlib inline
    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(returns = DRL_strat,
                                       benchmark_rets=dow_strat, set_context=False)
