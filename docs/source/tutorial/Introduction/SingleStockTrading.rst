:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

单股票交易
===================================

**我们的论文**：
`FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance`_.

.. _FinRL\: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance: https://arxiv.org/abs/2011.09607

在NeurIPS 2020:深度强化学习研讨会上展示。

Jupyter笔记本代码可在我们的Github_和`Google Colab`_上找到。

.. _Github: https://github.com/AI4Finance-LLC/FinRL-Library
.. _Google Colab: https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/FinRL_single_stock_trading.ipynb

.. tip::

    - FinRL `Single Stock Trading <https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/FinRL_single_stock_trading.ipynb>`_ at Google Colab.

    - FinRL `Multiple Stocks Trading <https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/FinRL_multiple_stock_trading.ipynb>`_ at Google Colab:

查看我们之前的教程:`Multiple Stock Trading <https://finrl.readthedocs.io/en/latest/tutorial/MultipleStockTrading.html>`_和`Portfolio Allocation <https://finrl.readthedocs.io/en/latest/tutorial/PortfolioAllocation.html>`_以获取FinRL架构和模块的详细解释。



概述
-------------

首先,我们想解释使用深度强化学习进行单股票交易的逻辑。我们在本文中以Apple(AAPL)股票为例,因为它是最受欢迎的股票之一。

假设我们在2019年初获得了100万美元。我们想将这1,000,000美元投资于股票市场,在这种情况下是Apple(AAPL)股票。假设没有保证金交易,没有卖空,没有国库券(使用所有资金仅交易AAPL股票)。

我们聘请了一位聪明的投资组合经理——深度强化学习先生。DRL先生每天都会给我们建议,包括应该买入或卖出多少AAPL股票。所以我们每天只需要根据DRL先生的建议执行买入或卖出操作。基本逻辑如下。
.. image:: ../../image/single_stock_trading.png

单股票交易与多股票交易不同,因为我们只专注于一只股票,这简化了决策过程。

我们介绍一个DRL库FinRL,它促进初学者接触量化金融。FinRL是一个专门为自动化股票交易设计的DRL库,致力于教育和演示目的。

本文重点关注我们论文中的用例之一:单股票交易。我们使用一个Jupyter notebook来包含所有必要的步骤。



问题定义
--------------------------

这个问题是设计用于股票交易的自动化解决方案。我们将股票交易过程建模为马尔可夫决策过程(MDP)。然后我们将我们的交易目标制定为最大化问题。

强化学习环境的组件包括:

    - **Action**: {卖出k股, 持有, 买入k股},其中k可以是任何整数。连续动作空间需要归一化为[-1, 1],因为策略定义在高斯分布上,这需要归一化和对称。

    - **State: {账户余额, 当前股价, MACD, RSI, CCI, ADX}**。状态空间描述了代理从环境接收的观察。

    - **Reward function**: r(s, a, s′) = V(t+1) - V(t),其中V是投资组合价值,包括现金余额和股票价值。奖励是在状态s下采取动作a并到达新状态s'时的投资组合价值变化。

    - **Environment**:Apple(AAPL)股票的交易环境。

本案例研究中股票的数据是从Yahoo Finance API获取的。数据包含开盘价-最高价-最低价-收盘价和成交量。



Load Python Packages
--------------------------

安装FinRL的不稳定开发版本:

.. code-block:: python
   :linenos:

    # 在Jupyter notebook中安装不稳定开发版本:
    !pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git

导入包:

.. code-block:: python
   :linenos:

    # 导入包
    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    import datetime

    from finrl import config
    from finrl import config_tickers
    from finrl.marketdata.yahoodownloader import YahooDownloader
    from finrl.preprocessing.preprocessors import FeatureEngineer
    from finrl.preprocessing.data import data_split
    from finrl.env.environment import EnvSetup
    from finrl.env.EnvSingleStock_train import StockEnvTrain
    from finrl.env.EnvSingleStock_trade import StockEnvTrade
    from finrl.model.models import DRLAgent
    from finrl.trade.backtest import BackTestStats, BaselineStats, BackTestPlot, backtest_strat, baseline_strat
    from finrl.trade.backtest import backtest_strat, baseline_strat

    import os
    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)



下载数据
--------------------------

FinRL使用YahooDownloader类来提取数据。

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
                         end_date = '2020-12-01',
                         ticker_list = ['AAPL']).fetch_data()


预处理数据
--------------------------

FinRL使用FeatureEngineer类来预处理数据。

.. code-block:: python

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
                        use_turbulence=False).preprocess_data()

构建环境
--------------------------

FinRL使用EnvSetup类来设置环境。


.. code-block:: python

    class EnvSetup:
        """
            提供从Yahoo Finance API检索每日股票数据的方法

            属性
            ----------
                stock_dim: int
                    唯一股票数量
                hmax : int
                    交易的最大股份数
                initial_amount: int
                    初始资金
                transaction_cost_pct : float
                    每笔交易的交易费用百分比
                reward_scaling: float
                    奖励缩放因子,有利于训练
                tech_indicator_list: list
                    技术指标名称列表(从config.py修改)
        方法
            -------
                create_env_training()
                    创建用于训练的env类
                create_env_validation()
                    创建用于验证的env类
                create_env_trading()
                    创建用于交易的env类
        """


初始化一个环境类:

用户定义的环境:模拟环境类。用于单股票交易的环境:

.. code-block:: python
   :linenos:

    import numpy as np
    import pandas as pd
    from gym.utils import seeding
    import gym
    from gym import spaces
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    class StockEnvSingle(gym.Env):
        """用于OpenAI gym的单股票交易环境
        属性
        ----------
            df: DataFrame
                    输入数据
            stock_dim : int
                    唯一股票数量
            hmax : int
                    交易的最大股份数
            initial_amount : int
                    初始资金
            transaction_cost_pct: float
                    每笔交易的交易费用百分比
            reward_scaling: float
                    奖励缩放因子,有利于训练
            state_space: int
                    输入特征的维度
            action_space: int
                    动作空间的维度
            tech_indicator_list: list
                    技术指标名称列表
            turbulence_threshold: int
                    控制风险厌恶的阈值
            day: int
                    控制日期的递增数字
        方法
        -------
            _sell_stock()
                根据动作的符号执行卖出操作
            _buy_stock()
                根据动作的符号执行买入操作
            step()
                在每个步骤,代理将返回动作,然后
                我们将计算奖励,并返回下一个观察。
            reset()
                重置环境
            render()
                使用render返回其他函数
            save_asset_memory()
                返回每个时间步的账户价值
            save_action_memory()
                返回每个时间步的动作/位置

        """
        metadata = {'render.modes': ['human']}

        def __init__(self,
                    df,
                    stock_dim,
                    hmax,
                    initial_amount,
                    transaction_cost_pct,
                    reward_scaling,
                    state_space,
                    action_space,
                    tech_indicator_list,
                    turbulence_threshold,
                    day = 0):
            self.day = day
            self.df = df
            self.stock_dim = stock_dim
            self.hmax = hmax
            self.initial_amount = initial_amount
            self.transaction_cost_pct = transaction_cost_pct
            self.reward_scaling = reward_scaling
            self.state_space = state_space
            self.action_space = action_space
            self.tech_indicator_list = tech_indicator_list

            # action_space归一化和形状是self.action_space
            self.action_space = spaces.Box(low = -1, high = 1,shape = (self.action_space,))
            # 形状 = 6:[账户余额] + [股价] + [持有股份] + [MACD] + [RSI] + [CCI] + [ADX]
            self.observation_space = spaces.Box(low=0, high=np.inf, shape = (self.state_space+len(self.tech_indicator_list)+1,))

            # 从pandas dataframe加载数据
            self.data = self.df.loc[self.day,:]
            self.state = [self.initial_amount] + \
                          [self.data.close] + \
                          [0] + \
                          [self.data[tech] for tech in self.tech_indicator_list]
            self.terminal = False
            self.turbulence_threshold = turbulence_threshold
            # 初始化账户余额
            self.asset_memory = [self.initial_amount]
            self.rewards_memory = []
            self.actions_memory=[]
            self.date_memory=[self.data.date]
            self._seed()

        def _sell_stock(self, index, action):
            # 根据动作的符号执行卖出操作
            if self.state[index+2] > 0: # 如果有股票
                # 更新账户余额
                self.state[index+0] += \
                self.state[index+1] * min(abs(action), self.state[index+2]) * \
                 (1 - self.transaction_cost_pct)

                self.state[index+2] -= min(abs(action), self.state[index+2])

        def _buy_stock(self, index, action):
            # 根据动作的符号执行买入操作
            available_amount = self.state[index+0] // self.state[index+1]

            # 更新账户余额
            self.state[index+0] -= self.state[index+1] * min(available_amount, action) * \
                              (1 + self.transaction_cost_pct)

            self.state[index+2] += min(available_amount, action)

        def step(self, actions):
            self.terminal = self.day >= len(self.df.index.unique())-1

            if self.terminal:
                df = pd.DataFrame(self.asset_memory)
                df.columns = ['account_value']
                df['daily_return']=df.account_value.pct_change(1)
                sharpe = (252**0.5)*df['daily_return'].mean()/ \
                           df['daily_return'].std()
                annual_return = ((df['daily_return'].mean()+1)**252-1)*100

                print("=================================")
                print("begin_total_asset:{}".format(self.asset_memory[0]))
                print("end_total_asset:{}".format(self.state[0]))
                print("total_profit:{}".format(self.state[0] - self.asset_memory[0]))
                print("total_reward:{}".format(self.state[0] - self.asset_memory[0]))
                print("total_cost: {}".format(self.cost))
                print("total trades: {}".format(self.trades))
                print("Sharpe: {}".format(sharpe))
                print("Annual Return: {}".format(annual_return))
                print("=================================")

                return self.state, self.reward, self.terminal,{}
            else:
                actions = actions * self.hmax

                # 执行卖出操作
                if actions < 0:
                    self._sell_stock(0, actions)

                # 执行买入操作
                elif actions > 0:
                    self._buy_stock(0, actions)

                # 保存动作
                self.actions_memory.append(actions)
                self.date_memory.append(self.data.date)

                self.day += 1
                self.data = self.df.loc[self.day,:]
                # 加载下一个状态
                self.state = [self.state[0]] + \
                          [self.data.close] + \
                          [self.state[2]] + \
                          [self.data[tech] for tech in self.tech_indicator_list]

                # 计算奖励
                reward = self.state[0] - self.asset_memory[-1]
                self.rewards_memory.append(reward)

                # 保存账户余额
                self.asset_memory.append(self.state[0])

            return self.state, reward, self.terminal, {}

        def reset(self):
            self.asset_memory = [self.initial_amount]
            self.day = 0
            self.data = self.df.loc[self.day,:]
            self.state = [self.initial_amount] + \
                          [self.data.close] + \
                          [0] + \
                          [self.data[tech] for tech in self.tech_indicator_list]
            self.terminal = False
            self.rewards_memory = []
            self.actions_memory=[]
            self.date_memory=[self.data.date]
            return self.state

        def render(self, mode='human'):
            return self.state

        def _seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]

实现DRL算法
--------------------------


FinRL使用DRLAgent类来实现算法。

.. code-block:: python

    class DRLAgent:
        """
            提供DRL算法的实现

            属性
            ----------
                env: gym environment类
                     用户定义的类
            方法
            -------
                train_PPO()
                    PPO算法的实现
                train_A2C()
                    A2C算法的实现
                train_DDPG()
                    DDPG算法的实现
                train_TD3()
                    TD3算法的实现
                DRL_prediction()
                    在测试数据集中进行预测并获取结果
        """

**模型训练**:

我们使用DDPG进行单股票交易,因为它在连续动作空间上表现良好。

交易:假设我们在2019/01/01拥有1,000,000美元的初始资本。我们使用DDPG模型来交易AAPL股票。

.. code-block:: python
   :linenos:

    trade = data_split(df,'2019-01-01', '2020-12-01')

    env_trade, obs_trade = env_setup.create_env_trading(data = trade,
                                             env_class = StockEnvSingle)

    df_account_value, df_actions = DRLAgent.DRL_prediction(model=model_ddpg,
                            test_data = trade,
                            test_env = env_trade,
                            test_obs = obs_trade)

回测性能
--------------------------

FinRL使用一组函数来使用Quantopian pyfolio进行回测。

.. code-block:: python
   :linenos:

    from pyfolio import timeseries
    DRL_strat = backtest_strat(df_account_value)
    perf_func = timeseries.perf_stats
    perf_stats_all = perf_func( returns=DRL_strat,
                                  factor_returns=DRL_strat,
                                    positions=None, transactions=None, turnover_denom="AGB")
    print("==============DRL Strategy Stats===========")
    perf_stats_all

    # 绘图
    aapl, aapl_strat = baseline_strat('AAPL','2019-01-01','2020-12-01')
    import pyfolio
    %matplotlib inline
    with pyfolio.plotting.plotting_context(font_scale=1.1):
            pyfolio.create_full_tear_sheet(returns = DRL_strat,
                                           benchmark_rets=aapl_strat, set_context=False)

左表是回测性能的统计,右表是AAPL股票性能的统计。


**绘图**:
