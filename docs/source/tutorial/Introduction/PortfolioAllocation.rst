:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

投资组合分配
===================================

**我们的论文**：
`FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance`_.

.. _FinRL\: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance: https://arxiv.org/abs/2011.09607

在NeurIPS 2020:深度强化学习研讨会上展示。

Jupyter笔记本代码可在我们的Github_和`Google Colab`_上找到。

.. _Github: https://github.com/AI4Finance-LLC/FinRL-Library
.. _Google Colab: https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/FinRL_multiple_stock_trading.ipynb

.. tip::

    - FinRL `Single Stock Trading <https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/FinRL_single_stock_trading.ipynb>`_ at Google Colab.

    - FinRL `Multiple Stocks Trading <https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/FinRL_multiple_stock_trading.ipynb>`_ at Google Colab:

查看我们之前的教程:`Single Stock Trading <https://finrl.readthedocs.io/en/latest/tutorial/SingleStockTrading.html>`_和`Multiple Stock Trading <https://finrl.readthedocs.io/en/latest/tutorial/MultipleStockTrading.html>`_以获取FinRL架构和模块的详细解释。



概述
-------------

首先,我们想解释使用深度强化学习进行投资组合分配的逻辑。我们以道琼斯30只成分股为例,因为它们是最受欢迎的股票。

假设我们在2019年初获得了100万美元。我们想将这1,000,000美元投资于股票市场,在这种情况下是道琼斯30只成分股。假设没有保证金交易,没有卖空,没有国库券(使用所有资金仅交易这30只股票)。因此,每只股票的权重是非负的,所有股票的权重加起来等于一。

我们聘请了一位聪明的投资组合经理——深度强化学习先生。DRL先生每天都会给我们建议,包括投资组合权重或投资于这30只股票的资金比例。所以我们每天只需要重新平衡股票的投资组合权重。基本逻辑如下。
.. image:: ../../image/portfolio_allocation_1.png

投资组合分配与多股票交易不同,因为我们在每个时间步骤本质上重新平衡权重,我们必须使用所有可用资金。

进行投资组合分配的传统和最流行的方式是均值方差或现代投资组合理论(MPT):

.. image:: ../../image/portfolio_allocation_2.png


然而,MPT在样本外数据上的表现不是很好。MPT仅基于股票回报计算,如果我们想考虑其他相关因素,例如一些技术指标如MACD或RSI,MPT可能无法很好地结合这些信息。

我们介绍一个DRL库FinRL,它促进初学者接触量化金融。FinRL是一个专门为自动化股票交易设计的DRL库,致力于教育和演示目的。

本文重点关注我们论文中的用例之一:投资组合分配。我们使用一个Jupyter notebook来包含所有必要的步骤。



问题定义
--------------------------

这个问题是设计用于投资组合分配的自动化交易解决方案。我们将股票交易过程建模为马尔可夫决策过程(MDP)。然后我们将我们的交易目标制定为最大化问题。

强化学习环境的组件包括:

    - **Action**:每只股票的投资组合权重在[0,1]范围内。我们使用softmax函数将动作归一化为总和为1。

    - **State: {Covariance Matrix, MACD, RSI, CCI, ADX}, **state space** 形状是(34, 30)。34是行数,30是列数。

    - **Reward function**: r(s, a, s′) = p_t, p_t是累计投资组合价值。

    - **Environment**:道琼斯30只成分股的投资组合分配。

协方差矩阵是一个很好的特征,因为投资组合经理使用它来量化与特定投资组合相关的风险(标准差)。

我们还假设没有交易成本,因为我们试图使一个简单的投资组合分配案例作为起点。



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
    from finrl.env.EnvMultipleStock_train import StockEnvTrain
    from finrl.env.EnvMultipleStock_trade import StockEnvTrade
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
    df = YahooDownloader(start_date = '2008-01-01',
                         end_date = '2020-12-01',
                         ticker_list = config_tickers.DOW_30_TICKER).fetch_data()


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

执行特征工程:协方差矩阵 + 技术指标:

.. code-block:: python
   :linenos:

    # 执行特征工程:
    df = FeatureEngineer(df.copy(),
                        use_technical_indicator=True,
                        use_turbulence=False).preprocess_data()

    # 添加协方差矩阵作为状态
    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list = []
    # 回顾一年
    lookback=252
    for i in range(lookback,len(df.index.unique())):
      data_lookback = df.loc[i-lookback:i,:]
      price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
      return_lookback = price_lookback.pct_change().dropna()
      covs = return_lookback.cov().values
      cov_list.append(covs)

    df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date','tic']).reset_index(drop=True)
    df.head()

.. image:: ../../image/portfolio_allocation_3.png

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

用户定义的环境:模拟环境类。用于投资组合分配的环境:

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

    class StockPortfolioEnv(gym.Env):
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
                    等于股票维度
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
                    lookback=252,
                    day = 0):
            #super(StockEnv, self).__init__()
            #money = 10 , scope = 1
            self.day = day
            self.lookback=lookback
            self.df = df
            self.stock_dim = stock_dim
            self.hmax = hmax
            self.initial_amount = initial_amount
            self.transaction_cost_pct =transaction_cost_pct
            self.reward_scaling = reward_scaling
            self.state_space = state_space
            self.action_space = action_space
            self.tech_indicator_list = tech_indicator_list

            # action_space归一化和形状是self.stock_dim
            self.action_space = spaces.Box(low = 0, high = 1,shape = (self.action_space,))
            # 形状 = (34, 30)
            # 协方差矩阵 + 技术指标
            self.observation_space = spaces.Box(low=0,
                                                high=np.inf,
                                                shape = (self.state_space+len(self.tech_indicator_list),
                                                         self.state_space))

            # 从pandas dataframe加载数据
            self.data = self.df.loc[self.day,:]
            self.covs = self.data['cov_list'].values[0]
            self.state =  np.append(np.array(self.covs),
                          [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
            self.terminal = False
            self.turbulence_threshold = turbulence_threshold
            # 初始化状态:初始投资组合回报 + 个股回报 + 个股权重
            self.portfolio_value = self.initial_amount

            # 记住每个步骤的投资组合价值
            self.asset_memory = [self.initial_amount]
            # 记住每个步骤的投资组合回报
            self.portfolio_return_memory = [0]
            self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
            self.date_memory=[self.data.date.unique()[0]]

        def step(self, actions):
            # print(self.day)
            self.terminal = self.day >= len(self.df.index.unique())-1
            # print(actions)

            if self.terminal:
                df = pd.DataFrame(self.portfolio_return_memory)
                df.columns = ['daily_return']
                plt.plot(df.daily_return.cumsum(),'r')
                plt.savefig('results/cumulative_reward.png')
                plt.close()

                plt.plot(self.portfolio_return_memory,'r')
                plt.savefig('results/rewards.png')
                plt.close()

                print("=================================")
                print("begin_total_asset:{}".format(self.asset_memory[0]))
                print("end_total_asset:{}".format(self.portfolio_value))

                df_daily_return = pd.DataFrame(self.portfolio_return_memory)
                df_daily_return.columns = ['daily_return']
                if df_daily_return['daily_return'].std() !=0:
                  sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
                           df_daily_return['daily_return'].std()
                  print("Sharpe: ",sharpe)
                print("=================================")

                return self.state, self.reward, self.terminal,{}
            else:
                #print(actions)
                # actions是投资组合权重
                # 归一化为总和为1
                norm_actions = (np.array(actions) - np.array(actions).min()) / (np.array(actions) - np.array(actions).min()).sum()
                weights = norm_actions
                #print(weights)
                self.actions_memory.append(weights)
                last_day_memory = self.data

                #加载下一个状态
                self.day += 1
                self.data = self.df.loc[self.day,:]
                self.covs = self.data['cov_list'].values[0]
                self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
                # 计算投资组合回报
                # 个股回报 * 权重
                portfolio_return = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights)
                # 更新投资组合价值
                new_portfolio_value = self.portfolio_value*(1+portfolio_return)
                self.portfolio_value = new_portfolio_value

                # 保存到内存
                self.portfolio_return_memory.append(portfolio_return)
                self.date_memory.append(self.data.date.unique()[0])
                self.asset_memory.append(new_portfolio_value)

                # 奖励是新投资组合价值或最终投资组合价值
                self.reward = new_portfolio_value
                #self.reward = self.reward*self.reward_scaling


            return self.state, self.reward, self.terminal, {}

        def reset(self):
            self.asset_memory = [self.initial_amount]
            self.day = 0
            self.data = self.df.loc[self.day,:]
            # 加载状态
            self.covs = self.data['cov_list'].values[0]
            self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
            self.portfolio_value = self.initial_amount
            #self.cost = 0
            #self.trades = 0
            self.terminal = False
            self.portfolio_return_memory = [0]
            self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
            self.date_memory=[self.data.date.unique()[0]]
            return self.state

        def render(self, mode='human'):
            return self.state

        def save_asset_memory(self):
            date_list = self.date_memory
            portfolio_return = self.portfolio_return_memory
            #print(len(date_list))
            #print(len(asset_list))
            df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
            return df_account_value

        def save_action_memory(self):
            # 日期和收盘价长度必须与动作长度匹配
            date_list = self.date_memory
            df_date = pd.DataFrame(date_list)
            df_date.columns = ['date']

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
            return df_actions

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

我们使用A2C进行投资组合分配,因为它是稳定的、成本效益的、更快的,并且在更大的批量大小下工作得更好。

交易:假设我们在2019/01/01拥有1,000,000美元的初始资本。我们使用A2C模型来对道琼斯30只股票执行投资组合分配。

.. code-block:: python
   :linenos:

    trade = data_split(df,'2019-01-01', '2020-12-01')

    env_trade, obs_trade = env_setup.create_env_trading(data = trade,
                                             env_class = StockPortfolioEnv)

    df_daily_return, df_actions = DRLAgent.DRL_prediction(model=model_a2c,
                            test_data = trade,
                            test_env = env_trade,
                            test_obs = obs_trade)

.. image:: ../../image/portfolio_allocation_4.png

输出动作或投资组合权重看起来像这样:

.. image:: ../../image/portfolio_allocation_5.png

回测性能
--------------------------

FinRL使用一组函数来使用Quantopian pyfolio进行回测。

.. code-block:: python
   :linenos:

    from pyfolio import timeseries
    DRL_strat = backtest_strat(df_daily_return)
    perf_func = timeseries.perf_stats
    perf_stats_all = perf_func( returns=DRL_strat,
                                  factor_returns=DRL_strat,
                                    positions=None, transactions=None, turnover_denom="AGB")
    print("==============DRL Strategy Stats===========")
    perf_stats_all
    print("==============Get Index Stats===========")
    baesline_perf_stats=BaselineStats('^DJI',
                                      baseline_start = '2019-01-01',
                                      baseline_end = '2020-12-01')


    # 绘图
    dji, dow_strat = baseline_strat('^DJI','2019-01-01','2020-12-01')
    import pyfolio
    %matplotlib inline
    with pyfolio.plotting.plotting_context(font_scale=1.1):
            pyfolio.create_full_tear_sheet(returns = DRL_strat,
                                           benchmark_rets=dow_strat, set_context=False)

左表是回测性能的统计,右表是指数(DJIA)性能的统计。


**绘图**:
