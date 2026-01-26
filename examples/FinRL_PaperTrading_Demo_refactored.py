# 免责声明：本文内容不构成任何财务建议，也不是进行真实交易的推荐。存在许多模拟交易（纸面交易）平台，可用于构建和开发所讨论的方法。请在交易或投资前使用常识，并始终先咨询专业人士。

# 安装finrl库
# %pip install --upgrade git+https://github.com/AI4Finance-Foundation/FinRL.git
# Alpaca密钥
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data_key", help="数据源API密钥")
parser.add_argument("data_secret", help="数据源API密钥")
parser.add_argument("data_url", help="数据源API基础URL")
parser.add_argument("trading_key", help="交易API密钥")
parser.add_argument("trading_secret", help="交易API密钥")
parser.add_argument("trading_url", help="交易API基础URL")
args = parser.parse_args()
DATA_API_KEY = args.data_key
DATA_API_SECRET = args.data_secret
DATA_API_BASE_URL = args.data_url
TRADING_API_KEY = args.trading_key
TRADING_API_SECRET = args.trading_secret
TRADING_API_BASE_URL = args.trading_url

print("DATA_API_KEY: ", DATA_API_KEY)
print("DATA_API_SECRET: ", DATA_API_SECRET)
print("DATA_API_BASE_URL: ", DATA_API_BASE_URL)
print("TRADING_API_KEY: ", TRADING_API_KEY)
print("TRADING_API_SECRET: ", TRADING_API_SECRET)
print("TRADING_API_BASE_URL: ", TRADING_API_BASE_URL)

from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.meta.paper_trading.common import train, test, alpaca_history, DIA_history
from finrl.config import INDICATORS

# 导入道琼斯30指数成分股
from finrl.config_tickers import DOW_30_TICKER

ticker_list = DOW_30_TICKER
env = StockTradingEnv
# 如果您想使用更大的数据集（更改为更长的时间段），并且出现错误，请尝试增加"target_step"。它应该大于episode步数。
ERL_PARAMS = {
    "learning_rate": 3e-6,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": [128, 64],
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 1,
}

# 设置6天训练和2天测试的滑动窗口
import datetime
from pandas.tseries.offsets import BDay  # BDay是工作日，不是生日...

today = datetime.datetime.today()

TEST_END_DATE = (today - BDay(1)).to_pydatetime().date()
TEST_START_DATE = (TEST_END_DATE - BDay(1)).to_pydatetime().date()
TRAIN_END_DATE = (TEST_START_DATE - BDay(1)).to_pydatetime().date()
TRAIN_START_DATE = (TRAIN_END_DATE - BDay(5)).to_pydatetime().date()
TRAINFULL_START_DATE = TRAIN_START_DATE
TRAINFULL_END_DATE = TEST_END_DATE

TRAIN_START_DATE = str(TRAIN_START_DATE)
TRAIN_END_DATE = str(TRAIN_END_DATE)
TEST_START_DATE = str(TEST_START_DATE)
TEST_END_DATE = str(TEST_END_DATE)
TRAINFULL_START_DATE = str(TRAINFULL_START_DATE)
TRAINFULL_END_DATE = str(TRAINFULL_END_DATE)

print("TRAIN_START_DATE: ", TRAIN_START_DATE)
print("TRAIN_END_DATE: ", TRAIN_END_DATE)
print("TEST_START_DATE: ", TEST_START_DATE)
print("TEST_END_DATE: ", TEST_END_DATE)
print("TRAINFULL_START_DATE: ", TRAINFULL_START_DATE)
print("TRAINFULL_END_DATE: ", TRAINFULL_END_DATE)

train(
    start_date=TRAIN_START_DATE,
    end_date=TRAIN_END_DATE,
    ticker_list=ticker_list,
    data_source="alpaca",
    time_interval="1Min",
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    if_vix=True,
    API_KEY=DATA_API_KEY,
    API_SECRET=DATA_API_SECRET,
    API_BASE_URL=DATA_API_BASE_URL,
    erl_params=ERL_PARAMS,
    cwd="./papertrading_erl",  # 当前工作目录
    break_step=1e5,
)

account_value_erl = test(
    start_date=TEST_START_DATE,
    end_date=TEST_END_DATE,
    ticker_list=ticker_list,
    data_source="alpaca",
    time_interval="1Min",
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    if_vix=True,
    API_KEY=DATA_API_KEY,
    API_SECRET=DATA_API_SECRET,
    API_BASE_URL=DATA_API_BASE_URL,
    cwd="./papertrading_erl",
    net_dimension=ERL_PARAMS["net_dimension"],
)

train(
    start_date=TRAINFULL_START_DATE,  # 调优完成后，在训练集和测试集上重新训练
    end_date=TRAINFULL_END_DATE,
    ticker_list=ticker_list,
    data_source="alpaca",
    time_interval="1Min",
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    if_vix=True,
    API_KEY=DATA_API_KEY,
    API_SECRET=DATA_API_SECRET,
    API_BASE_URL=DATA_API_BASE_URL,
    erl_params=ERL_PARAMS,
    cwd="./papertrading_erl_retrain",
    break_step=2e5,
)

action_dim = len(DOW_30_TICKER)
state_dim = (
    1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim
)  # 手动计算纸面交易的DRL状态维度。金额 + (波动率, 波动率布尔值) + (价格, 持仓, cd (持仓时间)) * 股票维度 + 技术指标维度

paper_trading_erl = PaperTradingAlpaca(
    ticker_list=DOW_30_TICKER,
    time_interval="1Min",
    drl_lib="elegantrl",
    agent="ppo",
    cwd="./papertrading_erl_retrain",
    net_dim=ERL_PARAMS["net_dimension"],
    state_dim=state_dim,
    action_dim=action_dim,
    API_KEY=TRADING_API_KEY,
    API_SECRET=TRADING_API_SECRET,
    API_BASE_URL=TRADING_API_BASE_URL,
    tech_indicator_list=INDICATORS,
    turbulence_thresh=30,
    max_stock=1e2,
)

paper_trading_erl.run()

# 检查投资组合表现
# ## 获取累计收益率
df_erl, cumu_erl = alpaca_history(
    key=DATA_API_KEY,
    secret=DATA_API_SECRET,
    url=DATA_API_BASE_URL,
    start="2022-09-01",  # 必须在1个月内
    end="2022-09-12",
)  # 如果出现错误，请更改日期

df_djia, cumu_djia = DIA_history(start="2022-09-01")
returns_erl = cumu_erl - 1
returns_dia = cumu_djia - 1
returns_dia = returns_dia[: returns_erl.shape[0]]

# 绘图并保存
import matplotlib.pyplot as plt

plt.figure(dpi=1000)
plt.grid()
plt.grid(which="minor", axis="y")
plt.title("股票交易（纸面交易）", fontsize=20)
plt.plot(returns_erl, label="ElegantRL智能体", color="red")
# plt.plot(returns_sb3, label = 'Stable-Baselines3智能体', color = 'blue' )
# plt.plot(returns_rllib, label = 'RLlib智能体', color = 'green')
plt.plot(returns_dia, label="道琼斯工业指数", color="grey")
plt.ylabel("收益率", fontsize=16)
plt.xlabel("2021年", fontsize=16)
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca()
ax.xaxis.set_major_locator(ticker_list.MultipleLocator(78))
ax.xaxis.set_minor_locator(ticker_list.MultipleLocator(6))
ax.yaxis.set_minor_locator(ticker_list.MultipleLocator(0.005))
ax.yaxis.set_major_formatter(ticker_list.PercentFormatter(xmax=1, decimals=2))
ax.xaxis.set_major_formatter(
    ticker_list.FixedFormatter(["", "10-19", "", "10-20", "", "10-21", "", "10-22"])
)
plt.legend(fontsize=10.5)
plt.savefig("papertrading_stock.png")
