:github_url: https://github.com/AI4Finance-Foundation/FinRL

快速开始
==================

打开``main.py``

.. code-block:: python
    :linenos:

    import os
    from typing import List
    from argparse import ArgumentParser
    from finrl import config
    from finrl.config_tickers import DOW_30_TICKER
    from finrl.config import (
        DATA_SAVE_DIR,
        TRAINED_MODEL_DIR,
        TENSORBOARD_LOG_DIR,
        RESULTS_DIR,
        INDICATORS,
        TRAIN_START_DATE,
        TRAIN_END_DATE,
        TEST_START_DATE,
        TEST_END_DATE,
        TRADE_START_DATE,
        TRADE_END_DATE,
        ERL_PARAMS,
        RLlib_PARAMS,
        SAC_PARAMS,
        ALPACA_API_KEY,
        ALPACA_API_SECRET,
        ALPACA_API_BASE_URL,
    )

    # construct environment
    from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv


    def build_parser():
        parser = ArgumentParser()
        parser.add_argument(
            "--mode",
            dest="mode",
            help="start mode, train, download_data" " backtest",
            metavar="MODE",
            default="train",
        )
        return parser


    # "./"将添加到每个目录的前面
    def check_and_make_directories(directories: List[str]):
        for directory in directories:
            if not os.path.exists("./" + directory):
                os.makedirs("./" + directory)



    def main():
        parser = build_parser()
        options = parser.parse_args()
        check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

        if options.mode == "train":
            from finrl import train

            env = StockTradingEnv

            # elegantrl示例
            kwargs = {}  # 在当前meta中，关于yahoofinance，kwargs为{}。对于其他数据源，如joinquant，kwargs不为空
            train(
                start_date=TRAIN_START_DATE,
                end_date=TRAIN_END_DATE,
                ticker_list=DOW_30_TICKER,
                data_source="yahoofinance",
                time_interval="1D",
                technical_indicator_list=INDICATORS,
                drl_lib="elegantrl",
                env=env,
                model_name="ppo",
                cwd="./test_ppo",
                erl_params=ERL_PARAMS,
                break_step=1e5,
                kwargs=kwargs,
            )
        elif options.mode == "test":
            from finrl import test
            env = StockTradingEnv

            # elegantrl示例
            kwargs = {}  # 在当前meta中，关于yahoofinance，kwargs为{}。对于其他数据源，如joinquant，kwargs不为空

            account_value_erl = test(
                start_date=TEST_START_DATE,
                end_date=TEST_END_DATE,
                ticker_list=DOW_30_TICKER,
                data_source="yahoofinance",
                time_interval="1D",
                technical_indicator_list=INDICATORS,
                drl_lib="elegantrl",
                env=env,
                model_name="ppo",
                cwd="./test_ppo",
                net_dimension=512,
                kwargs=kwargs,
            )
        elif options.mode == "trade":
            from finrl import trade
            env = StockTradingEnv
            kwargs = {}
            trade(
                start_date=TRADE_START_DATE,
                end_date=TRADE_END_DATE,
                ticker_list=DOW_30_TICKER,
                data_source="yahoofinance",
                time_interval="1D",
                technical_indicator_list=INDICATORS,
                drl_lib="elegantrl",
                env=env,
                model_name="ppo",
                API_KEY=ALPACA_API_KEY,
                API_SECRET=ALPACA_API_SECRET,
                API_BASE_URL=ALPACA_API_BASE_URL,
                trade_mode='backtesting',
                if_vix=True,
                kwargs=kwargs,
            )
        else:
            raise ValueError("Wrong mode.")


    ## 用户可以在终端中输入以下命令
    # python main.py --mode=train
    # python main.py --mode=test
    # python main.py --mode=trade
    if __name__ == "__main__":
        main()


运行库：

.. code-block:: python

    python main.py --mode=train # 如果是训练。默认使用DOW_30_TICKER。
    python main.py --mode=test  # 如果是测试。默认使用DOW_30_TICKER。
    python main.py --mode=trade # 如果是交易。用户应在config.py中输入您的alpaca参数

``--mode``选项：启动模式、训练、下载数据、回测
