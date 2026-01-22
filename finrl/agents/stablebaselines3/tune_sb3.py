from __future__ import annotations

import datetime

import joblib
import optuna
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3

import finrl.agents.stablebaselines3.hyperparams_opt as hpt
from finrl import config
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.main import check_and_make_directories
from finrl.plot import backtest_stats


class LoggingCallback:
    def __init__(self, threshold: int, trial_number: int, patience: int):
        """
        threshold:int 夏普比率增加的容差
        trial_number: int 在最少试验次数后剪枝
        patience: int 阈值的耐心
        """
        self.threshold = threshold
        self.trial_number = trial_number
        self.patience = patience
        self.cb_list = []  # Trials list for which threshold is reached

    def __call__(self, study: optuna.study, frozen_trial: optuna.Trial):
        # 在当前试验中设置最佳值
        study.set_user_attr("previous_best_value", study.best_value)

        # 检查是否已通过最少试验次数
        if frozen_trial.number > self.trial_number:
            previous_best_value = study.user_attrs.get("previous_best_value", None)
            # 检查先前和当前目标值是否具有相同的符号
            if previous_best_value * study.best_value >= 0:
                # 检查阈值条件
                if abs(previous_best_value - study.best_value) < self.threshold:
                    self.cb_list.append(frozen_trial.number)
                    # 如果阈值在耐心时间内达到
                    if len(self.cb_list) > self.patience:
                        print("研究现在停止...")
                        print(
                            "编号为",
                            frozen_trial.number,
                            "值为 ",
                            frozen_trial.value,
                        )
                        print(
                            "先前和当前的最佳值分别为 {} 和 {}".format(
                                previous_best_value, study.best_value
                            )
                        )
                        study.stop()


class TuneSB3Optuna:
    """
    使用Optuna进行SB3智能体的超参数调优

    属性
    ----------
      env_train: SB3的训练环境
      model_name: str
      env_trade: 测试环境
      logging_callback: 调优的回调
      total_timesteps: int
      n_trials: 超参数配置的数量

    注意：
      默认使用的采样器和剪枝器分别是
      树Parzen估计器和超带调度器
    """

    def __init__(
        self,
        env_train,
        model_name: str,
        env_trade,
        logging_callback,
        total_timesteps: int = 50000,
        n_trials: int = 30,
    ):
        self.env_train = env_train
        self.agent = DRLAgent(env=env_train)
        self.model_name = model_name
        self.env_trade = env_trade
        self.total_timesteps = total_timesteps
        self.n_trials = n_trials
        self.logging_callback = logging_callback
        self.MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

        check_and_make_directories(
            [
                config.DATA_SAVE_DIR,
                config.TRAINED_MODEL_DIR,
                config.TENSORBOARD_LOG_DIR,
                config.RESULTS_DIR,
            ]
        )

    def default_sample_hyperparameters(self, trial: optuna.Trial):
        if self.model_name == "a2c":
            return hpt.sample_a2c_params(trial)
        elif self.model_name == "ddpg":
            return hpt.sample_ddpg_params(trial)
        elif self.model_name == "td3":
            return hpt.sample_td3_params(trial)
        elif self.model_name == "sac":
            return hpt.sample_sac_params(trial)
        elif self.model_name == "ppo":
            return hpt.sample_ppo_params(trial)

    def calculate_sharpe(self, df: pd.DataFrame):
        df["daily_return"] = df["account_value"].pct_change(1)
        if df["daily_return"].std() != 0:
            sharpe = (252**0.5) * df["daily_return"].mean() / df["daily_return"].std()
            return sharpe
        else:
            return 0

    def objective(self, trial: optuna.Trial):
        hyperparameters = self.default_sample_hyperparameters(trial)
        policy_kwargs = hyperparameters["policy_kwargs"]
        del hyperparameters["policy_kwargs"]
        model = self.agent.get_model(
            self.model_name, policy_kwargs=policy_kwargs, model_kwargs=hyperparameters
        )
        trained_model = self.agent.train_model(
            model=model,
            tb_log_name=self.model_name,
            total_timesteps=self.total_timesteps,
        )
        trained_model.save(
            f"./{config.TRAINED_MODEL_DIR}/{self.model_name}_{trial.number}.pth"
        )
        df_account_value, _ = DRLAgent.DRL_prediction(
            model=trained_model, environment=self.env_trade
        )
        sharpe = self.calculate_sharpe(df_account_value)

        return sharpe

    def run_optuna(self):
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            study_name=f"{self.model_name}_study",
            direction="maximize",
            sampler=sampler,
            pruner=optuna.pruners.HyperbandPruner(),
        )

        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            catch=(ValueError,),
            callbacks=[self.logging_callback],
        )

        joblib.dump(study, f"{self.model_name}_study.pkl")
        return study

    def backtest(
        self, final_study: optuna.Study
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("Hyperparameters after tuning", final_study.best_params)
        print("Best Trial", final_study.best_trial)

        tuned_model = self.MODELS[self.model_name].load(
            f"./{config.TRAINED_MODEL_DIR}/{self.model_name}_{final_study.best_trial.number}.pth",
            env=self.env_train,
        )

        df_account_value_tuned, df_actions_tuned = DRLAgent.DRL_prediction(
            model=tuned_model, environment=self.env_trade
        )

        print("==============Get Backtest Results===========")
        now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

        perf_stats_all_tuned = backtest_stats(account_value=df_account_value_tuned)
        perf_stats_all_tuned = pd.DataFrame(perf_stats_all_tuned)
        perf_stats_all_tuned.to_csv(
            "./" + config.RESULTS_DIR + "/perf_stats_all_tuned_" + now + ".csv"
        )

        return df_account_value_tuned, df_actions_tuned, perf_stats_all_tuned
