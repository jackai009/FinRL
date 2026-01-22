# 来自RLlib的DRL模型
from __future__ import annotations

import ray
from ray.rllib.algorithms.a2c import a2c
from ray.rllib.algorithms.ddpg import ddpg
from ray.rllib.algorithms.ppo import ppo
from ray.rllib.algorithms.sac import sac
from ray.rllib.algorithms.td3 import td3

MODELS = {"a2c": a2c, "ddpg": ddpg, "td3": td3, "sac": sac, "ppo": ppo}


# MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}


class DRLAgent:
    """DRL算法的实现

    属性
    ----------
        env: gym环境类
            用户定义的类
        price_array: numpy数组
            OHLC数据
        tech_array: numpy数组
            技术指标数据
        turbulence_array: numpy数组
            湍流/风险数据
    方法
    -------
        get_model()
            设置DRL算法
        train_model()
            在训练数据集中训练DRL算法
            并输出训练后的模型
        DRL_prediction()
            在测试数据集中进行预测并获得结果
    """

    def __init__(self, env, price_array, tech_array, turbulence_array):
        self.env = env
        self.price_array = price_array
        self.tech_array = tech_array
        self.turbulence_array = turbulence_array

    def get_model(
        self,
        model_name,
        # policy="MlpPolicy",
        # policy_kwargs=None,
        # model_kwargs=None,
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        # if model_kwargs is None:
        #    model_kwargs = MODEL_KWARGS[model_name]

        model = MODELS[model_name]
        # 基于RLlib中的算法获取算法默认配置
        if model_name == "a2c":
            model_config = model.A2C_DEFAULT_CONFIG.copy()
        elif model_name == "td3":
            model_config = model.TD3_DEFAULT_CONFIG.copy()
        else:
            model_config = model.DEFAULT_CONFIG.copy()
        # 将env、log_level、price_array、tech_array和turbulence_array传递给配置
        model_config["env"] = self.env
        model_config["log_level"] = "WARN"
        model_config["env_config"] = {
            "price_array": self.price_array,
            "tech_array": self.tech_array,
            "turbulence_array": self.turbulence_array,
            "if_train": True,
        }

        return model, model_config

    def train_model(
        self, model, model_name, model_config, total_episodes=100, init_ray=True
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        if init_ray:
            ray.init(
                ignore_reinit_error=True
            )  # 其他Ray API将无法工作，直到调用`ray.init()`。

        if model_name == "ppo":
            trainer = model.PPOTrainer(env=self.env, config=model_config)
        elif model_name == "a2c":
            trainer = model.A2CTrainer(env=self.env, config=model_config)
        elif model_name == "ddpg":
            trainer = model.DDPGTrainer(env=self.env, config=model_config)
        elif model_name == "td3":
            trainer = model.TD3Trainer(env=self.env, config=model_config)
        elif model_name == "sac":
            trainer = model.SACTrainer(env=self.env, config=model_config)

        for _ in range(total_episodes):
            trainer.train()

        ray.shutdown()

        # 保存训练好的模型
        cwd = "./test_" + str(model_name)
        trainer.save(cwd)

        return trainer

    @staticmethod
    def DRL_prediction(
        model_name,
        env,
        price_array,
        tech_array,
        turbulence_array,
        agent_path="./test_ppo/checkpoint_000100/checkpoint-100",
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_name == "a2c":
            model_config = MODELS[model_name].A2C_DEFAULT_CONFIG.copy()
        elif model_name == "td3":
            model_config = MODELS[model_name].TD3_DEFAULT_CONFIG.copy()
        else:
            model_config = MODELS[model_name].DEFAULT_CONFIG.copy()
        model_config["env"] = env
        model_config["log_level"] = "WARN"
        model_config["env_config"] = {
            "price_array": price_array,
            "tech_array": tech_array,
            "turbulence_array": turbulence_array,
            "if_train": False,
        }
        env_config = {
            "price_array": price_array,
            "tech_array": tech_array,
            "turbulence_array": turbulence_array,
            "if_train": False,
        }
        env_instance = env(config=env_config)

        # ray.init() # 其他Ray API将无法工作，直到调用`ray.init()`。
        if model_name == "ppo":
            trainer = MODELS[model_name].PPOTrainer(env=env, config=model_config)
        elif model_name == "a2c":
            trainer = MODELS[model_name].A2CTrainer(env=env, config=model_config)
        elif model_name == "ddpg":
            trainer = MODELS[model_name].DDPGTrainer(env=env, config=model_config)
        elif model_name == "td3":
            trainer = MODELS[model_name].TD3Trainer(env=env, config=model_config)
        elif model_name == "sac":
            trainer = MODELS[model_name].SACTrainer(env=env, config=model_config)

        try:
            trainer.restore(agent_path)
            print("从检查点路径恢复", agent_path)
        except BaseException:
            raise ValueError("加载智能体失败!")

        # 在测试环境中测试
        state = env_instance.reset()
        episode_returns = []  # 累计收益 / 初始账户
        episode_total_assets = [env_instance.initial_total_asset]
        done = False
        while not done:
            action = trainer.compute_single_action(state)
            state, reward, done, _ = env_instance.step(action)

            total_asset = (
                env_instance.amount
                + (env_instance.price_ary[env_instance.day] * env_instance.stocks).sum()
            )
            episode_total_assets.append(total_asset)
            episode_return = total_asset / env_instance.initial_total_asset
            episode_returns.append(episode_return)
        ray.shutdown()
        print("episode return: " + str(episode_return))
        print("测试完成!")
        return episode_total_assets
