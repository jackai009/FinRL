"""
用于通过强化学习解决投资组合优化任务的DRL模型。
该智能体被开发用于与PortfolioOptimizationEnv等环境一起工作。
"""

from __future__ import annotations

from .algorithms import PolicyGradient

MODELS = {"pg": PolicyGradient}


class DRLAgent:
    """投资组合优化的DRL算法实现。

    注意：
        在测试期间，智能体通过在线学习进行优化。
        策略的参数在固定时间段后重复更新。要禁用它，请将学习率设置为0。

    属性：
        env: Gym环境类。
    """

    def __init__(self, env):
        """智能体初始化。

        参数：
            env: 用于训练的Gym环境。
        """
        self.env = env

    def get_model(
        self, model_name, device="cpu", model_kwargs=None, policy_kwargs=None
    ):
        """设置DRL模型。

        参数：
            model_name: 根据MODELS列表的模型名称。
            device: 用于实例化神经网络的设备。
            model_kwargs: 要传递给模型类的参数。
            policy_kwargs: 要传递给策略类的参数。

        注意：
            model_kwargs和policy_kwargs是字典。键必须是字符串，
            名称与类参数相同。model_kwargs的示例：

            { "lr": 0.01, "policy": EIIE }

        返回：
            模型的实例。
        """
        if model_name not in MODELS:
            raise NotImplementedError("The model requested was not implemented.")

        model = MODELS[model_name]
        model_kwargs = {} if model_kwargs is None else model_kwargs
        policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        # add device settings
        model_kwargs["device"] = device
        policy_kwargs["device"] = device

        # add policy_kwargs inside model_kwargs
        model_kwargs["policy_kwargs"] = policy_kwargs

        return model(self.env, **model_kwargs)

    @staticmethod
    def train_model(model, episodes=100):
        """训练投资组合优化模型。

        参数：
            model: 模型的实例。
            episodes: 回合数。

        返回：
            训练后的模型实例。
        """
        model.train(episodes)
        return model

    @staticmethod
    def DRL_validation(
        model,
        test_env,
        policy=None,
        online_training_period=10,
        learning_rate=None,
        optimizer=None,
    ):
        """在测试环境中测试模型。

        参数：
            model: 模型的实例。
            test_env: 用于测试的Gym环境。
            policy: 要使用的策略架构。如果为None，将使用训练
            架构。
            online_training_period: 在线训练将发生的周期。要
                禁用在线学习，请使用非常大的值。
            batch_size: 训练神经网络的批次大小。如果为None，将使用
                训练批次大小。
            lr: 策略神经网络学习率。如果为None，将使用训练
                学习率
            optimizer: 神经网络的优化器。如果为None，将使用训练
                优化器
        """
        model.test(test_env, policy, online_training_period, learning_rate, optimizer)
