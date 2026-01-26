"""
来自ElegantRL的DRL模型：https://github.com/AI4Finance-Foundation/ElegantRL
"""

from __future__ import annotations

import torch
from elegantrl.agents import *
from elegantrl.train.config import Config
from elegantrl.train.run import train_agent

MODELS = {
    "ddpg": AgentDDPG,
    "td3": AgentTD3,
    "sac": AgentSAC,
    "ppo": AgentPPO,
    "a2c": AgentA2C,
}
OFF_POLICY_MODELS = ["ddpg", "td3", "sac"]
ON_POLICY_MODELS = ["ppo"]
# MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}
#
# NOISE = {
#     "normal": NormalActionNoise,
#     "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
# }


class DRLAgent:
    """DRL算法的实现
    属性
    ----------
        env: gym环境类
            用户定义的类
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

    def get_model(self, model_name, model_kwargs):
        self.env_config = {
            "price_array": self.price_array,
            "tech_array": self.tech_array,
            "turbulence_array": self.turbulence_array,
            "if_train": True,
        }
        self.model_kwargs = model_kwargs
        self.gamma = model_kwargs.get("gamma", 0.985)

        env = self.env
        env.env_num = 1
        agent = MODELS[model_name]
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        stock_dim = self.price_array.shape[1]
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_array.shape[1]
        self.action_dim = stock_dim
        self.env_args = {
            "env_name": "StockEnv",
            "config": self.env_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "if_discrete": False,
            "max_step": self.price_array.shape[0] - 1,
        }

        model = Config(agent_class=agent, env_class=env, env_args=self.env_args)
        model.if_off_policy = model_name in OFF_POLICY_MODELS
        if model_kwargs is not None:
            try:
                model.break_step = int(
                    2e5
                )  # 如果'total_step > break_step'则中断训练
                model.net_dims = (
                    128,
                    64,
                )  # 多层感知机的中间层维度
                model.gamma = self.gamma  # 未来奖励的折扣因子
                model.horizon_len = model.max_step
                model.repeat_times = 16  # 重复使用ReplayBuffer更新网络以保持评论家的损失较小
                model.learning_rate = model_kwargs.get("learning_rate", 1e-4)
                model.state_value_tau = 0.1  # 价值和状态标准化的tau `std = (1-std)*std + tau*std`
                model.eval_times = model_kwargs.get("eval_times", 2**5)
                model.eval_per_step = int(2e4)
            except BaseException:
                raise ValueError("读取参数失败，请检查'model_kwargs'输入。")
        return model

    def train_model(self, model, cwd, total_timesteps=5000):
        model.cwd = cwd
        model.break_step = total_timesteps
        train_agent(model)

    @staticmethod
    def DRL_prediction(model_name, cwd, net_dimension, environment, env_args):
        import torch

        gpu_id = 0  # >=0 means GPU ID, -1 means CPU
        agent_class = MODELS[model_name]
        stock_dim = env_args["price_array"].shape[1]
        state_dim = 1 + 2 + 3 * stock_dim + env_args["tech_array"].shape[1]
        action_dim = stock_dim
        env_args = {
            "env_num": 1,
            "env_name": "StockEnv",
            "state_dim": state_dim,
            "action_dim": action_dim,
            "if_discrete": False,
            "max_step": env_args["price_array"].shape[0] - 1,
            "config": env_args,
        }

        actor_path = f"{cwd}/act.pth"
        net_dim = [2**7]

        """初始化"""
        env = environment
        env_class = env
        args = Config(agent_class=agent_class, env_class=env_class, env_args=env_args)
        args.cwd = cwd
        act = agent_class(
            net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id, args=args
        ).act
        parameters_dict = {}
        act = torch.load(actor_path)
        for name, param in act.named_parameters():
            parameters_dict[name] = torch.tensor(param.detach().cpu().numpy())

        act.load_state_dict(parameters_dict)

        if_discrete = env.if_discrete
        device = next(act.parameters()).device
        state = env.reset()
        episode_returns = []  # 累计收益 / 初始账户
        episode_total_assets = [env.initial_total_asset]
        max_step = env.max_step
        for steps in range(max_step):
            s_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            a_tensor = act(s_tensor).argmax(dim=1) if if_discrete else act(s_tensor)
            action = (
                a_tensor.detach().cpu().numpy()[0]
            )  # 不需要detach()，因为外部使用torch.no_grad()
            state, reward, done, _ = env.step(action)
            total_asset = env.amount + (env.price_ary[env.day] * env.stocks).sum()
            episode_total_assets.append(total_asset)
            episode_return = total_asset / env.initial_total_asset
            episode_returns.append(episode_return)
            if done:
                break
        print("测试完成!")
        print("episode_return", episode_return)
        return episode_total_assets
