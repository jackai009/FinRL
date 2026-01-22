# 投资组合优化智能体

此目录包含投资组合优化智能体中常用的架构和算法。

要实例化模型，必须有[PortfolioOptimizationEnv](/finrl/meta/env_portfolio_optimization/)的实例。在下面的示例中，我们使用`DRLAgent`类来实例化策略梯度（"pg"）模型。通过字典`model_kwargs`，我们可以设置`PolicyGradient`类参数，而通过字典`policy_kwargs`，可以更改所选架构的参数。

```python
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures import EIIE

# 设置策略梯度算法参数
model_kwargs = {
    "lr": 0.01,
    "policy": EIIE,
}

# 设置EIIE架构参数
policy_kwargs = {
    "k_size": 4
}

model = DRLAgent(train_env).get_model("pg", model_kwargs, policy_kwargs)
```

在下面的示例中，模型在5个回合中进行训练（我们将一个回合定义为所用环境的完整周期）。

```python
DRLAgent.train_model(model, episodes=5)
```

重要的是架构和环境具有相同的`time_window`定义。默认情况下，它们都使用50个时间步作为`time_window`。有关时间窗口的更多详细信息，请查看此[文章](https://doi.org/10.5753/bwaif.2023.231144)。

### 策略梯度算法

`PolicyGradient`类实现了*Jiang等人*论文中使用的策略梯度算法。该算法受到DDPG（深度确定性策略梯度）的启发，但有一些差异：
- DDPG是演员-评论家算法，因此它有演员和评论家神经网络。然而，下面的算法没有评论家神经网络，而是使用投资组合价值作为价值函数：策略将被更新以最大化投资组合价值。
- DDPG通常在训练期间在行动中使用噪声参数来创建探索行为。另一方面，PG算法采用完全开发的方法。
- DDPG随机从其回放缓冲区中采样经验。然而，实施的策略梯度在时间上按顺序采样一批经验，以便能够计算批次中投资组合价值的变化并将其用作价值函数。

该算法的实施如下：
1. 初始化策略网络和回放缓冲区；
2. 对于每个回合，执行以下操作：
    1. 对于`batch_size`时间步的每个周期，执行以下操作：
        1. 对于每个时间步，定义要执行的行动，模拟时间步并将经验保存在回放缓冲区中。
        2. 模拟`batch_size`时间步后，对回放缓冲区进行采样。
        4. 计算价值函数：$V = \sum\limits_{t=1}^{batch\_size} ln(\mu_{t}(W_{t} \cdot P_{t}))$，其中$W_{t}$是在时间步t执行的行動，$P_{t}$是在时间步t的价格变化向量，$\mu_{t}$是在时间步t的交易剩余因子。查看*Jiang等人*论文以获取更多详细信息。
        5. 在策略网络中执行梯度上升。
    2. 如果在回合结束时，回放缓冲区中有一系列剩余的经验，请对剩余的经验执行步骤1至5。

### 参考文献

如果您在研究中使用其中一个，可以使用以下参考文献。

#### EIIE Architecture and Policy Gradient algorithm

[A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://doi.org/10.48550/arXiv.1706.10059)
```
@misc{jiang2017deep,
      title={A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem},
      author={Zhengyao Jiang and Dixing Xu and Jinjun Liang},
      year={2017},
      eprint={1706.10059},
      archivePrefix={arXiv},
      primaryClass={q-fin.CP}
}
```

#### EI3 Architecture

[A Multi-Scale Temporal Feature Aggregation Convolutional Neural Network for Portfolio Management](https://doi.org/10.1145/3357384.3357961)
```
@inproceedings{shi2018multiscale,
               author = {Shi, Si and Li, Jianjun and Li, Guohui and Pan, Peng},
               title = {A Multi-Scale Temporal Feature Aggregation Convolutional Neural Network for Portfolio Management},
               year = {2019},
               isbn = {9781450369763},
               publisher = {Association for Computing Machinery},
               address = {New York, NY, USA},
               url = {https://doi.org/10.1145/3357384.3357961},
               doi = {10.1145/3357384.3357961},
               booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
               pages = {1613–1622},
               numpages = {10},
               keywords = {portfolio management, reinforcement learning, inception network, convolution neural network},
               location = {Beijing, China},
               series = {CIKM '19} }
```
