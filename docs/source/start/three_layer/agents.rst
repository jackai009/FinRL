:github_url: https://github.com/AI4Finance-Foundation/FinRL

2. DRL智能体
============================

FinRL包含ElegantRL、Stable Baseline 3和RLlib中经过微调的标准DRL算法。ElegantRL是由AI4Finance维护的可扩展且灵活的DRL库，性能比Stable Baseline 3和RLlib更快、更稳定。在*三层架构*部分，将详细解释ElegantRL如何完美完成其在FinRL中的角色。如有兴趣，请参阅ElegantRL的`GitHub页面 <https://github.com/AI4Finance-Foundation/ElegantRL>`_或`文档 <https://elegantrl.readthedocs.io>`_。

凭借这三个强大的DRL库，FinRL为用户提供以下算法：

.. image:: ../../image/alg_compare.png

正如介绍中提到的，FinRL的DRL智能体基于三个著名的DRL库构建：ElegantRL、Stable Baseline 3和RLlib，采用经过微调的标准DRL算法。

支持的算法包括：DQN、DDPG、多智能体DDPG、PPO、SAC、A2C和TD3。我们还允许用户通过调整这些DRL算法（例如自适应DDPG）或采用集成方法来设计自己的DRL算法。DRL算法的比较如下表所示：

.. image:: ../../image/alg_compare.png
   :align: center

用户能够选择自己喜欢的DRL智能体进行训练。不同的DRL智能体在不同任务中可能具有不同的性能。

ElegantRL：DRL库
------------------------

.. image:: ../../image/ElegantRL_icon.jpeg
    :width: 30%
    :align: center
    :target: https://github.com/AI4Finance-Foundation/ElegantRL


强化学习（RL）的一句话总结：在RL中，智能体通过不断与未知环境交互，以试错的方式学习，在不确定性下做出序列决策，并在探索（新领域）和利用（使用从经验中学到的知识）之间实现平衡。

深度强化学习（DRL）在解决对人类具有挑战性的现实世界问题方面具有巨大潜力，例如游戏、自然语言处理（NLP）、自动驾驶汽车和金融交易。从AlphaGo的成功开始，各种DRL算法和应用以颠覆性的方式涌现。ElegantRL库使研究人员和实践者能够流水线化DRL技术的颠覆性'设计、开发和部署'。

所展示的库在以下方面具有'elegant'特色：

    - 轻量级：核心代码少于1,000行，例如helloworld。
    - 高效：性能可与Ray RLlib相媲美。
    - 稳定：比Stable Baseline 3更稳定。

ElegantRL支持最先进的DRL算法，包括离散和连续算法，并在Jupyter笔记本中提供用户友好的教程。ElegantRL在Actor-Critic框架下实现DRL算法，其中智能体（也称为DRL算法）由Actor网络和Critic网络组成。由于代码结构的完整性和简单性，用户能够轻松自定义自己的智能体。

更多详细信息请参阅ElegantRL的`GitHub页面 <https://github.com/AI4Finance-Foundation/ElegantRL>`_或`文档 <https://elegantrl.readthedocs.io>`_。
