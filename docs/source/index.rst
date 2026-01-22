.. Finrl Library documentation master file, created by
   sphinx-quickstart on Wed Nov 18 08:14:32 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/AI4Finance-Foundation/FinRL

欢迎来到FinRL库！
=====================================================================================================

.. meta::
   :description: FinRL is the first open source framework for financial reinforcement learning. It facilitates beginners to expose themselves to quantitative finance and to develop stock trading strategies using deep reinforcement learning. It provides fine-tuned deep reinforcement learning algorithms, including DQN, DDPG, PPO, SAC, A2C, TD3, etc.
   :keywords: finance AI, OpenAI, artificial intelligence in finance, machine learning, deep reinforcement learning, DRL, RL, neural networks, deep q network, multi agent reinforcement learning

.. image:: image/logo_transparent_background.png
   :target:  https://github.com/AI4Finance-Foundation/FinRL

**免责声明：本文中的任何内容都不构成财务建议，也不是交易真钱的推荐。请使用常识，并在交易或投资前始终首先咨询专业人士。**

**AI4Finance** 社区提供这个演示性和教育资源，以便有效地自动化交易。FinRL是第一个用于金融强化学习的开源框架。

.. _FinRL: https://github.com/AI4Finance-Foundation/FinRL

强化学习（RL）通过试错训练智能体解决问题，而DRL使用深度神经网络作为函数逼近器。DRL平衡了探索（未知领域）和利用（当前知识），并被认为是自动化交易的竞争优势。DRL框架通过与未知环境的交互学习来解决动态决策问题，从而展现出两大优势：投资组合可扩展性和市场模型独立性。自动化交易本质上是做出动态决策，即在高度随机和复杂的股票市场上**决定在哪里交易、以什么价格交易以及交易什么数量**。考虑到许多复杂的金融因素，DRL交易智能体构建多因子模型并提供算法交易策略，这对人类交易员来说是困难的。

`FinRL`_ 提供了一个支持各种市场、SOTA DRL算法、许多量化金融任务基准、实时交易等的框架。

.. _FinRL: https://github.com/AI4Finance-Foundation/FinRL

加入或与我们讨论FinRL：`AI4Finance邮件列表 <https://groups.google.com/u/1/g/ai4finance>`_。

欢迎随时给我们反馈：使用`Github问题 <https://github.com/AI4Finance-LLC/FinRL-Library/issues>`_ 报告错误，或在Slack频道讨论FinRL开发。

.. _Github issues: https://github.com/AI4Finance-LLC/FinRL-Library/issues

.. image:: image/join_slack.png
   :target: https://join.slack.com/t/ai4financeworkspace/shared_invite/zt-jyaottie-hHqU6TdvuhMHHAMXaLw_~w
   :width: 400
   :align: center

|

.. toctree::
   :maxdepth: 1
   :hidden:

   Home <self>


.. toctree::
   :maxdepth: 1
   :caption: 入门指南

   start/introduction
   start/first_glance
   start/three_layer
   start/installation
   start/quick_start

   start/introduction
   start/first_glance
   start/three_layer
   start/installation
   start/quick_start


.. toctree::
   :maxdepth: 1
   :caption: FinRL-Meta

   finrl_meta/background
   finrl_meta/overview
   finrl_meta/Data_layer
   finrl_meta/Environment_layer
   finrl_meta/Benchmark


.. toctree::
   :maxdepth: 3
   :caption: 教程

   tutorial/Guide
   tutorial/Homegrown_example
   tutorial/1-Introduction
   tutorial/2-Advance
   tutorial/3-Practical
   tutorial/4-Optimization
   tutorial/5-Others


.. toctree::
   :maxdepth: 1
   :caption: 开发者指南

   developer_guide/file_architecture
   developer_guide/development_setup
   developer_guide/contributing


.. toctree::
   :maxdepth: 1
   :caption: 参考

   reference/publication
   reference/reference.md


.. toctree::
   :maxdepth: 2
   :caption: 常见问题

   faq
