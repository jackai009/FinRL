:github_url: https://github.com/AI4Finance-Foundation/FinRL

===========================
三层架构
===========================

在初步了解如何使用DRL建立股票交易任务后，现在我们将介绍FinRL最核心的理念。

FinRL库由三层组成：**市场环境（FinRL-Meta）**、**DRL智能体**和**应用**。下层为上层提供API，使下层对上层透明。智能体层以探索-利用的方式与环境层交互，无论是重复先前表现良好的决策还是采取新行动以期望获得更大的累积奖励。


.. image:: ../image/finrl_framework.png
   :width: 80%
   :align: center

我们的构建具有以下优势：

**模块化**：每层包含多个模块，每个模块定义独立的功能。用户可以从某层选择特定模块来实现其股票交易任务。此外，可以更新现有模块。

**简洁性、适用性和可扩展性**：专为自动化股票交易设计，FinRL将DRL算法呈现为模块。通过这种方式，FinRL变得易于使用且不过于苛刻。FinRL提供三个交易任务作为用例，可以轻松复现。每层包含预留接口，允许用户开发新模块。

**更好的市场环境建模**：我们构建了一个复制实时股票市场的交易模拟器，并提供回测支持，包含重要的市场摩擦，如交易成本、市场流动性和投资者的风险规避程度。所有这些因素都是净回报关键决定因素中的重要组成部分。

FinRL如何在DRL中构建问题的高级视图：

.. image:: ../image/finrl_overview_drl.png
   :width: 80%
   :align: center

请参阅以下页面获取更详细的解释：

.. toctree::
   :maxdepth: 1

   three_layer/environments
   three_layer/agents
   three_layer/applications
