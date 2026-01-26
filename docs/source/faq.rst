:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

=============================
常见问题解答
=============================

:版本: 0.3
:日期: 2022-05-29
:贡献者: Roberto Fray da Silva, Xiao-Yang Liu, Ziyi Xia, Ming Zhu


本文档包含与FinRL相关的最常见问题,这些问题基于在slack频道和Github issues上发布的问题。

.. _Github: https://github.com/AI4Finance-Foundation/FinRL


大纲
==================

    - :ref:`Section-1`

    - :ref:`Section-2`

    - :ref:`Section-3`

    - :ref:`Section-4`

    - :ref:`Section-5`


.. _Section-1:

1-输入和数据集
========================================================================

	.. role:: raw-html(raw)
	   :format: html

    - :raw-html:`<font color="#A52A2A">我可以将FinRL用于加密货币吗？ </font>`

	*还不可以。我们正在开发这个功能*

    - :raw-html:`<font color="#A52A2A">我可以将FinRL用于实时交易吗？  </font>`

	*还不可以。我们正在开发这个功能*

    - :raw-html:`<font color="#A52A2A">我可以将FinRL用于外汇吗？ </font>`

	*还不可以。我们正在开发这个功能*

    - :raw-html:`<font color="#A52A2A">我可以将FinRL用于期货吗？ </font>`

	*还不可以*

    -  :raw-html:`<font color="#A52A2A">免费日数据的最佳数据源是什么？</font>`

	*Yahoo Finance(通过yfinance库)*

    - :raw-html:`<font color="#A52A2A">分钟数据的最佳数据源是什么？ </font>`

	*Yahoo Finance(仅限最近7天),通过yfinance库。这是除了爬取(或付费给服务提供商)之外的唯一选择*

    - :raw-html:`<font color="#A52A2A">FinRL支持杠杆交易吗？ </font>`

	*不支持,因为这更多是与风险控制相关的执行策略。你可以将其作为系统的一部分使用,将风险控制部分作为单独的组件添加*

    - :raw-html:`<font color="#A52A2A">可以添加情感特征来提高模型性能吗？ </font>`

	*可以,你可以添加它。记住检查代码,确保这个额外的特征被输入到模型(状态)中*

    - :raw-html:`<font color="#A52A2A">有没有好的免费的市场情绪来源可以用作特征？  </font>`

	*没有,你必须使用付费服务或库/代码来抓取新闻并从中获得情感(通常,使用深度学习和NLP)*

.. _Section-2:

2-代码和实现
========================================================================

	.. role:: raw-html(raw)
	   :format: html

    - :raw-html:`<font color="#A52A2A">FinRL支持GPU训练吗？  </font>`

	*是的,支持*

    - :raw-html:`<font color="#A52A2A">代码适用于日数据,但在日内频率上给出糟糕的结果。</font>`

	*是的,因为当前参数是为日数据定义的。你必须为日内交易调整模型*

    - :raw-html:`<font color="#A52A2A">有不同的奖励函数可用吗？ </font>`

	*目前还不多,但我们正在努力提供不同的奖励函数和一种简单的方法来设置你自己的奖励函数*

    - :raw-html:`<font color="#A52A2A">我可以使用预训练模型吗？  </font>`

	*可以,但目前还没有。有时在文献中你会发现这被称为迁移学习*

    - :raw-html:`<font color="#A52A2A">在模型上调优的最重要的超参数是什么？  </font>`

	*每个模型都有自己的超参数,但最重要的是total_timesteps(将其视为神经网络中的epochs:即使所有其他超参数都是最优的,如果epochs很少,模型也会表现不佳)。其他重要的超参数,一般来说,是:learning_rate、batch_size、ent_coef、buffer_size、policy和reward scaling*

    - :raw-html:`<font color="#A52A2A">我可以使用哪些库来更好地调优模型？ </font>`

	*有几个,例如:Ray Tune和Optuna。你可以从教程中的我们的示例开始*

    - :raw-html:`<font color="#A52A2A">我可以将哪些DRL算法与FinRL一起使用？  </font>`

	*我们建议使用ElegantRL或Stable Baselines 3。我们成功测试了以下模型:A2C、A3C、DDPG、PPO、SAC、TD3、TRPO。你也可以使用OpenAI Gym风格的市场环境创建自己的算法*

    - :raw-html:`<font color="#A52A2A">模型呈现奇怪的结果或没有训练。   </font>`

	*请更新到最新版本(https://github.com/AI4Finance-LLC/FinRL-Library),检查使用的超参数是否超出正常范围(例如:学习率过高),然后再次运行代码。如果你仍然有问题,请查看第2节(遇到问题时该怎么办)*

    - :raw-html: `<font color="#A52A2A">遇到问题时该怎么办？ </font>`

    *1. 检查这个FAQ是否已经回答了 2. 检查是否在GitHub仓库的*问题*中发布过* `issues <https://github.com/AI4Finance-LLC/FinRL-Library/issues>`_。如果没有,欢迎在GitHub上提交issue 3. 使用AI4Finance slack或微信群上的正确频道。*

    - :raw-html: `<font color="#A52A2A">有人知道是否有用于单只股票的交易环境吗？文档中有一个,但collab链接似乎坏了。 </font>`

        *我们很长时间没有更新单只股票了。单只股票的性能不是很好,因为状态空间太小,以至于代理从环境中提取的信息很少。请使用多股票环境,训练后只使用单只股票进行交易。*


.. _Section-3:

3-模型评估
========================================================================

	.. role:: raw-html(raw)
	   :format: html

    - :raw-html:`<font color="#A52A2A">模型没有用我的数据击败买入并持有(BH)。是模型或代码错了吗？  </font>`

	*不完全是。根据时期、资产、选择的模型和使用的超参数,BH可能很难击败(在低波动率和稳定增长的股票/时期几乎从未被击败)。尽管如此,请更新库及其依赖项(github仓库有最新版本),并查看特定环境类型(单一、多重、投资组合优化)的示例notebook,以查看代码是否正确运行*

    - :raw-html:`<font color="#A52A2A">库中的回测是如何工作的？  </font>`

	*我们使用来自Quantopian的Pyfolio回测库( https://github.com/quantopian/pyfolio ),特别是简单的tear sheet及其图表。一般来说,最重要的指标是:年化回报、累计回报、年化波动率、夏普比率、卡玛比率、稳定性和最大回撤*

    - :raw-html:`<font color="#A52A2A">我应该使用哪些指标来评估模型？  </font>`

	*有几个指标,但我们推荐以下指标,因为它们是市场中最常用的:年化回报、累计回报、年化波动率、夏普比率、卡玛比率、稳定性和最大回撤*

    - :raw-html:`<font color="#A52A2A">我应该使用哪些模型作为比较的基准？  </font>`

	*我们建议使用买入并持有(BH),因为这是可以在任何市场上遵循的策略,并且倾向于在长期内提供良好的结果。你也可以与其他DRL模型和交易策略进行比较,例如最小方差投资组合*

.. _Section-4:

4-其他
========================================================================

	.. role:: raw-html(raw)
	   :format: html

    - :raw-html:`<font color="#A52A2A">我很感兴趣,但我什么都不懂。我应该如何开始？ </font>`

    *1. 从一开始就阅读文档 2. 完成* `tutorials <https://github.com/AI4Finance-Foundation/FinRL/tree/master/tutorials>`_ *3. 阅读我们的论文*

    - :raw-html:`<font color="#A52A2A">库的开发路线图是什么？  </font>`

	*这在我们的GitHub仓库上可用* https://github.com/AI4Finance-LLC/FinRL-Library

    - :raw-html:`<font color="#A52A2A">我如何为开发做出贡献？  </font>`

	*参与slack频道,查看当前问题和路线图,并以任何可能的方式帮助(与他人分享库,为不同市场/模型/策略测试库,为代码开发做出贡献等)*

    - :raw-html:`<font color="#A52A2A">在使用库之前有哪些好的参考资料？  </font>`

	*请阅读* :ref:`Section-1`

    - :raw-html:`<font color="#A52A2A">对于从事金融工作的人来说,有哪些好的RL参考资料？对于从事ML工作的人来说,有哪些好的金融参考资料？ </font>`

	*请阅读* :ref:`Section-4`

    - :raw-html:`<font color="#A52A2A">FinRL将纳入哪些新的SOTA模型？  </font>`

	*请查看我们在GitHub仓库的开发路线图:https://github.com/AI4Finance-LLC/FinRL-Library*

    - :raw-html:`<font color="#A52A2A">FinRL和FinRL-Meta之间的主要区别是什么？  </font>`

	*FinRL旨在教育和演示,而FinRL-Meta旨在构建金融大数据和数据驱动的金融RL的元宇宙。*

.. _Section-5:

5-常见问题/错误
====================================
- 软件包trading_calendars在Windows系统中报告错误:\
    Trading_calendars现在不再维护了。它可能在Windows系统中(python>=3.7)报告错误。有两种可能的解决方案:1). 使用python=3.6环境。2). 用exchange_calendars替换trading_calendars。
