:github_url: https://github.com/AI4Finance-Foundation/FinRL

============================
安装指南
============================

macOS系统
=======

步骤1：安装`Anaconda <https://www.anaconda.com/products/individual>`_
---------------------------------------------------------------------------------------------


-下载`Anaconda安装程序 <https://www.anaconda.com/products/individual#macos>`_，Anaconda包含Python编程所需的一切。

-按照Anaconda的说明：`macOS图形安装 <https://docs.anaconda.com/anaconda/install/mac-os/>`_，安装最新版本的Anaconda。

-打开终端并输入：*'which python'*，应该显示：

.. code-block:: bash

   /Users/your_user_name/opt/anaconda3/bin/python

这意味着您的Python解释器路径已固定到Anaconda的Python版本。如果显示类似以下内容：

.. code-block:: bash

   /Users/your_user_name/opt/anaconda3/bin/python

这意味着您仍在使用默认的Python路径，您需要修复它并固定到Anaconda路径（`尝试这个博客 <https://towardsdatascience.com/how-to-successfully-install-anaconda-on-a-mac-and-actually-get-it-to-work-53ce18025f97>`_），或者您可以使用Anaconda Navigator手动打开终端。

步骤2：安装`Homebrew <https://brew.sh/>`_
---------------------------------------------------------------------

-打开终端并确保已安装Anaconda。

-安装Homebrew：

.. code-block:: bash

   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

步骤3：安装`OpenAI <https://github.com/openai/baselines>`_
-----------------------------------------------------------------

在Mac上安装系统包需要Homebrew。安装Homebrew后，在终端中运行以下命令：

.. code-block:: bash

   brew install cmake openmpi

步骤4：安装`FinRL <https://github.com/AI4Finance-Foundation/FinRL>`_
--------------------------------------------------------------------------

由于我们仍在积极更新FinRL仓库，请使用pip安装不稳定的开发版本FinRL：

.. code-block:: bash

   pip install git+https://github.com/AI4Finance-Foundation/FinRL.git


步骤5：安装box2d（如果使用box2d）
--------------------------------------------------------------------------
用户可以尝试：

.. code-block:: bash

  brew install swig
  pip install box2d-py
  pip install box2d
  pip install Box2D

如果出现错误"AttributeError: module '_Box2D' has no attribute 'RAND_LIMIT_swigconstant'"，用户可以尝试：

.. code-block:: bash

  pip install box2d box2d-kengz


步骤6：运行`FinRL <https://github.com/AI4Finance-Foundation/FinRL>`_
--------------------------------------------------------------------------

下载FinRL-Tutorials仓库，可以使用终端：

.. code-block:: bash

   git clone https://github.com/AI4Finance-Foundation/FinRL-Tutorials.git

或手动下载

.. image:: ../image/download_FinRL.png

通过Anaconda Navigator打开Jupyter Notebook，并在刚刚下载的FinRL-Tutorials中找到股票交易笔记本。您应该能够运行它。


Ubuntu系统
=======

步骤1：安装`Anaconda <https://www.anaconda.com/products/individual>`_
----------------------------------------------------------------------------

请按照此`博客 <https://linuxize.com/post/how-to-install-anaconda-on-ubuntu-18-04/>`_中的步骤操作

步骤2：安装`OpenAI <https://github.com/openai/baselines>`_
----------------------------------------------------------------

打开Ubuntu终端并输入：

.. code-block:: bash

   sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig

步骤3：安装`FinRL <https://github.com/AI4Finance-Foundation/FinRL>`_
--------------------------------------------------------------------------

由于我们仍在积极更新FinRL仓库，请使用pip安装不稳定的开发版本FinRL：

.. code-block:: bash

   pip install git+https://github.com/AI4Finance-Foundation/FinRL.git


步骤4：安装box2d（如果使用box2d）
--------------------------------------------------------------------------



步骤5：运行`FinRL <https://github.com/AI4Finance-Foundation/FinRL>`_
--------------------------------------------------------------------------

在终端中下载FinRL仓库：

.. code-block:: bash

   git clone https://github.com/AI4Finance-Foundation/FinRL.git

在Ubuntu终端中输入'jupyter notebook'打开Jupyter Notebook。

在刚刚下载的FinRL/tutorials中找到股票交易笔记本。您应该能够运行它。

Windows 10系统
======================
安装准备
--------------------------------------------------------------------------
1. 如果在中国使用YahooFinance则需要VPN（pyfolio、elegantRL pip依赖需要拉取代码，YahooFinance已在中国停止服务）。否则请忽略。
2. Python版本 >=3.7
3. pip remove zipline，如果您的系统已安装zipline，zipline与FinRL有冲突。

步骤1：克隆`FinRL <https://github.com/AI4Finance-Foundation/FinRL>`_
--------------------------------------------------------------------------
.. code-block:: bash

   git clone https://github.com/AI4Finance-Foundation/FinRL.git

步骤2：安装依赖
--------------------------------------------------------------------------
.. code-block:: bash

    cd FinRL
    pip install .


步骤3：安装box2d（如果使用box2d）
--------------------------------------------------------------------------


步骤4：测试（如果在中国使用YahooFinance，需要VPN）
-------------------------------------------------------------------------------------
.. code-block:: bash

    python Stock_NeurIPS2018.py

运行错误提示
--------------------------------------------------------------------------

如果出现以下输出，请放心，因为安装仍然成功。

1. UserWarning: Module "zipline.assets" not found; multipliers will not be applied to position notionals. Module "zipline.assets" not found; multipliers will not be applied'


如果出现以下输出，请确保VPN有助于访问YahooFinance

1. Failed download: xxxx: No data found for this date range, the stock may be delisted, or the value is missing.


Windows 10（wsl安装）
=========================

步骤1：在Windows 10上安装Ubuntu
--------------------------------------
请查看此视频了解详细步骤：

.. raw:: html

   <iframe width="692" height="389" src="https://www.youtube.com/embed/X-DHaQLrBi8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

步骤2：安装`Anaconda <https://www.anaconda.com/products/individual>`_
----------------------------------------------------------------------------

请按照此`博客 <https://linuxize.com/post/how-to-install-anaconda-on-ubuntu-18-04/>`_中的步骤操作

步骤3：安装`OpenAI <https://github.com/openai/baselines>`_
----------------------------------------------------------------

打开Ubuntu终端并输入：

.. code-block:: bash

   sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig

步骤4：安装`FinRL <https://github.com/AI4Finance-Foundation/FinRL>`_
--------------------------------------------------------------------------

由于我们仍在积极更新FinRL仓库，请使用pip安装不稳定的开发版本FinRL：

.. code-block:: bash

   pip install git+https://github.com/AI4Finance-Foundation/FinRL.git


步骤5：安装box2d（如果使用box2d）
--------------------------------------------------------------------------

步骤6：运行`FinRL <https://github.com/AI4Finance-Foundation/FinRL>`_
--------------------------------------------------------------------------

在终端中下载FinRL-Tutorials仓库：

.. code-block:: bash

   git clone https://github.com/AI4Finance-Foundation/FinRL-Tutorials.git

在Ubuntu终端中输入'jupyter notebook'打开Jupyter Notebook。请参阅`jupyter notebook <https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html>`_

在刚刚下载的FinRL-Tutorials中找到股票交易笔记本。您应该能够运行它。
