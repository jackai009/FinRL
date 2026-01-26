:github_url: https://github.com/AI4Finance-Foundation/FinRL

============================
开发指南
============================

Git是软件工程中常用的工具。PyCharm是流行的Python IDE,开发者也可以选择其他自己喜欢的IDE。现在,我们以PyCharm为例。使用PyCharm进行这种设置可以轻松地在AI4Finance-Foundation的所有存储库上同时工作,同时便于调试、提交到相应的存储库以及创建PR/MR。

步骤1:下载软件
=======

-下载并安装`Anaconda <https://www.anaconda.com/>`_。

-下载并安装`PyCharm <https://www.jetbrains.com/pycharm/>`_。社区版(免费版本)提供了运行Jupyter notebooks之外所需的一切。功能齐全的专业版提供了所有功能。在社区版中运行现有notebooks的解决方法是将所有notebook单元格复制到.py文件中。
对于notebook支持,可以考虑PyCharm专业版。

-在GitHub上,将`FinRL <https://github.com/AI4Finance-Foundation/FinRL>`_fork到你的私有GitHub仓库。

-在GitHub上,将`ElegantRL <https://github.com/AI4Finance-Foundation/ElegantRL>`_fork到你的私有GitHub仓库。

-在GitHub上,将`FinRL-Meta <https://github.com/AI4Finance-Foundation/FinRL-Meta>`_fork到你的私有GitHub仓库。

-接下来的所有步骤都在你的本地计算机上进行。

步骤2:Git Clone
=======

.. code-block:: bash

    mkdir ~/ai4finance
    cd ~/ai4finance
    git clone https://github.com/[你的_github用户名]/FinRL.git
    git clone https://github.com/[你的_github用户名]/ElegantRL.git
    git clone https://github.com/[你的_github用户名]/FinRL-Meta.git


步骤3:创建Conda环境
======

.. code-block:: bash

    cd ~/ai4finance
    conda create --name ai4finance python=3.8
    conda activate ai4finance

    cd FinRL
    pip install -r requirements.txt

使用requirements.txt安装ElegantRL,或者用文本编辑器打开ElegantRL/setup.py,并pip install你能找到的任何东西:gym、matplotlib、numpy、pybullet、torch、opencv-python和box2d-py。


步骤4:配置PyCharm项目
======

-启动PyCharm

-文件 > 打开 > [ai4finance项目文件夹]

.. image:: ../image/pycharm_status_bar.png

-在状态栏的右下角,更改或添加解释器到ai4finance conda环境。确保当你点击左下角的"终端"栏时,它显示ai4finance。

.. image:: ../image/pycharm_MarkDirectoryAsSourcesRoot.png

-在屏幕左侧的项目文件树中:

    -右键单击FinRL文件夹 > 将目录标记为 > 源根目录
    -右键单击ElegantRL文件夹 > 将目录标记为 > 源根目录
    -右键单击FinRL-Meta文件夹 > 将目录标记为 > 源根目录

-一旦你运行.py文件,你会注意到你可能仍然缺少一些包。在这种情况下,只需pip install它们即可。

例如,我们修改FinRL。

.. code-block:: bash

    cd ~/ai4finance
    cd ./FinRL
    git checkout -b branch_xxx

其中branch_xxx是一个新的分支名称。在这个分支中,我们修改config.py。

步骤5:创建新分支
=======

请基于分支"staging"(NOT "master")创建一个新分支,这是给所有开发者的。不要直接推送代码到分支"staging"或"master"。


步骤6:创建提交和PR/MR
=======

-像往常一样通过PyCharm创建提交。

-确保每个提交只覆盖3个存储库中的1个。不要创建跨越多个存储库的提交,例如FinRL和ElegantRL。

.. image:: ../image/pycharm_push_PR.png

-当你执行Git Push时,PyCharm会询问你要推送到3个存储库中的哪一个。就像上图一样,我们选择存储库"FinRL"。


关于创建pull request(PR)或merge request(MR),请参考`创建PR <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_或`开源创建PR <https://opensource.com/article/19/7/create-pull-request-github>`_。

步骤7:提交PR/MR
=======

当提交PR/MR时,请选择分支"staging",NOT "master"。

步骤8:合并"staging"到"master"
=======

此步骤适用于管理员。如果分支"staging"稳定并在一系列测试后成功运行,该仓库的管理员将每隔2-4周将其合并到分支"master"。为避免任何风险,我们希望管理员在合并前在本地下载"master"分支。
