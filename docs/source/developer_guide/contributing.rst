:github_url: https://github.com/AI4Finance-Foundation/FinRL

============================
贡献指南
============================



本项目旨在为交易社区提供强化学习环境。
社区中始终存在竞争性的优先事项,我们要确保能够共同实现一个可靠、可持续和可维护的项目。

指导原则
=======

* 我们应该在这个项目中有可靠的代码
    * 带有测试的可靠代码
    * 能够正常工作的可靠代码
    * 不会消耗过多资源的可靠代码
* 我们应该相互帮助,共同实现最先进的结果
* 我们应该编写清晰的代码
    * 代码不应冗余
    * 代码应包含内联文档(标准PEP格式)
    * 代码应组织为类和函数
* 我们应该合理地利用外部工具
* 我们共同工作,在交流中友善、耐心和清晰。不受欢迎粗鲁的人。

## 如果发现问题,请说出来!
* 提交[issue](https://guides.github.com/features/issues/)是帮助改进项目的好方法


接受PR
=======

* 你发现了一个bug以及修复它的方法
* 你为该项目协调员优先考虑的问题做出了贡献
* 你要添加新功能,并且已经为此编写了issue,并且有文档+测试

PR指南
=======

* 请在每个PR中标记@bruceyang、@spencerromo或@xiaoyang。(P.S.我们正在寻找更多有软件经验的合作者!)
* 请引用或编写并引用一个[issue](https://guides.github.com/features/issues/)
* 请使用清晰的提交消息
* 请为每个添加的功能编写详细的文档和测试
* 请尽量不要破坏现有功能,或者如果需要,请计划证明这种必要性并与合作者协调
* 请对反馈保持耐心和尊重
* 请使用pre-commit钩子


其他
=======

-使用pre-commit
```
pip install pre-commit
pre-commit install
```

-运行测试
```
-本地
python3 -m unittest discover

-Docker
./docker/bin/build_container.sh
./docker/bin/test.sh
```
