# Sphinx文档的最小Makefile
# 使用make html运行。
# 您可以从命令行设置这些变量。
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
# 首先放置它，这样不带参数的"make"就像"make help"一样。
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
.PHONY: help Makefile
# 捕获所有目标：使用新的"make mode"选项将所有未知目标路由到Sphinx。
# $(O)意作为$(SPHINXOPTS)的快捷方式。
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
