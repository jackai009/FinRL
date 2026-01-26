#
# Sphinx文档构建器的配置文件
#
# 此文件仅包含最常见选项的子集。完整列表请参阅文档：
# http://www.sphinx-doc.org/en/master/config
# -- 路径设置 --------------------------------------------------------------
# 如果扩展（或使用autodoc记录的模块）在另一个目录中，
# 请在此处将这些目录添加到sys.path。如果目录相对于文档根目录，
# 请使用os.path.abspath使其成为绝对路径，如下所示。
#
from __future__ import annotations

import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath(os.path.join("../..", "finrl")))  # 重要

# -- 项目信息 -----------------------------------------------------

project = "FinRL"
copyright = "2021, FinRL"
author = "FinRL"

# 短版本X.Y
version = ""
# 完整版本，包括alpha/beta/rc标签
release = "0.3.1"


# -- 常规配置 ---------------------------------------------------

# 如果您的文档需要特定的Sphinx最低版本，请在此处说明
#
# needs_sphinx = '1.0'

# 在此处添加任何Sphinx扩展模块名称，作为字符串。它们可以是
# 来自Sphinx的扩展（命名为'sphinx.ext.*'）或您的自定义扩展。
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosectionlabel",
    "recommonmark",  # 用于包含markdown
    #     'sphinx_markdown_tables'  # 支持在markdown中渲染表格
]

autodoc_mock_imports = [
    "gym",
    "matplotlib",
    "numpy",
    "pybullet",
    "torch",
    "opencv-python",
]

pygments_style = "sphinx"


# 在此处添加包含模板的任何路径，相对于此目录。
templates_path = ["_templates"]

# 源文件名的后缀。
# 您可以将多个后缀指定为字符串列表：
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# 主文档。
master_doc = "index"

# Sphinx自动生成内容的语言。请参阅文档以获取
# 支持的语言列表。
#
# 如果您通过gettext目录进行内容翻译，这也被使用。
# 通常在这些情况下，您从命令行设置"language"。
language = None

# 要在查找源文件时匹配的文件和目录的模式列表，相对于源目录。
# 此模式也影响html_static_path和html_extra_path。
exclude_patterns = []

# 要使用的Pygments（语法高亮）样式的名称。
pygments_style = None


# -- HTML输出选项 -------------------------------------------------

# 用于HTML和HTML帮助页面的主题。请参阅文档以获取
# 内置主题列表。
#

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_logo = "./image/logo_transparent_background.png"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}


# 主题选项是特定于主题的，可以进一步自定义外观和感觉。
# 有关每个主题可用选项的列表，请参阅文档。
#
# html_theme_options = {}

# 在此处添加包含自定义静态文件（如样式表）的任何路径，相对于此目录。
# 它们在内置静态文件之后被复制，因此名为"default.css"的文件将覆盖内置的"default.css"。
html_static_path = ["_static"]

# 自定义侧边栏模板，必须是映射文档名称到模板名称的字典。
#
# 不匹配任何模式的文档的默认侧边栏由主题本身定义。
# 内置主题默认使用以下模板：``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``。
#
# html_sidebars = {}


# -- HTMLHelp输出选项 ---------------------------------------------

# HTML帮助构建器的输出文件基本名称。
htmlhelp_basename = "FinRLdoc"


# -- LaTeX输出选项 ------------------------------------------------

latex_elements = {
    # 纸张大小（'letterpaper'或'a4paper'）。
    #
    # 'papersize': 'letterpaper',
    # 字体大小（'10pt'、'11pt'或'12pt'）。
    #
    # 'pointsize': '10pt',
    # LaTeX前言的额外内容。
    #
    # 'preamble': '',
    # LaTeX图形（浮动）对齐
    #
    # 'figure_align': 'htbp',
}

# 将文档树分组为LaTeX文件。元组列表
# （源开始文件，目标名称，标题，作者，文档类[howto、manual或自己的类]）。
latex_documents = [
    (master_doc, "FinRL.tex", "FinRL文档", "FinRL", "manual"),
]


# -- 手册页输出选项 ------------------------------------------

# 每个手册页一个条目。元组列表
# （源开始文件，名称，描述，作者，手册节）。
man_pages = [(master_doc, "finrl", "FinRL文档", [author], 1)]


# -- Texinfo输出选项 ----------------------------------------------

# 将文档树分组为Texinfo文件。元组列表
# （源开始文件，目标名称，标题，作者，目录菜单条目，描述，类别）。
texinfo_documents = [
    (
        master_doc,
        "FinRL",
        "FinRL文档",
        author,
        "FinRL",
        "项目的一行描述。",
        "其他",
    ),
]


# -- Epub输出选项 -------------------------------------------------

# 书目都柏林核心信息。
epub_title = project

# 文本的唯一标识符。这可以是ISBN号
# 或项目主页。
#
# epub_identifier = ''

# 文本的唯一标识符。
#
# epub_uid = ''

# 不应打包到epub文件中的文件列表。
epub_exclude_files = ["search.html"]


# -- 扩展配置 -------------------------------------------------
