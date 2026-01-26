@ECHO OFF

pushd %~dp0

REM Sphinx文档的命令文件。使用.\make.bat html运行。

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.'sphinx-build'命令未找到。请确保已安装Sphinx
	echo.，然后设置SPHINXBUILD环境变量指向
	echo.'sphinx-build'可执行文件的完整路径。或者您
	echo.可以将Sphinx目录添加到PATH。
	echo.
	echo.如果您没有安装Sphinx，请从
	echo.http://sphinx-doc.org/获取
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
popd
