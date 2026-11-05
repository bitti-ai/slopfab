@echo off
setlocal DisableDelayedExpansion
rem Package current builds without compiling. Run package.cmd --help for paths/options.
python "%~dp0tools\package_bundle.py" %*
exit /b %errorlevel%
