@echo off
setlocal DisableDelayedExpansion
rem Release archives always contain both platforms. Forward build options such as -Distro.
call "%~dp0build.cmd" -Target both %*
if errorlevel 1 exit /b 1
python "%~dp0tools\package_bundle.py"
exit /b %errorlevel%
