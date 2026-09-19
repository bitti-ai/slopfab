@echo off
setlocal

rem Build one CUDA 12.8 static-runtime core shared by slopfab.exe and slopfab.dll.
rem cuBLAS is resolved in-process from an installed CUDA 13 or CUDA 12 toolkit;
rem no CUDA DLL is copied into the archive.
for %%I in ("%~dp0.") do set "ROOT=%%~fI"
set "DIST=%ROOT%\dist"
set "BUILD=%ROOT%\build"

set "CUDA12=%CUDA_PATH_V12_8%"
if not defined CUDA12 set "CUDA12=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA13=%CUDA_PATH_V13_0%"
if not defined CUDA13 set "CUDA13=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
if not exist "%CUDA12%\bin\nvcc.exe" (
  echo package: CUDA 12.8 compiler not found; set CUDA_PATH_V12_8
  exit /b 1
)
if not exist "%CUDA13%\include\cublas_v2.h" (
  echo package: CUDA 13.0 headers not found; set CUDA_PATH_V13_0
  exit /b 1
)

rem CMake compares toolset strings literally when reusing an existing build tree.
set "CUDA12=%CUDA12:\=/%"

echo package: configuring CUDA 12.8 core...
cmake -S "%ROOT%" -B "%BUILD%" -G "Visual Studio 17 2022" -A x64 ^
  -T "cuda=%CUDA12%" ^
  "-DCMAKE_CUDA_ARCHITECTURES=86;120a" ^
  "-DSLOPFAB_CUDA13_ROOT=%CUDA13%" ^
  -DSLOPFAB_ENABLE_CUDA=ON ^
  -DSLOPFAB_ENABLE_VULKAN=ON ^
  -DSLOPFAB_WITH_FFMPEG=ON ^
  -DSLOPFAB_BUILD_C_API=ON
if errorlevel 1 exit /b 1
cmake --build "%BUILD%" --config Release --target slopfab slopfab_c
if errorlevel 1 exit /b 1

set "VERSION="
for /f "tokens=5" %%V in ('findstr /b /c:"project(slopfab " "%ROOT%\CMakeLists.txt"') do for /f "delims=)" %%W in ("%%V") do set "VERSION=%%W"
if not defined VERSION (
  echo package: could not read the project version
  exit /b 1
)

set "NAME=slopfab-%VERSION%-windows-x64"
set "STAGE=%DIST%\%NAME%"
set "ZIP=%DIST%\%NAME%.zip"
if not exist "%DIST%" mkdir "%DIST%"
if errorlevel 1 exit /b 1
if exist "%STAGE%" rmdir /s /q "%STAGE%"
if exist "%ZIP%" del /q "%ZIP%"
mkdir "%STAGE%"
if errorlevel 1 exit /b 1

copy /y "%BUILD%\Release\slopfab.exe" "%STAGE%\slopfab.exe" >nul
if errorlevel 1 exit /b 1
copy /y "%BUILD%\Release\slopfab.dll" "%STAGE%\slopfab.dll" >nul
if errorlevel 1 exit /b 1
mkdir "%STAGE%\include\slopfab"
if errorlevel 1 exit /b 1
copy /y "%ROOT%\include\slopfab\capi.h" "%STAGE%\include\slopfab\capi.h" >nul
if errorlevel 1 exit /b 1
mkdir "%STAGE%\lib"
if errorlevel 1 exit /b 1
copy /y "%BUILD%\Release\slopfab_c.lib" "%STAGE%\lib\slopfab_c.lib" >nul
if errorlevel 1 exit /b 1
copy /y "%ROOT%\README.md" "%STAGE%\README.md" >nul
if errorlevel 1 exit /b 1
copy /y "%ROOT%\third_party\sageattention\LICENSE" "%STAGE%\SAGEATTENTION-LICENSE.txt" >nul
if errorlevel 1 exit /b 1
copy /y "%ROOT%\third_party\motioncache\LICENSE" "%STAGE%\MOTIONCACHE-LICENSE.txt" >nul
if errorlevel 1 exit /b 1
copy /y "%ROOT%\external\ffmpeg\LICENSE" "%STAGE%\FFMPEG-LICENSE.txt" >nul
if errorlevel 1 exit /b 1

for %%F in (avcodec-62.dll avformat-62.dll avutil-60.dll swresample-6.dll swscale-9.dll) do (
  if not exist "%ROOT%\external\ffmpeg\bin\%%F" (
    echo package: required FFmpeg runtime not found at %ROOT%\external\ffmpeg\bin\%%F
    exit /b 1
  )
  copy /y "%ROOT%\external\ffmpeg\bin\%%F" "%STAGE%\%%F" >nul
  if errorlevel 1 exit /b 1
)

powershell -NoLogo -NoProfile -NonInteractive -Command ^
  "Compress-Archive -LiteralPath '%STAGE%' -DestinationPath '%ZIP%' -CompressionLevel Optimal"
if errorlevel 1 exit /b 1
rmdir /s /q "%STAGE%"

echo.
echo package: wrote %ZIP%
echo package: CUDA and model weights are not included.
exit /b 0
