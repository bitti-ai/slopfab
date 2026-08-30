@echo off
setlocal

rem Build both toolkit ABIs. The archive contains the CUDA-free launcher and
rem two fat-binary backends, but no CUDA/cuBLAS DLLs; target machines provide
rem those through an installed CUDA 12 or CUDA 13 toolkit.
set "ROOT=%~dp0"
set "DIST=%ROOT%dist"
set "BUILD12=%ROOT%build-cuda12"
set "BUILD13=%ROOT%build-cuda13"

set "CUDA12=%CUDA_PATH_V12_8%"
if not defined CUDA12 set "CUDA12=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA13=%CUDA_PATH_V13_0%"
if not defined CUDA13 set "CUDA13=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.0"

if not exist "%CUDA12%\bin\nvcc.exe" (
  echo package: CUDA 12.8 compiler not found; set CUDA_PATH_V12_8
  exit /b 1
)
if not exist "%CUDA13%\bin\nvcc.exe" (
  echo package: CUDA 13.0 compiler not found; set CUDA_PATH_V13_0
  exit /b 1
)

echo package: configuring CUDA 12...
cmake -S "%ROOT%" -B "%BUILD12%" -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_CUDA_COMPILER="%CUDA12%\bin\nvcc.exe" ^
  "-DCMAKE_CUDA_ARCHITECTURES=86;120a"
if errorlevel 1 exit /b 1
cmake --build "%BUILD12%" --config Release --target vidfab vidfab_cuda_backend
if errorlevel 1 exit /b 1

echo package: configuring CUDA 13...
cmake -S "%ROOT%" -B "%BUILD13%" -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_CUDA_COMPILER="%CUDA13%\bin\nvcc.exe" ^
  "-DCMAKE_CUDA_ARCHITECTURES=86;120a"
if errorlevel 1 exit /b 1
cmake --build "%BUILD13%" --config Release --target vidfab vidfab_cuda_backend
if errorlevel 1 exit /b 1

set "VERSION="
for /f "tokens=5" %%V in ('findstr /b /c:"project(vidfab " "%ROOT%CMakeLists.txt"') do for /f "delims=)" %%W in ("%%V") do set "VERSION=%%W"
if not defined VERSION (
  echo package: could not read the project version
  exit /b 1
)

set "NAME=vidfab-%VERSION%-windows-x64"
set "STAGE=%DIST%\%NAME%"
set "ZIP=%DIST%\%NAME%.zip"
if not exist "%DIST%" mkdir "%DIST%"
if errorlevel 1 exit /b 1
if exist "%STAGE%" rmdir /s /q "%STAGE%"
if exist "%ZIP%" del /q "%ZIP%"
mkdir "%STAGE%"
if errorlevel 1 exit /b 1

copy /y "%BUILD13%\Release\vidfab.exe" "%STAGE%\vidfab.exe" >nul
if errorlevel 1 exit /b 1
copy /y "%BUILD13%\Release\vidfab-cuda13.exe" "%STAGE%\vidfab-cuda13.exe" >nul
if errorlevel 1 exit /b 1
copy /y "%BUILD12%\Release\vidfab-cuda12.exe" "%STAGE%\vidfab-cuda12.exe" >nul
if errorlevel 1 exit /b 1
copy /y "%ROOT%README.md" "%STAGE%\README.md" >nul
if errorlevel 1 exit /b 1
copy /y "%ROOT%external\ffmpeg\LICENSE" "%STAGE%\FFMPEG-LICENSE.txt" >nul
if errorlevel 1 exit /b 1

for %%F in (avcodec-62.dll avformat-62.dll avutil-60.dll swresample-6.dll swscale-9.dll) do (
  if not exist "%ROOT%external\ffmpeg\bin\%%F" (
    echo package: required FFmpeg runtime not found at %ROOT%external\ffmpeg\bin\%%F
    exit /b 1
  )
  copy /y "%ROOT%external\ffmpeg\bin\%%F" "%STAGE%\%%F" >nul
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
