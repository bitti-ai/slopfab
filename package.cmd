@echo off
setlocal

rem Build and package the current Release executable. The staging directory is
rem populated from an explicit allow-list, so weights and developer artifacts
rem cannot be included accidentally.

set "ROOT=%~dp0"
set "BUILD=%ROOT%build"
set "EXE=%BUILD%\Release\vidfab.exe"
set "DIST=%ROOT%dist"

echo package: building latest Release configuration...
cmake --build "%BUILD%" --config Release --target vidfab
if errorlevel 1 exit /b 1

if not exist "%EXE%" (
  echo package: Release executable not found at %EXE%
  exit /b 1
)

set "VERSION="
for /f "tokens=5" %%V in ('findstr /b /c:"project(vidfab " "%ROOT%CMakeLists.txt"') do for /f "delims=)" %%W in ("%%V") do set "VERSION=%%W"
if not defined VERSION (
  echo package: could not read the version from %EXE%
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

copy /y "%EXE%" "%STAGE%\vidfab.exe" >nul
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
echo package: weights are not included.
exit /b 0
