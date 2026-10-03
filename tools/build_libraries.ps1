# Build the current checkout's C ABI libraries; build.cmd is the public entry point.
[CmdletBinding()]
param(
    [ValidateSet('both', 'windows', 'linux')][string]$Target = 'both',
    [ValidateRange(1, 64)][int]$Jobs = 2,
    [string]$Distro = '',
    [string]$WindowsBuildDir = '',
    [string]$LinuxBuildDir = '',
    [string]$LinuxCudaCompiler = '',
    [switch]$Help
)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE"
    }
}

function Publish-File([string]$Source, [string]$Destination) {
    $pending = "$Destination.pending-$([Guid]::NewGuid().ToString('N'))"
    try {
        Copy-Item -LiteralPath $Source -Destination $pending
        Move-Item -LiteralPath $pending -Destination $Destination -Force
    } finally {
        if (Test-Path -LiteralPath $pending) {
            Remove-Item -LiteralPath $pending
        }
    }
}

if ($Help) {
    Write-Host @'
Usage: build.cmd [-Target both|windows|linux] [-Jobs N] [-Distro NAME]
                 [-WindowsBuildDir PATH] [-LinuxBuildDir LINUX_PATH]
                 [-LinuxCudaCompiler LINUX_PATH]

Builds Release libraries from the current checkout with CUDA, Vulkan and the
full embedded tokenizer. Does not pull source or download dependencies/models.
Runs C API and tokenizer tests before publishing each platform's outputs.

Windows: dist\windows\slopfab.dll, slopfab_c.lib, capi.h
Linux:   dist\linux\libslopfab.so, versioned copies, capi.h
Both folders also contain SHA256SUMS and build-info.txt.

Defaults: both platforms, 2 jobs, default WSL distribution, Windows build\,
Linux $HOME/slopfab-build-linux. Relative Windows paths are based on the repo.
Linux paths must be absolute. Existing build configurations are reused;
other CMake settings such as FFmpeg are retained (new builds use CMake defaults).

Windows requires CMake >=3.31, VS 2022 C++ tools, CUDA 12.8 and CUDA 13.0 headers.
Linux requires WSL, CMake >=3.31, C++ build tools, CUDA, libpng-dev/libjpeg-dev.
The tokenizer must exist at ref\text_encoder\tokenizer.json.
'@
    exit 0
}

try {
    $tokenizer = Join-Path $root 'ref\text_encoder\tokenizer.json'
    if (-not (Test-Path -LiteralPath $tokenizer -PathType Leaf)) {
        throw "Missing tokenizer: $tokenizer"
    }
    $buildWindows = $Target -in @('both', 'windows')
    $buildLinux = $Target -in @('both', 'linux')

    # Check both platforms before starting a potentially lengthy Windows build.
    if ($buildWindows) {
        Get-Command cmake -ErrorAction Stop | Out-Null
        Get-Command ctest -ErrorAction Stop | Out-Null
        $cuda12 = $env:CUDA_PATH_V12_8
        if (-not $cuda12) { $cuda12 = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA\v12.8' }
        $cuda13 = $env:CUDA_PATH_V13_0
        if (-not $cuda13) { $cuda13 = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA\v13.0' }
        if (-not (Test-Path -LiteralPath (Join-Path $cuda12 'bin\nvcc.exe'))) {
            throw 'CUDA 12.8 compiler not found; set CUDA_PATH_V12_8.'
        }
        if (-not (Test-Path -LiteralPath (Join-Path $cuda13 'include\cublas_v2.h'))) {
            throw 'CUDA 13.0 headers not found; set CUDA_PATH_V13_0.'
        }
        if (-not $WindowsBuildDir) { $WindowsBuildDir = Join-Path $root 'build' }
        elseif (-not [IO.Path]::IsPathRooted($WindowsBuildDir)) {
            $WindowsBuildDir = Join-Path $root $WindowsBuildDir
        }
        $WindowsBuildDir = [IO.Path]::GetFullPath($WindowsBuildDir)
    }
    if ($buildLinux) {
        Get-Command wsl.exe -ErrorAction Stop | Out-Null
        $wslPrefix = @()
        if ($Distro) { $wslPrefix += @('--distribution', $Distro) }
        $linuxRoot = & wsl.exe @wslPrefix --exec wslpath -a -u $root.Replace('\', '/')
        if ($LASTEXITCODE -ne 0) { throw 'Cannot access the checkout in WSL; check -Distro.' }
        $linuxRoot = ($linuxRoot -join "`n").Trim()
        $linuxArguments = $wslPrefix + @('--exec', 'bash', "$linuxRoot/tools/build_linux_library.sh", '--jobs', "$Jobs")
        if ($LinuxBuildDir) { $linuxArguments += @('--build-dir', $LinuxBuildDir) }
        if ($LinuxCudaCompiler) { $linuxArguments += @('--cuda-compiler', $LinuxCudaCompiler) }
        Invoke-Checked 'wsl.exe' ($linuxArguments + '--check')
    }

    if ($buildWindows) {
        Write-Host 'build: configuring Windows CUDA/Vulkan library...'
        Invoke-Checked 'cmake' @('-S', $root, '-B', $WindowsBuildDir,
            '-G', 'Visual Studio 17 2022', '-A', 'x64', '-T', "cuda=$($cuda12.Replace('\', '/'))",
            '-DCMAKE_CUDA_ARCHITECTURES=86;120a', "-DSLOPFAB_CUDA13_ROOT=$cuda13",
            '-DSLOPFAB_ENABLE_CUDA=ON', '-DSLOPFAB_ENABLE_VULKAN=ON',
            '-DSLOPFAB_BUILD_C_API=ON', '-DSLOPFAB_BUILD_TESTS=ON',
            '-DSLOPFAB_EMBED_TOKENIZER=ON', "-DSLOPFAB_TOKENIZER_FILE=$tokenizer")
        Invoke-Checked 'cmake' @('--build', $WindowsBuildDir, '--config', 'Release',
            '--target', 'slopfab_c', 'slopfab_capi_tests', 'slopfab_embedded_tokenizer_tests',
            '--parallel', "$Jobs")
        Invoke-Checked 'ctest' @('--test-dir', $WindowsBuildDir, '-C', 'Release',
            '--output-on-failure', '-R', '^(capi|embedded_tokenizer)$')

        $destination = Join-Path $root 'dist\windows'
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        Publish-File (Join-Path $WindowsBuildDir 'Release\slopfab.dll') (Join-Path $destination 'slopfab.dll')
        Publish-File (Join-Path $WindowsBuildDir 'Release\slopfab_c.lib') (Join-Path $destination 'slopfab_c.lib')
        Publish-File (Join-Path $root 'include\slopfab\capi.h') (Join-Path $destination 'capi.h')
        $hashes = foreach ($name in @('slopfab.dll', 'slopfab_c.lib', 'capi.h')) {
            $digest = (Get-FileHash -LiteralPath (Join-Path $destination $name) -Algorithm SHA256).Hash.ToLowerInvariant()
            "$digest  $name"
        }
        $hashes | Set-Content -LiteralPath (Join-Path $destination 'SHA256SUMS') -Encoding Ascii
        $version = foreach ($part in @('MAJOR', 'MINOR', 'PATCH')) {
            $match = Select-String -LiteralPath (Join-Path $root 'include\slopfab\capi.h') -Pattern "^#define SLOPFAB_CAPI_VERSION_$part\s+(\d+)"
            $match.Matches.Groups[1].Value
        }
        @("slopfab C API $($version -join '.')", 'Windows x64 Release; CUDA 12.8; Vulkan enabled',
            'CUDA targets: sm_86 and sm_120a; full tokenizer embedded',
            "Source: $root", "Build directory: $WindowsBuildDir", "Built UTC: $([DateTime]::UtcNow.ToString('o'))",
            'Tests passed: capi, embedded_tokenizer') |
            Set-Content -LiteralPath (Join-Path $destination 'build-info.txt') -Encoding UTF8
        Write-Host "build: wrote $destination\slopfab.dll"
    }

    if ($buildLinux) {
        Invoke-Checked 'wsl.exe' $linuxArguments
    }
    Write-Host "build: finished. Libraries are under $root\dist"
    exit 0
} catch {
    [Console]::Error.WriteLine("build: $($_.Exception.Message)")
    exit 1
}
