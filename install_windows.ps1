param(
    [string]$ComfyDir,
    [string]$PythonLauncher = 'py -3.13',
    [string]$TorchIndexUrl = 'https://download.pytorch.org/whl/cu128',
    [switch]$SkipModelDownloads,
    [switch]$DownloadModels,
    [switch]$SkipDeepExemplar,
    [switch]$SkipComfyManager,
    [switch]$InstallCorrelationExtension,
    [switch]$NonInteractive
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$DefaultDownloadCache = Join-Path $Root '.cache\huggingface'
$DownloadCache = if (-not [string]::IsNullOrWhiteSpace($env:ARP_HF_CACHE_DIR)) {
    $env:ARP_HF_CACHE_DIR
} elseif (-not [string]::IsNullOrWhiteSpace($env:HF_HUB_CACHE)) {
    $env:HF_HUB_CACHE
} elseif (-not [string]::IsNullOrWhiteSpace($env:HF_HOME)) {
    Join-Path $env:HF_HOME 'hub'
} else {
    $DefaultDownloadCache
}
$DefaultComfyDir = Join-Path $Root 'tools\comfyui'
$PipelinePython = Join-Path $Root '.venv\Scripts\python.exe'
$BundledCustomNodes = Join-Path $Root 'vendor\comfyui_custom_nodes'
$UseExistingComfy = $false
$ComfyManagedByArp = $true
$ResolvedPythonLauncher = $null
$PythonLauncherFailures = @()

function Get-ArpVersion {
    $versionPath = Join-Path $Root 'VERSION'
    if (Test-Path -LiteralPath $versionPath) {
        return (Get-Content -LiteralPath $versionPath -TotalCount 1).Trim()
    }
    return '0.0.0'
}

function Get-ArpCommitHash {
    try {
        $commit = (& git -C $Root rev-parse --short HEAD 2>$null | Select-Object -First 1)
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($commit)) {
            return $commit.Trim()
        }
    } catch {
    }
    $commit = Get-ArpCommitHashFromGitDir
    if (-not [string]::IsNullOrWhiteSpace($commit)) {
        return $commit
    }
    return 'unknown'
}

function Get-ArpCommitHashFromGitDir {
    $gitPath = Join-Path $Root '.git'
    if (-not (Test-Path -LiteralPath $gitPath)) {
        return $null
    }

    $gitDir = $gitPath
    if (-not (Test-Path -LiteralPath $gitPath -PathType Container)) {
        $gitFile = Get-Content -LiteralPath $gitPath -TotalCount 1 -ErrorAction SilentlyContinue
        if ($gitFile -notmatch '^gitdir:\s*(.+)$') {
            return $null
        }
        $gitDir = $Matches[1]
        if (-not [System.IO.Path]::IsPathRooted($gitDir)) {
            $gitDir = Join-Path $Root $gitDir
        }
    }

    $headPath = Join-Path $gitDir 'HEAD'
    if (-not (Test-Path -LiteralPath $headPath)) {
        return $null
    }
    $head = (Get-Content -LiteralPath $headPath -TotalCount 1 -ErrorAction SilentlyContinue).Trim()
    if ($head -match '^ref:\s*(.+)$') {
        $refName = $Matches[1]
        $refPath = Join-Path $gitDir $refName
        if (-not (Test-Path -LiteralPath $refPath)) {
            $packedRefs = Join-Path $gitDir 'packed-refs'
            if (-not (Test-Path -LiteralPath $packedRefs)) {
                return $null
            }
            $packedRefLine = Get-Content -LiteralPath $packedRefs -ErrorAction SilentlyContinue |
                Where-Object { $_ -match "^[0-9a-fA-F]{40}\s+$([regex]::Escape($refName))$" } |
                Select-Object -First 1
            if (-not $packedRefLine) {
                return $null
            }
            $head = ($packedRefLine -split '\s+')[0]
        } else {
            $head = (Get-Content -LiteralPath $refPath -TotalCount 1 -ErrorAction SilentlyContinue).Trim()
        }
    }
    if ($head -match '^[0-9a-fA-F]{7,40}$') {
        return $head.Substring(0, 7)
    }
    return $null
}

function Write-ArpBanner {
    $version = Get-ArpVersion
    $commit = Get-ArpCommitHash
    if ([string]::IsNullOrWhiteSpace($commit) -or $commit -eq 'unknown') {
        Write-Host "ARP $version"
    } else {
        Write-Host "ARP $version-$commit"
    }
}

function Invoke-Step {
    param([string]$Label, [scriptblock]$Block)
    Write-Host "`n==> $Label" -ForegroundColor Cyan
    & $Block
}

function ConvertTo-NativeArgument {
    param([AllowEmptyString()][string]$Argument)
    if ([string]::IsNullOrEmpty($Argument)) {
        return '""'
    }
    if ($Argument -notmatch '[\s"]') {
        return $Argument
    }

    # Start-Process joins ArgumentList entries with spaces. Quote using the Windows
    # command-line escaping rules so paths such as "E:\AI Remaster\..." stay intact.
    $escaped = [regex]::Replace($Argument, '(\\*)"', {
        param($match)
        return ($match.Groups[1].Value * 2) + '\"'
    })
    $trailingBackslashes = [regex]::Match($escaped, '\\+$')
    if ($trailingBackslashes.Success) {
        $escaped += $trailingBackslashes.Value
    }
    return '"' + $escaped + '"'
}

function Join-NativeArguments {
    param([string[]]$Arguments)
    return (($Arguments | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' ')
}

function Invoke-External {
    param([string[]]$Command, [string]$WorkingDirectory = $Root)
    if (-not $Command -or $Command.Count -eq 0) {
        throw 'No command was provided.'
    }
    Write-Host ($Command -join ' ')
    $executable = Resolve-CommandExecutable $Command[0] ($Command -join ' ')
    $startArgs = @{
        FilePath = $executable
        WorkingDirectory = $WorkingDirectory
        NoNewWindow = $true
        Wait = $true
        PassThru = $true
    }
    if ($Command.Count -gt 1) {
        $startArgs.ArgumentList = Join-NativeArguments @($Command[1..($Command.Count - 1)])
    }
    $process = Start-Process @startArgs
    if ($process.ExitCode -ne 0) {
        throw "Command failed with exit code $($process.ExitCode): $($Command -join ' ')"
    }
}

function Invoke-External-Optional {
    param([string[]]$Command, [string]$WorkingDirectory = $Root)
    if (-not $Command -or $Command.Count -eq 0) {
        throw 'No command was provided.'
    }
    Write-Host ($Command -join ' ')
    $executable = Resolve-CommandExecutable $Command[0] ($Command -join ' ')
    $startArgs = @{
        FilePath = $executable
        WorkingDirectory = $WorkingDirectory
        NoNewWindow = $true
        Wait = $true
        PassThru = $true
    }
    if ($Command.Count -gt 1) {
        $startArgs.ArgumentList = Join-NativeArguments @($Command[1..($Command.Count - 1)])
    }
    $process = Start-Process @startArgs
    return $process.ExitCode -eq 0
}

function Resolve-CommandExecutable {
    param([string]$FilePath, [string]$DisplayCommand = $FilePath)
    if ([string]::IsNullOrWhiteSpace($FilePath)) {
        throw "Command has an empty executable: $DisplayCommand"
    }
    if ([System.IO.Path]::IsPathRooted($FilePath) -or $FilePath.Contains('\') -or $FilePath.Contains('/')) {
        if (Test-Path -LiteralPath $FilePath) {
            return $FilePath
        }
    } else {
        $resolved = Get-Command $FilePath -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($resolved) {
            return $resolved.Source
        }
    }
    throw "Could not find executable '$FilePath' while running: $DisplayCommand"
}

function Split-CommandLine {
    param([string]$CommandLine)
    $matches = [regex]::Matches($CommandLine, '("[^"]+"|''[^'']+''|\S+)')
    $parts = @()
    foreach ($match in $matches) {
        $part = $match.Value
        if (($part.StartsWith('"') -and $part.EndsWith('"')) -or ($part.StartsWith("'") -and $part.EndsWith("'"))) {
            $part = $part.Substring(1, $part.Length - 2)
        }
        $parts += $part
    }
    return $parts
}

function Convert-PythonLauncherArgument {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return @()
    }

    $trimmed = $Value.Trim()
    if (($trimmed.StartsWith('"') -and $trimmed.EndsWith('"')) -or ($trimmed.StartsWith("'") -and $trimmed.EndsWith("'"))) {
        $trimmed = $trimmed.Substring(1, $trimmed.Length - 2)
    }

    $isPathLike = [System.IO.Path]::IsPathRooted($trimmed) -or $trimmed.Contains('\') -or $trimmed.Contains('/')
    if ($isPathLike) {
        return @($trimmed)
    }

    return Split-CommandLine $trimmed
}

function Get-CommandTail {
    param([string[]]$Command)
    if ($Command.Count -le 1) {
        return @()
    }
    return @($Command[1..($Command.Count - 1)])
}

function Invoke-CapturedProcess {
    param([string]$FilePath, [string[]]$Arguments = @(), [string]$WorkingDirectory = $Root)
    $stdout = [System.IO.Path]::GetTempFileName()
    $stderr = [System.IO.Path]::GetTempFileName()
    try {
        $startArgs = @{
            FilePath = $FilePath
            WorkingDirectory = $WorkingDirectory
            NoNewWindow = $true
            Wait = $true
            PassThru = $true
            RedirectStandardOutput = $stdout
            RedirectStandardError = $stderr
        }
        if ($Arguments.Count -gt 0) {
            $startArgs.ArgumentList = Join-NativeArguments $Arguments
        }
        $process = Start-Process @startArgs
        return [pscustomobject]@{
            ExitCode = $process.ExitCode
            Stdout = (Get-Content -LiteralPath $stdout -Raw -ErrorAction SilentlyContinue)
            Stderr = (Get-Content -LiteralPath $stderr -Raw -ErrorAction SilentlyContinue)
        }
    } finally {
        Remove-Item -LiteralPath $stdout, $stderr -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-PythonVersionProbe {
    param([string[]]$Command)
    $display = $Command -join ' '
    $executable = Resolve-CommandExecutable $Command[0] $display
    $probeScript = Join-Path $Root 'scripts\python_version_probe.py'
    $arguments = (Get-CommandTail $Command) + @($probeScript)
    return Invoke-CapturedProcess -FilePath $executable -Arguments $arguments
}

function Add-PythonLauncherCandidate {
    param(
        [System.Collections.ArrayList]$Candidates,
        [string[]]$Command
    )
    if (-not $Command -or $Command.Count -eq 0) {
        return
    }
    $display = $Command -join "`0"
    foreach ($candidate in $Candidates) {
        if (($candidate -join "`0") -eq $display) {
            return
        }
    }
    [void]$Candidates.Add([string[]]$Command)
}

function Expand-PythonLauncherCandidates {
    param([string[]]$Requested)
    $candidates = [System.Collections.ArrayList]::new()
    Add-PythonLauncherCandidate $candidates $Requested

    if ($Requested.Count -eq 1) {
        $requestedPath = $Requested[0]
        if ([System.IO.Path]::IsPathRooted($requestedPath) -or $requestedPath.Contains('\') -or $requestedPath.Contains('/')) {
            if (Test-Path -LiteralPath $requestedPath -PathType Container) {
                Add-PythonLauncherCandidate $candidates @((Join-Path $requestedPath 'python.exe'))
                Add-PythonLauncherCandidate $candidates @((Join-Path $requestedPath 'Python313.exe'))
            } else {
                $parent = Split-Path -Path $requestedPath -Parent
                $leaf = Split-Path -Path $requestedPath -Leaf
                if (-not [string]::IsNullOrWhiteSpace($parent)) {
                    if ($leaf -ieq 'python.exe') {
                        $parentLeaf = Split-Path -Path $parent -Leaf
                        if ($parentLeaf -ieq 'Python313') {
                            $grandparent = Split-Path -Path $parent -Parent
                            if (-not [string]::IsNullOrWhiteSpace($grandparent)) {
                                Add-PythonLauncherCandidate $candidates @((Join-Path $grandparent 'Python313.exe'))
                            }
                        } else {
                            Add-PythonLauncherCandidate $candidates @((Join-Path $parent 'Python313.exe'))
                            Add-PythonLauncherCandidate $candidates @((Join-Path (Join-Path $parent 'Python313') 'python.exe'))
                        }
                    } elseif ($leaf -ieq 'Python313.exe') {
                        Add-PythonLauncherCandidate $candidates @((Join-Path (Join-Path $parent 'Python313') 'python.exe'))
                    }
                }
            }
        }
    }

    return ,$candidates
}

function Get-PythonLauncherCheck {
    param([string[]]$Command)
    $display = $Command -join ' '
    try {
        $probe = Invoke-PythonVersionProbe -Command $Command
    } catch {
        return [pscustomobject]@{
            Success = $false
            Reason = $_.Exception.Message
        }
    }
    if ($probe.ExitCode -ne 0) {
        $detail = (($probe.Stdout, $probe.Stderr) -join "`n").Trim()
        if ([string]::IsNullOrWhiteSpace($detail)) {
            $detail = "exited with code $($probe.ExitCode)"
        }
        return [pscustomobject]@{
            Success = $false
            Reason = $detail
        }
    }
    $version = (($probe.Stdout, $probe.Stderr) -join "`n").Split("`n") |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -match '^\d+\.\d+$' } |
        Select-Object -First 1
    if ($version -ne '3.13') {
        $found = if ([string]::IsNullOrWhiteSpace($version)) { 'unknown version' } else { "Python $version" }
        return [pscustomobject]@{
            Success = $false
            Reason = "found $found"
        }
    }
    $implementation = (($probe.Stdout, $probe.Stderr) -join "`n").Split("`n") |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -match '^implementation=' } |
        ForEach-Object { $_.Substring('implementation='.Length) } |
        Select-Object -First 1
    if ($implementation -and $implementation -ne 'cpython') {
        return [pscustomobject]@{
            Success = $false
            Reason = "found $implementation, but CPython is required"
        }
    }
    $bits = (($probe.Stdout, $probe.Stderr) -join "`n").Split("`n") |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -match '^bits=' } |
        ForEach-Object { $_.Substring('bits='.Length) } |
        Select-Object -First 1
    if ($bits -and $bits -ne '64') {
        return [pscustomobject]@{
            Success = $false
            Reason = "found $bits-bit Python, but 64-bit Python is required for PyTorch"
        }
    }
    $gilDisabled = (($probe.Stdout, $probe.Stderr) -join "`n").Split("`n") |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -match '^gil_disabled=' } |
        ForEach-Object { $_.Substring('gil_disabled='.Length) } |
        Select-Object -First 1
    if ($gilDisabled -eq '1') {
        return [pscustomobject]@{
            Success = $false
            Reason = 'found free-threaded Python, but standard CPython is required for PyTorch wheels'
        }
    }
    return [pscustomobject]@{
        Success = $true
        Reason = 'found 64-bit CPython 3.13'
    }
}

function Find-PythonLauncher {
    param([array]$Candidates)
    $script:PythonLauncherFailures = @()
    foreach ($candidate in $candidates) {
        $check = Get-PythonLauncherCheck $candidate
        if ($check.Success) {
            Write-Host "Using Python launcher: $($candidate -join ' ')"
            return $candidate
        }
        $script:PythonLauncherFailures += "  $($candidate -join ' '): $($check.Reason)"
    }
    return $null
}

function Update-ProcessPathFromRegistry {
    $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $paths = @($machinePath, $userPath) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    if ($paths) {
        $env:Path = ($paths -join ';')
    }
}

function Show-PythonInstallPrompt {
    param([array]$Candidates)
    $candidateText = (($Candidates | ForEach-Object { $_ -join ' ' }) -join ', ')
    Write-Host ''
    Write-Host 'Python 3.13 is required before ARP can create its virtual environment.' -ForegroundColor Yellow
    Write-Host "The installer looked for Python 3.13 with: $candidateText"
    if ($script:PythonLauncherFailures.Count -gt 0) {
        Write-Host 'Detection details:' -ForegroundColor Yellow
        foreach ($failure in $script:PythonLauncherFailures) {
            Write-Host $failure -ForegroundColor Yellow
        }
    }
    Write-Host 'Install Python 3.13 from https://www.python.org/downloads/'
    Write-Host 'On Windows, enable the Python Launcher option or add python.exe to PATH.'
    Write-Host 'After installing Python 3.13, press R to retry. Press Q to quit.'
    Write-Host 'If Python 3.13 is already installed somewhere custom, rerun with: install_windows.bat -PythonLauncher C:\Path\To\python.exe'
    while ($true) {
        $answer = Read-Host 'Retry Python detection? (R/Q)'
        if ($answer -ieq 'R') {
            Update-ProcessPathFromRegistry
            return
        }
        if ($answer -ieq 'Q') {
            throw 'Install cancelled because Python 3.13 was not found.'
        }
        Write-Host 'Please enter R to retry or Q to quit.' -ForegroundColor Yellow
    }
}

function Resolve-PythonLauncher {
    $requested = Convert-PythonLauncherArgument $PythonLauncher
    if (-not $requested -or $requested.Count -eq 0) {
        throw 'Python launcher command is empty.'
    }

    $candidates = Expand-PythonLauncherCandidates $requested
    if (($requested -join ' ') -eq 'py -3.13') {
        Add-PythonLauncherCandidate $candidates @('python3.13')
        Add-PythonLauncherCandidate $candidates @('python')
    }

    while ($true) {
        $launcher = Find-PythonLauncher $candidates
        if ($launcher) {
            return $launcher
        }
        if ($NonInteractive) {
            throw "Could not find Python 3.13. Install Python 3.13 from https://www.python.org/downloads/ with the Python Launcher option enabled, or rerun with -PythonLauncher pointing at a Python 3.13 executable."
        }
        Show-PythonInstallPrompt $candidates
    }
}

function Invoke-PythonLauncher {
    param([string[]]$Arguments, [string]$WorkingDirectory = $Root)
    if (-not $script:ResolvedPythonLauncher) {
        $script:ResolvedPythonLauncher = Resolve-PythonLauncher
    }
    Invoke-External -Command (@($script:ResolvedPythonLauncher) + $Arguments) -WorkingDirectory $WorkingDirectory
}

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Test-ComfyDir {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    return (Test-Path -LiteralPath (Join-Path $Path 'main.py'))
}

function Resolve-ComfyDirPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }

    $full = [System.IO.Path]::GetFullPath($Path)
    if (Test-ComfyDir $full) {
        return $full
    }

    foreach ($child in @('ComfyUI', 'comfyui')) {
        $nested = Join-Path $full $child
        if (Test-ComfyDir $nested) {
            Write-Host "Found ComfyUI checkout inside selected folder: $nested"
            return $nested
        }
    }

    return $null
}

function Read-Choice {
    param(
        [string]$Prompt,
        [string[]]$Valid,
        [string]$Default
    )
    while ($true) {
        $suffix = if ($Default) { " [$Default]" } else { '' }
        $answer = Read-Host "$Prompt$suffix"
        if ([string]::IsNullOrWhiteSpace($answer)) {
            $answer = $Default
        }
        foreach ($value in $Valid) {
            if ($answer -ieq $value) {
                return $value
            }
        }
        Write-Host "Please enter one of: $($Valid -join ', ')." -ForegroundColor Yellow
    }
}

Write-ArpBanner

function Resolve-ComfyInstallMode {
    Write-Host ''
    Write-Host 'ComfyUI setup' -ForegroundColor Cyan
    if ($ComfyDir) {
        $full = [System.IO.Path]::GetFullPath($ComfyDir)
        $resolved = Resolve-ComfyDirPath $full
        if ($resolved) {
            Write-Host "Using ComfyUI from -ComfyDir: $resolved"
            return @{
                Dir = $resolved
                Existing = $true
            }
        }
        Write-Host "Using ComfyUI target from -ComfyDir: $full"
        return @{
            Dir = $full
            Existing = $false
        }
    }

    $defaultFull = [System.IO.Path]::GetFullPath($DefaultComfyDir)
    $selected = $defaultFull
    if (-not $NonInteractive) {
        Write-Host 'Choose where ARP should find or install ComfyUI.'
        Write-Host "Press Enter for the ARP-managed default: $selected"
        $answer = Read-Host 'ComfyUI directory'
        if (-not [string]::IsNullOrWhiteSpace($answer)) {
            $selected = [System.IO.Path]::GetFullPath($answer.Trim())
        }
    }

    $resolvedSelected = Resolve-ComfyDirPath $selected
    if ($resolvedSelected) {
        $selectedIsDefault = [string]::Equals(
            [System.IO.Path]::GetFullPath($resolvedSelected),
            $defaultFull,
            [System.StringComparison]::OrdinalIgnoreCase
        )
        if ($selectedIsDefault) {
            Write-Host "Using ARP-managed ComfyUI at: $resolvedSelected"
            return @{
                Dir = $resolvedSelected
                Existing = $false
            }
        }
        Write-Host "Using existing external ComfyUI at: $resolvedSelected"
        return @{
            Dir = $resolvedSelected
            Existing = $true
        }
    }

    Write-Host "Using ARP-managed ComfyUI target: $selected"
    Write-Host 'This is the lowest-maintenance path: rerunning install_windows.bat refreshes ComfyUI and required custom nodes.'
    Write-Host 'If you need an external checkout, rerun with: install_windows.bat -ComfyDir C:\Path\To\ComfyUI'

    return @{
        Dir = $selected
        Existing = $false
    }
}

function Git-Clone-IfMissing {
    param([string]$Repo, [string]$Destination, [switch]$UpdateExisting, [switch]$RequireComfyMain)
    if (Test-Path -LiteralPath $Destination) {
        if ($UpdateExisting) {
            if (Test-Path -LiteralPath (Join-Path $Destination '.git')) {
                Write-Host "Updating existing checkout: $Destination"
                $ok = Invoke-External-Optional -Command @('git', '-C', $Destination, 'pull', '--ff-only')
                if (-not $ok) {
                    Write-Host "WARNING: Could not fast-forward update '$Destination'." -ForegroundColor Yellow
                    Write-Host "  This can happen if local files were changed or the history diverged." -ForegroundColor Yellow
                    Write-Host "  To fix: delete the folder below and rerun install_windows.bat." -ForegroundColor Yellow
                    Write-Host "  Folder: $Destination" -ForegroundColor Yellow
                }
            } else {
                # Folder exists but is not a git repo — likely a manual zip-download install.
                # We can't 'git pull' it, so newer node types may be missing.
                Write-Host ""
                Write-Host "WARNING: '$Destination' exists but is not a Git checkout." -ForegroundColor Yellow
                Write-Host "  It was probably installed by extracting a zip download, which cannot be updated automatically." -ForegroundColor Yellow
                Write-Host "  Some required node types may be missing or outdated." -ForegroundColor Yellow
                if ($RequireComfyMain -and -not (Test-ComfyDir $Destination)) {
                    Write-Host "  This folder is not a usable ComfyUI install because main.py is missing." -ForegroundColor Yellow
                    Write-Host "  ARP cannot continue unless this folder is replaced or you pass -ComfyDir with a valid ComfyUI checkout." -ForegroundColor Yellow
                }
                Write-Host ""
                Write-Host "  Option: replace the existing folder with a fresh clone from:" -ForegroundColor Cyan
                Write-Host "  $Repo" -ForegroundColor Cyan
                Write-Host ""
                if ($NonInteractive) {
                    throw "Cannot update '$Destination' because it is not a Git checkout. Rename or delete it, then rerun install_windows.bat."
                }
                $answer = Read-Choice "  Replace it with a fresh Git clone?" @('Y', 'N') 'Y'
                if ($answer -match '^[Yy]') {
                    Remove-Item -LiteralPath $Destination -Recurse -Force
                    Invoke-External -Command @('git', 'clone', $Repo, $Destination)
                } else {
                    if ($RequireComfyMain -and -not (Test-ComfyDir $Destination)) {
                        throw "Install cancelled because '$Destination' is not a usable ComfyUI checkout. Rerun install_windows.bat and choose Y, or pass -ComfyDir with a valid ComfyUI checkout."
                    }
                    Write-Host "  Skipping. If you see missing-node errors, rename or delete '$Destination' and rerun install_windows.bat." -ForegroundColor Yellow
                }
            }
        } else {
            Write-Host "Already exists: $Destination"
        }
        return
    }
    Ensure-Directory (Split-Path -Parent $Destination)
    Invoke-External -Command @('git', 'clone', $Repo, $Destination)
}

function Copy-BundledCustomNode {
    param([string]$Name, [string]$Destination, [switch]$UpdateExisting)
    $source = Join-Path $BundledCustomNodes $Name
    if (-not (Test-Path -LiteralPath $source -PathType Container)) {
        return $false
    }
    if (Test-Path -LiteralPath $Destination) {
        $existing = Get-ChildItem -LiteralPath $Destination -Force -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($existing -and -not $UpdateExisting) {
            return $true
        }
    } else {
        Ensure-Directory $Destination
    }
    Write-Host "Installing bundled custom node: $Name"
    Get-ChildItem -LiteralPath $source -Force | Copy-Item -Destination $Destination -Recurse -Force
    return $true
}

function Install-CustomNodePackage {
    param(
        [string]$Name,
        [string]$Repo,
        [string]$Destination,
        [switch]$UpdateExisting,
        [switch]$PreferBundled
    )
    if ($PreferBundled -and (Copy-BundledCustomNode $Name $Destination -UpdateExisting:$UpdateExisting)) {
        return
    }

    if (Test-Path -LiteralPath $Destination) {
        if ($UpdateExisting) {
            Git-Clone-IfMissing $Repo $Destination -UpdateExisting
        } else {
            Git-Clone-IfMissing $Repo $Destination
        }
        return
    }

    try {
        Git-Clone-IfMissing $Repo $Destination
    } catch {
        Write-Host "WARNING: Could not clone $Name from $Repo." -ForegroundColor Yellow
        Write-Host "  Falling back to the bundled copy if one is available. Details: $_" -ForegroundColor Yellow
        if (-not (Copy-BundledCustomNode $Name $Destination)) {
            throw
        }
    }
}

function Test-DirectoryContainsText {
    param([string]$Directory, [string]$Needle)
    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
        return $false
    }
    $match = Get-ChildItem -LiteralPath $Directory -Recurse -File -Include '*.py' -ErrorAction SilentlyContinue |
        Select-String -SimpleMatch -Pattern $Needle -List -ErrorAction SilentlyContinue |
        Select-Object -First 1
    return $null -ne $match
}

function Assert-CustomNodeSymbols {
    param(
        [string]$Name,
        [string]$Directory,
        [string]$RepoUrl,
        [string[]]$Symbols
    )
    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
        throw "$Name was not installed. Expected folder: $Directory. Install it from $RepoUrl into ComfyUI\custom_nodes and rerun install_windows.bat."
    }
    $isGitCheckout = Test-Path -LiteralPath (Join-Path $Directory '.git')
    $missing = @()
    foreach ($symbol in $Symbols) {
        if (-not (Test-DirectoryContainsText $Directory $symbol)) {
            $missing += $symbol
        }
    }
    if ($missing.Count -gt 0) {
        $msg = "$Name is installed at $Directory, but required node type(s) are missing: $($missing -join ', '). "
        if (-not $isGitCheckout) {
            $msg += "The folder is not a Git checkout (it was likely installed by extracting a zip). "
            $msg += "Delete '$Directory' and rerun install_windows.bat so it can clone the latest version. "
        } else {
            $msg += "Delete '$Directory' and rerun install_windows.bat to get a fresh clone, then fully restart ComfyUI. "
        }
        $msg += "Repo: $RepoUrl"
        throw $msg
    }
    Write-Host "$Name node definitions found: $($Symbols -join ', ')"
}

function Install-Pip {
    param([string[]]$Packages)
    Invoke-External -Command (@($PipelinePython, '-m', 'pip', 'install') + $Packages)
}

function Install-Pip-WithPyTorchHint {
    param([string[]]$Packages)
    try {
        Install-Pip $Packages
    } catch {
        $packageText = $Packages -join ' '
        if ($packageText -match '(^|\s)torch(\s|$)') {
            Write-Host ''
            Write-Host 'PyTorch wheel resolution failed.' -ForegroundColor Yellow
            Write-Host 'This usually means the installer venv was created with an unsupported Python build, such as 32-bit Python or free-threaded Python.' -ForegroundColor Yellow
            Write-Host 'Install standard 64-bit CPython 3.13 from python.org, delete this repo''s .venv folder, and rerun install_windows.bat.' -ForegroundColor Yellow
            Write-Host "Current PyTorch index URL: $TorchIndexUrl" -ForegroundColor Yellow
        }
        throw
    }
}

function Get-PyTorchStatus {
    $probeName = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetRandomFileName(), '.py')
    $probePath = Join-Path ([System.IO.Path]::GetTempPath()) $probeName
    $probe = @'
import json
try:
    import torch
    print(json.dumps({
        "ok": True,
        "version": getattr(torch, "__version__", ""),
        "cuda_build": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
    }))
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
'@
    Set-Content -LiteralPath $probePath -Value $probe -Encoding UTF8
    try {
        $result = Invoke-CapturedProcess -FilePath $PipelinePython -Arguments @($probePath)
    } finally {
        Remove-Item -LiteralPath $probePath -Force -ErrorAction SilentlyContinue
    }
    if ($result.ExitCode -ne 0) {
        return [pscustomobject]@{
            Ok = $false
            Error = (($result.Stderr + "`n" + $result.Stdout).Trim())
            Version = ''
            CudaBuild = ''
            CudaAvailable = $false
        }
    }
    try {
        $json = $result.Stdout | ConvertFrom-Json
        return [pscustomobject]@{
            Ok = [bool]$json.ok
            Error = [string]$json.error
            Version = [string]$json.version
            CudaBuild = [string]$json.cuda_build
            CudaAvailable = [bool]$json.cuda_available
        }
    } catch {
        return [pscustomobject]@{
            Ok = $false
            Error = "Could not parse PyTorch probe output: $($result.Stdout)"
            Version = ''
            CudaBuild = ''
            CudaAvailable = $false
        }
    }
}

function Install-PyTorch-Cuda {
    Write-Host "Installing PyTorch CUDA packages from $TorchIndexUrl"
    Install-Pip-WithPyTorchHint @('--upgrade', '--force-reinstall', 'torch', 'torchvision', 'torchaudio', '--index-url', $TorchIndexUrl)
}

function Get-DesiredTorchCudaBuild {
    $match = [regex]::Match($TorchIndexUrl, 'cu(\d{2,3})(?:\D|$)')
    if (-not $match.Success) {
        return ''
    }
    $tag = $match.Groups[1].Value
    if ($tag.Length -eq 3) {
        return "$($tag.Substring(0, 2)).$($tag.Substring(2, 1))"
    }
    if ($tag.Length -eq 2) {
        return "$($tag.Substring(0, 1)).$($tag.Substring(1, 1))"
    }
    return ''
}

function Install-CorrelationExtension-BestEffort {
    Write-Host 'Installing optional Pytorch-Correlation-extension for optimized ColorMNet.'
    Write-Host 'If local CUDA/MSVC build tools are missing or mismatched, ARP will continue with the PyTorch fallback.'
    $readiness = Test-CorrelationExtensionBuildReady
    if (-not $readiness.Ready) {
        Write-Host "WARNING: Skipping Pytorch-Correlation-extension build: $($readiness.Reason)" -ForegroundColor Yellow
        Write-Host '  ColorMNet will use its PyTorch fallback path. Quality is unchanged.' -ForegroundColor Yellow
        return
    }
    try {
        Install-Pip @('git+https://github.com/ClementPinard/Pytorch-Correlation-extension.git', '--no-build-isolation')
        Write-Host 'Pytorch-Correlation-extension installed.'
    } catch {
        Write-Host 'WARNING: Pytorch-Correlation-extension could not be built. Continuing with ColorMNet fallback mode.' -ForegroundColor Yellow
        Write-Host '  Quality is unchanged; only ColorMNet performance may be lower.' -ForegroundColor Yellow
        Write-Host '  For optimized mode, install a CUDA Toolkit matching PyTorch CUDA plus Visual Studio C++ Build Tools, then rerun install_windows.bat.' -ForegroundColor Yellow
        Write-Host "  Details: $_" -ForegroundColor Yellow
    }
}

function Test-CorrelationExtensionBuildReady {
    $cl = Get-Command 'cl.exe' -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $cl) {
        return [pscustomobject]@{ Ready = $false; Reason = 'Visual Studio C++ compiler cl.exe was not found on PATH.' }
    }

    $cudaHome = $env:CUDA_HOME
    if ([string]::IsNullOrWhiteSpace($cudaHome)) {
        $cudaHome = $env:CUDA_PATH
    }
    if ([string]::IsNullOrWhiteSpace($cudaHome)) {
        return [pscustomobject]@{ Ready = $false; Reason = 'CUDA_HOME/CUDA_PATH is not set.' }
    }

    $nvcc = Join-Path $cudaHome 'bin\nvcc.exe'
    if (-not (Test-Path -LiteralPath $nvcc)) {
        return [pscustomobject]@{ Ready = $false; Reason = "nvcc.exe was not found at $nvcc." }
    }

    $toolkitVersion = Get-CudaToolkitVersion $nvcc
    $torchStatus = Get-PyTorchStatus
    if (-not $torchStatus.Ok) {
        return [pscustomobject]@{ Ready = $false; Reason = "PyTorch is not importable: $($torchStatus.Error)" }
    }
    if ($toolkitVersion -and $torchStatus.CudaBuild -and ($toolkitVersion -ne $torchStatus.CudaBuild)) {
        return [pscustomobject]@{ Ready = $false; Reason = "CUDA Toolkit $toolkitVersion does not match PyTorch CUDA $($torchStatus.CudaBuild)." }
    }

    return [pscustomobject]@{ Ready = $true; Reason = 'CUDA/MSVC build tools found.' }
}

function Get-CudaToolkitVersion {
    param([string]$NvccPath)
    try {
        $output = (& $NvccPath --version 2>&1) -join "`n"
        $match = [regex]::Match($output, 'release\s+(\d+\.\d+)')
        if ($match.Success) {
            return $match.Groups[1].Value
        }
    } catch {
        return $null
    }
    return $null
}

function Ensure-PyTorch-Cuda {
    $desiredCudaBuild = Get-DesiredTorchCudaBuild
    $status = Get-PyTorchStatus
    if (-not $status.Ok) {
        Write-Host "PyTorch is not importable yet: $($status.Error)" -ForegroundColor Yellow
        Install-PyTorch-Cuda
        $status = Get-PyTorchStatus
    } elseif ([string]::IsNullOrWhiteSpace($status.CudaBuild)) {
        Write-Host "Existing PyTorch is CPU-only: torch $($status.Version)." -ForegroundColor Yellow
        Write-Host 'Replacing it with a CUDA build so ComfyUI can start on NVIDIA systems.' -ForegroundColor Yellow
        Install-PyTorch-Cuda
        $status = Get-PyTorchStatus
    } elseif ($desiredCudaBuild -and $status.CudaBuild -and ($status.CudaBuild -ne $desiredCudaBuild)) {
        Write-Host "Existing PyTorch CUDA build is $($status.CudaBuild), but installer target is CUDA $desiredCudaBuild." -ForegroundColor Yellow
        Write-Host "Replacing it from: $TorchIndexUrl" -ForegroundColor Yellow
        Install-PyTorch-Cuda
        $status = Get-PyTorchStatus
    } else {
        Write-Host "PyTorch CUDA build found: torch $($status.Version), CUDA $($status.CudaBuild)"
    }

    if (-not $status.Ok) {
        throw "PyTorch still cannot be imported after installation: $($status.Error)"
    }
    if ([string]::IsNullOrWhiteSpace($status.CudaBuild)) {
        throw "PyTorch installed successfully, but it is still a CPU-only build. Delete '$Root\.venv' and rerun install_windows.bat. Current torch: $($status.Version)"
    }
    if ($desiredCudaBuild -and ($status.CudaBuild -ne $desiredCudaBuild)) {
        throw "PyTorch installed successfully, but torch $($status.Version) reports CUDA $($status.CudaBuild) instead of the installer target CUDA $desiredCudaBuild from $TorchIndexUrl."
    }
    if (-not $status.CudaAvailable) {
        throw "PyTorch has a CUDA build (torch $($status.Version), CUDA $($status.CudaBuild)), but torch.cuda.is_available() is false. Install or update your NVIDIA driver, then rerun install_windows.bat."
    }
    Write-Host "PyTorch CUDA check passed: torch $($status.Version), CUDA $($status.CudaBuild)"
}

function Install-PythonBuildTools {
    Install-Pip @('--upgrade', 'pip', 'wheel')
}

function Assert-PipelinePythonCompatible {
    if (-not (Test-Path -LiteralPath $PipelinePython)) {
        return
    }
    $check = Get-PythonLauncherCheck @($PipelinePython)
    if (-not $check.Success) {
        throw "The existing installer virtual environment is not compatible: $($check.Reason). Delete '$Root\.venv', install standard 64-bit CPython 3.13, and rerun install_windows.bat."
    }
    Write-Host "Installer Python check: $($check.Reason)"
}

function Install-RequirementsIfPresent {
    param([string]$RequirementsPath)
    if (Test-Path -LiteralPath $RequirementsPath) {
        Install-Pip @('-r', $RequirementsPath)
    }
}

function Download-HfFile {
    param(
        [string]$Repo,
        [string]$File,
        [string]$Destination
    )
    if (-not [System.IO.Path]::IsPathRooted($Destination)) {
        $Destination = Resolve-ComfyModelDestination $Destination
    }
    if (Test-Path -LiteralPath $Destination) {
        Write-Host "Model already exists: $Destination"
        return
    }
    Ensure-Directory (Split-Path -Parent $Destination)
    Ensure-Directory $DownloadCache
    $HfExe = Join-Path $Root '.venv\Scripts\hf.exe'
    if (-not (Test-Path -LiteralPath $HfExe)) { $HfExe = 'hf' }
    $oldPythonUtf8 = $env:PYTHONUTF8
    $oldPythonIoEncoding = $env:PYTHONIOENCODING
    $oldDisableProgress = $env:HF_HUB_DISABLE_PROGRESS_BARS
    try {
        $env:PYTHONUTF8 = '1'
        $env:PYTHONIOENCODING = 'utf-8'
        Remove-Item Env:\HF_HUB_DISABLE_PROGRESS_BARS -ErrorAction SilentlyContinue
        Write-Host "Downloading model: $Repo/$File"
        try {
            Invoke-External -Command @($HfExe, 'download', $Repo, $File, '--local-dir', $DownloadCache)
        } catch {
            throw "hf download failed for $Repo/$File`n$_"
        }
    } finally {
        $env:PYTHONUTF8 = $oldPythonUtf8
        $env:PYTHONIOENCODING = $oldPythonIoEncoding
        $env:HF_HUB_DISABLE_PROGRESS_BARS = $oldDisableProgress
    }
    $source = Join-Path $DownloadCache $File
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Downloaded file was not found for $Repo/$File."
    }
    Move-Item -LiteralPath $source -Destination $Destination
    Write-Host "Downloaded: $Destination"
}

function Resolve-ComfyModelDestination {
    param([string]$Destination)
    $resolver = Join-Path $Root 'scripts\model_paths.py'
    $result = Invoke-CapturedProcess `
        -FilePath $PipelinePython `
        -Arguments @($resolver, '--comfy-dir', $ComfyDir, '--destination', $Destination)
    if ($result.ExitCode -ne 0) {
        $detail = (($result.Stdout, $result.Stderr) -join "`n").Trim()
        throw "Could not resolve ComfyUI model destination '$Destination': $detail"
    }
    $resolved = $result.Stdout.Trim()
    if ([string]::IsNullOrWhiteSpace($resolved)) {
        throw "ComfyUI model destination resolver returned an empty path for '$Destination'."
    }
    return $resolved
}

$mode = Resolve-ComfyInstallMode
$ComfyDir = $mode.Dir
$UseExistingComfy = [bool]$mode.Existing
$ComfyManagedByArp = -not $UseExistingComfy
$CustomNodes = Join-Path $ComfyDir 'custom_nodes'

Invoke-Step 'Configure ComfyUI directory' {
    if ($UseExistingComfy) {
        Write-Host "Using existing ComfyUI: $ComfyDir"
        Write-Host 'ARP will refresh required custom nodes, but will not update this external ComfyUI checkout.'
        Write-Host 'Keep it current yourself with git pull and pip install -r requirements.txt.'
    } else {
        Git-Clone-IfMissing 'https://github.com/comfyanonymous/ComfyUI.git' $ComfyDir -UpdateExisting -RequireComfyMain
    }
    if (-not (Test-ComfyDir $ComfyDir)) {
        throw "ComfyUI main.py was not found in: $ComfyDir"
    }
}

function Install-FfmpegIfMissing {
    $ToolDir = Join-Path $Root '.cache\tools\ffmpeg'
    $FfmpegExe = Join-Path $ToolDir 'ffmpeg.exe'
    $FfprobeExe = Join-Path $ToolDir 'ffprobe.exe'
    if ((Test-Path -LiteralPath $FfmpegExe) -and (Test-Path -LiteralPath $FfprobeExe)) {
        Write-Host "FFmpeg already exists: $ToolDir"
        return
    }
    $ArchiveDir = Join-Path $Root '.cache\downloads'
    $Archive = Join-Path $ArchiveDir 'ffmpeg-release-essentials.zip'
    Ensure-Directory $ArchiveDir
    Ensure-Directory $ToolDir
    if (-not (Test-Path -LiteralPath $Archive)) {
        Write-Host 'Downloading FFmpeg essentials from https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
        Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile $Archive -UseBasicParsing
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($Archive)
    try {
        foreach ($entry in $zip.Entries) {
            $name = [System.IO.Path]::GetFileName($entry.FullName)
            $normalized = $entry.FullName.Replace('\', '/').ToLowerInvariant()
            if (($name -eq 'ffmpeg.exe' -or $name -eq 'ffprobe.exe') -and $normalized.Contains('/bin/')) {
                $target = Join-Path $ToolDir $name
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
            }
        }
    } finally {
        $zip.Dispose()
    }
    if (-not ((Test-Path -LiteralPath $FfmpegExe) -and (Test-Path -LiteralPath $FfprobeExe))) {
        throw "Could not extract ffmpeg.exe and ffprobe.exe from $Archive"
    }
    Write-Host "Installed FFmpeg tools: $ToolDir"
}

Invoke-Step 'Create ai-remaster-pipeline venv' {
    if (-not (Test-Path -LiteralPath $PipelinePython)) {
        Invoke-PythonLauncher -Arguments @('-m', 'venv', (Join-Path $Root '.venv'))
    }
    Assert-PipelinePythonCompatible
    Install-PythonBuildTools
    Invoke-External -Command @($PipelinePython, '-m', 'pip', 'install', '-r', (Join-Path $Root 'requirements.txt'))
}

Invoke-Step 'Install PyTorch CUDA and ComfyUI requirements' {
    Ensure-PyTorch-Cuda
    # The exact-semantics LTX guide patch routes only genuinely unmasked rows
    # through xFormers. 0.0.35 uses PyTorch's stable ABI and the official
    # CUDA wheel index, so it remains compatible with the managed Torch 2.10+
    # runtime without replacing Torch itself.
    Invoke-External -Command @($PipelinePython, '-m', 'pip', 'install', 'xformers==0.0.35', '--index-url', $TorchIndexUrl)
    Install-RequirementsIfPresent (Join-Path $ComfyDir 'requirements.txt')
    Install-Pip @('huggingface_hub[cli]', 'opencv-contrib-python', 'imageio-ffmpeg', 'pillow', 'numpy', 'numba')
}

Invoke-Step 'Install ComfyUI custom nodes' {
    Ensure-Directory $CustomNodes
    if (-not $SkipComfyManager) {
        Git-Clone-IfMissing 'https://github.com/ltdrdata/ComfyUI-Manager.git' (Join-Path $CustomNodes 'ComfyUI-Manager') -UpdateExisting
    }
    Install-CustomNodePackage 'ComfyUI-LTXVideo' 'https://github.com/Lightricks/ComfyUI-LTXVideo.git' (Join-Path $CustomNodes 'ComfyUI-LTXVideo') -UpdateExisting -PreferBundled
    Install-CustomNodePackage 'ComfyUI-GGUF' 'https://github.com/city96/ComfyUI-GGUF.git' (Join-Path $CustomNodes 'ComfyUI-GGUF') -UpdateExisting
    Install-CustomNodePackage 'ComfyUI-VideoHelperSuite' 'https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git' (Join-Path $CustomNodes 'ComfyUI-VideoHelperSuite') -UpdateExisting
    Install-CustomNodePackage 'ComfyUI_ProPainter_Nodes' 'https://github.com/daniabib/ComfyUI_ProPainter_Nodes.git' (Join-Path $CustomNodes 'ComfyUI_ProPainter_Nodes') -UpdateExisting
    Install-CustomNodePackage 'ComfyUI-FlashVSR_Ultra_Fast' 'https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast.git' (Join-Path $CustomNodes 'ComfyUI-FlashVSR_Ultra_Fast') -UpdateExisting
    Install-CustomNodePackage 'ComfyUI-MMAudio' 'https://github.com/kijai/ComfyUI-MMAudio.git' (Join-Path $CustomNodes 'ComfyUI-MMAudio') -UpdateExisting
    if (-not $SkipDeepExemplar) {
        Install-CustomNodePackage 'reference-video-colorization' 'https://github.com/jonstreeter/ComfyUI-Reference-Based-Video-Colorization.git' (Join-Path $CustomNodes 'reference-video-colorization') -UpdateExisting -PreferBundled
    }
}

Invoke-Step 'Verify required ComfyUI custom nodes' {
    Assert-CustomNodeSymbols `
        'ComfyUI-LTXVideo' `
        (Join-Path $CustomNodes 'ComfyUI-LTXVideo') `
        'https://github.com/Lightricks/ComfyUI-LTXVideo' `
        @('LTXVImgToVideoConditionOnly', 'LTXAddVideoICLoRAGuideAdvanced', 'LTXVInpaintPreprocess', 'LTXVLaplacianPyramidBlend')
    Assert-CustomNodeSymbols `
        'ComfyUI-GGUF' `
        (Join-Path $CustomNodes 'ComfyUI-GGUF') `
        'https://github.com/city96/ComfyUI-GGUF' `
        @('UnetLoaderGGUF')
    Assert-CustomNodeSymbols `
        'ComfyUI-VideoHelperSuite' `
        (Join-Path $CustomNodes 'ComfyUI-VideoHelperSuite') `
        'https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite' `
        @('VHS_LoadVideo', 'VHS_VideoCombine')
    Assert-CustomNodeSymbols `
        'ComfyUI_ProPainter_Nodes' `
        (Join-Path $CustomNodes 'ComfyUI_ProPainter_Nodes') `
        'https://github.com/daniabib/ComfyUI_ProPainter_Nodes' `
        @('ProPainterInpaint')
    Assert-CustomNodeSymbols `
        'ComfyUI-FlashVSR_Ultra_Fast' `
        (Join-Path $CustomNodes 'ComfyUI-FlashVSR_Ultra_Fast') `
        'https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast' `
        @('FlashVSRNode')
    Assert-CustomNodeSymbols `
        'ComfyUI-MMAudio' `
        (Join-Path $CustomNodes 'ComfyUI-MMAudio') `
        'https://github.com/kijai/ComfyUI-MMAudio' `
        @('MMAudioModelLoader', 'MMAudioFeatureUtilsLoader', 'MMAudioSampler')
    if (-not $SkipDeepExemplar) {
        Assert-CustomNodeSymbols `
            'ComfyUI-Reference-Based-Video-Colorization' `
            (Join-Path $CustomNodes 'reference-video-colorization') `
            'https://github.com/jonstreeter/ComfyUI-Reference-Based-Video-Colorization' `
            @('DeepExColorVideoNode', 'ColorMNetVideo')
    }
}

Invoke-Step 'Install custom-node requirements' {
    Install-RequirementsIfPresent (Join-Path $CustomNodes 'ComfyUI-LTXVideo\requirements.txt')
    Install-RequirementsIfPresent (Join-Path $CustomNodes 'ComfyUI-GGUF\requirements.txt')
    Install-RequirementsIfPresent (Join-Path $CustomNodes 'ComfyUI-VideoHelperSuite\requirements.txt')
    Install-RequirementsIfPresent (Join-Path $CustomNodes 'ComfyUI-FlashVSR_Ultra_Fast\requirements.txt')
    Install-RequirementsIfPresent (Join-Path $CustomNodes 'ComfyUI-MMAudio\requirements.txt')
    if (-not $SkipDeepExemplar) {
        Install-RequirementsIfPresent (Join-Path $CustomNodes 'reference-video-colorization\requirements.txt')
        Install-Pip @('scikit-image', 'einops', 'tqdm', 'matplotlib')
        if ($InstallCorrelationExtension) {
            Install-CorrelationExtension-BestEffort
        } else {
            Write-Host 'Skipping Pytorch-Correlation-extension. ColorMNet will use its PyTorch fallback path; pass -InstallCorrelationExtension to try building it.'
        }
    }
}

Invoke-Step 'Create model directories' {
    foreach ($dir in @('checkpoints','diffusion_models','loras','text_encoders','unet','vae','latent_upscale_models','mmaudio')) {
        Ensure-Directory (Resolve-ComfyModelDestination "models\$dir")
    }
}

Invoke-Step 'Install local FFmpeg tools' {
    Install-FfmpegIfMissing
}

if ($DownloadModels -and -not $SkipModelDownloads) {
    Invoke-Step 'Download LTX 2.3 models, Clean Up LoRA, and outpainting LoRAs' {
        Download-HfFile 'QuantStack/LTX-2.3-GGUF' 'LTX-2.3-distilled/LTX-2.3-distilled-Q4_K_M.gguf' 'models\unet\LTX-2.3-distilled-Q4_K_M.gguf'
        Download-HfFile 'Lightricks/LTX-2.3-fp8' 'ltx-2.3-22b-dev-fp8.safetensors' 'models\checkpoints\ltx-2.3-22b-dev-fp8.safetensors'
        Download-HfFile 'Comfy-Org/ltx-2' 'split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors' 'models\text_encoders\gemma_3_12B_it_fp8_scaled.safetensors'
        Download-HfFile 'Kijai/LTX2.3_comfy' 'vae/LTX23_video_vae_bf16.safetensors' 'models\vae\LTX23_video_vae_bf16.safetensors'
        Download-HfFile 'Kijai/LTX2.3_comfy' 'vae/LTX23_audio_vae_bf16.safetensors' 'models\vae\LTX23_audio_vae_bf16.safetensors'
        Download-HfFile 'oumoumad/ltx-2.3-dearchive-lora' 'lora_weights_step_05000.safetensors' 'models\loras\ltx-2.3-dearchive-lora.safetensors'
        Download-HfFile 'oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint' 'ltx-2.3-22b-ic-lora-outpaint.safetensors' 'models\loras\ltx-2.3-22b-ic-lora-outpaint.safetensors'
        Download-HfFile 'Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting' 'ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors' 'models\loras\ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors'
    }

    Invoke-Step 'Download Qwen Image Edit 2511 GGUF Q4_K_M models and Lightning LoRA' {
        Download-HfFile 'unsloth/Qwen-Image-Edit-2511-GGUF' 'qwen-image-edit-2511-Q4_K_M.gguf' 'models\diffusion_models\qwen-image-edit-2511-Q4_K_M.gguf'
        Download-HfFile 'Comfy-Org/Qwen-Image_ComfyUI' 'split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors' 'models\text_encoders\qwen_2.5_vl_7b_fp8_scaled.safetensors'
        Download-HfFile 'Comfy-Org/Qwen-Image_ComfyUI' 'split_files/vae/qwen_image_vae.safetensors' 'models\vae\qwen_image_vae.safetensors'
        Download-HfFile 'lightx2v/Qwen-Image-Edit-2511-Lightning' 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors' 'models\loras\Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'
    }

    Invoke-Step 'Download soundtrack models (music + sound effects)' {
        # Stable Audio Open is a gated model: accept its licence at
        # https://huggingface.co/stabilityai/stable-audio-open-1.0 and run 'hf auth login'
        # (or set HF_TOKEN) for this to succeed. Wrapped so a skip never fails the install;
        # it is retried on demand when the Create Audio Track stage first runs.
        try {
            Download-HfFile 'stabilityai/stable-audio-open-1.0' 'model.safetensors' 'models\checkpoints\stable_audio_open_1.0.safetensors'
        } catch {
            Write-Warning "Skipped Stable Audio Open (music). Accept the licence + 'hf auth login', or it downloads on first use. Details: $_"
        }
        try {
            Download-HfFile 'google-t5/t5-base' 'model.safetensors' 'models\text_encoders\t5_base.safetensors'
        } catch {
            Write-Warning "Skipped T5-base text encoder for Stable Audio music; it downloads on first use. Details: $_"
        }
        foreach ($mmaudioFile in @('mmaudio_large_44k_v2_fp16.safetensors','mmaudio_vae_44k_fp16.safetensors','mmaudio_synchformer_fp16.safetensors','apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors')) {
            try {
                Download-HfFile 'Kijai/MMAudio_safetensors' $mmaudioFile "models\mmaudio\$mmaudioFile"
            } catch {
                Write-Warning "Skipped MMAudio file $mmaudioFile (sound effects); it downloads on first use. Details: $_"
            }
        }
    }
} else {
    Write-Host 'Skipping model downloads. Models and LoRAs will download on demand when their pipeline stages run.'
}

Invoke-Step 'Write local ARP configuration' {
    $config = [ordered]@{
        comfy_dir = $ComfyDir
        comfy_url = 'http://127.0.0.1:8188'
        comfy_host = '127.0.0.1'
        comfy_port = '8188'
        comfy_managed_by_arp = if ($ComfyManagedByArp) { 'true' } else { 'false' }
    }
    $configPath = Join-Path $Root '.ai_remaster_config.json'
    ($config | ConvertTo-Json -Depth 4) | Set-Content -LiteralPath $configPath -Encoding UTF8
    Write-Host "Wrote: $configPath"
}

Write-Host "`nInstall complete." -ForegroundColor Green
Write-Host "ComfyUI: $ComfyDir"
Write-Host "Python environment: $PipelinePython"
Write-Host "Start ARP with:"
Write-Host "  launch_gui.bat"

