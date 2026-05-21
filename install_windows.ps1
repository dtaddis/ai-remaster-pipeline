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
$DownloadCache = Join-Path $Root '.cache\huggingface'
$DefaultComfyDir = Join-Path $Root 'tools\comfyui'
$PipelinePython = Join-Path $Root '.venv\Scripts\python.exe'
$UseExistingComfy = $false

function Invoke-Step {
    param([string]$Label, [scriptblock]$Block)
    Write-Host "`n==> $Label" -ForegroundColor Cyan
    & $Block
}

function Invoke-External {
    param([string[]]$Command, [string]$WorkingDirectory = $Root)
    Write-Host ($Command -join ' ')
    $process = Start-Process -FilePath $Command[0] -ArgumentList $Command[1..($Command.Count-1)] -WorkingDirectory $WorkingDirectory -NoNewWindow -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Command failed with exit code $($process.ExitCode): $($Command -join ' ')"
    }
}

function Invoke-PythonLauncher {
    param([string[]]$Arguments, [string]$WorkingDirectory = $Root)
    $parts = $PythonLauncher -split ' '
    Invoke-External -Command ($parts + $Arguments) -WorkingDirectory $WorkingDirectory
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

function Resolve-ComfyInstallMode {
    if ($ComfyDir) {
        $full = [System.IO.Path]::GetFullPath($ComfyDir)
        return @{
            Dir = $full
            Existing = (Test-ComfyDir $full)
        }
    }

    if ($NonInteractive) {
        return @{
            Dir = [System.IO.Path]::GetFullPath($DefaultComfyDir)
            Existing = $false
        }
    }

    Write-Host ''
    Write-Host 'ComfyUI setup' -ForegroundColor Cyan
    Write-Host '1. Clone ComfyUI into this project under tools\comfyui'
    Write-Host '2. Use an existing ComfyUI directory'
    $choice = Read-Choice 'Choose ComfyUI setup mode (1 or 2)' @('1', '2') '1'

    if ($choice -eq '1') {
        $target = Read-Host "Clone destination [$DefaultComfyDir]"
        if ([string]::IsNullOrWhiteSpace($target)) {
            $target = $DefaultComfyDir
        }
        return @{
            Dir = [System.IO.Path]::GetFullPath($target)
            Existing = $false
        }
    }

    while ($true) {
        $target = Read-Host 'Existing ComfyUI directory'
        if ([string]::IsNullOrWhiteSpace($target)) {
            Write-Host 'Please enter a ComfyUI directory.' -ForegroundColor Yellow
            continue
        }
        $full = [System.IO.Path]::GetFullPath($target)
        if (Test-ComfyDir $full) {
            return @{
                Dir = $full
                Existing = $true
            }
        }
        Write-Host "That does not look like a ComfyUI checkout because main.py was not found: $full" -ForegroundColor Yellow
        $retry = Read-Choice 'Try another path? (Y/N)' @('Y', 'N') 'Y'
        if ($retry -eq 'N') {
            throw 'Install cancelled.'
        }
    }
}

function Git-Clone-IfMissing {
    param([string]$Repo, [string]$Destination)
    if (Test-Path -LiteralPath $Destination) {
        Write-Host "Already exists: $Destination"
        return
    }
    Ensure-Directory (Split-Path -Parent $Destination)
    Invoke-External -Command @('git', 'clone', $Repo, $Destination)
}

function Install-Pip {
    param([string[]]$Packages)
    Invoke-External -Command (@($PipelinePython, '-m', 'pip', 'install') + $Packages)
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
    if (Test-Path -LiteralPath $Destination) {
        Write-Host "Model already exists: $Destination"
        return
    }
    Ensure-Directory (Split-Path -Parent $Destination)
    Ensure-Directory $DownloadCache
    $HfExe = Join-Path $Root '.venv\Scripts\hf.exe'
    if (-not (Test-Path -LiteralPath $HfExe)) { $HfExe = 'hf' }
    $stdout = [System.IO.Path]::GetTempFileName()
    $stderr = [System.IO.Path]::GetTempFileName()
    $oldPythonUtf8 = $env:PYTHONUTF8
    $oldPythonIoEncoding = $env:PYTHONIOENCODING
    $oldDisableProgress = $env:HF_HUB_DISABLE_PROGRESS_BARS
    try {
        $env:PYTHONUTF8 = '1'
        $env:PYTHONIOENCODING = 'utf-8'
        $env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
        $process = Start-Process `
            -FilePath $HfExe `
            -ArgumentList @('download', $Repo, $File, '--local-dir', $DownloadCache) `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr
        $downloaded = ((Get-Content -LiteralPath $stdout -Raw -ErrorAction SilentlyContinue) + "`n" + (Get-Content -LiteralPath $stderr -Raw -ErrorAction SilentlyContinue)).Trim()
        if ($process.ExitCode -ne 0) {
            throw "hf download failed for $Repo/$File`n$downloaded"
        }
        if ($downloaded) {
            Write-Host $downloaded
        }
    } finally {
        $env:PYTHONUTF8 = $oldPythonUtf8
        $env:PYTHONIOENCODING = $oldPythonIoEncoding
        $env:HF_HUB_DISABLE_PROGRESS_BARS = $oldDisableProgress
        Remove-Item -LiteralPath $stdout, $stderr -Force -ErrorAction SilentlyContinue
    }
    $source = Join-Path $DownloadCache $File
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Downloaded file was not found for $Repo/$File. Last output: $downloaded"
    }
    Move-Item -LiteralPath $source -Destination $Destination
    Write-Host "Downloaded: $Destination"
}

$mode = Resolve-ComfyInstallMode
$ComfyDir = $mode.Dir
$UseExistingComfy = [bool]$mode.Existing
$CustomNodes = Join-Path $ComfyDir 'custom_nodes'

Invoke-Step 'Configure ComfyUI directory' {
    if ($UseExistingComfy) {
        Write-Host "Using existing ComfyUI: $ComfyDir"
    } else {
        Git-Clone-IfMissing 'https://github.com/comfyanonymous/ComfyUI.git' $ComfyDir
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
    Invoke-External -Command @($PipelinePython, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel')
    Invoke-External -Command @($PipelinePython, '-m', 'pip', 'install', '-r', (Join-Path $Root 'requirements.txt'))
}

Invoke-Step 'Install PyTorch CUDA and ComfyUI requirements' {
    Install-Pip @('torch', 'torchvision', 'torchaudio', '--index-url', $TorchIndexUrl)
    Install-RequirementsIfPresent (Join-Path $ComfyDir 'requirements.txt')
    Install-Pip @('huggingface_hub[cli]', 'opencv-contrib-python', 'imageio-ffmpeg', 'pillow', 'numpy', 'numba')
}

Invoke-Step 'Install ComfyUI custom nodes' {
    Ensure-Directory $CustomNodes
    if (-not $SkipComfyManager) {
        Git-Clone-IfMissing 'https://github.com/ltdrdata/ComfyUI-Manager.git' (Join-Path $CustomNodes 'ComfyUI-Manager')
    }
    Git-Clone-IfMissing 'https://github.com/Lightricks/ComfyUI-LTXVideo.git' (Join-Path $CustomNodes 'ComfyUI-LTXVideo')
    Git-Clone-IfMissing 'https://github.com/city96/ComfyUI-GGUF.git' (Join-Path $CustomNodes 'ComfyUI-GGUF')
    if (-not $SkipDeepExemplar) {
        Git-Clone-IfMissing 'https://github.com/jonstreeter/ComfyUI-Reference-Based-Video-Colorization.git' (Join-Path $CustomNodes 'reference-video-colorization')
    }
}

Invoke-Step 'Install custom-node requirements' {
    Install-RequirementsIfPresent (Join-Path $CustomNodes 'ComfyUI-LTXVideo\requirements.txt')
    Install-RequirementsIfPresent (Join-Path $CustomNodes 'ComfyUI-GGUF\requirements.txt')
    if (-not $SkipDeepExemplar) {
        Install-RequirementsIfPresent (Join-Path $CustomNodes 'reference-video-colorization\requirements.txt')
        Install-Pip @('scikit-image', 'einops', 'tqdm', 'matplotlib')
        if ($InstallCorrelationExtension) {
            Install-Pip @('git+https://github.com/ClementPinard/Pytorch-Correlation-extension.git')
        } else {
            Write-Host 'Skipping Pytorch-Correlation-extension. The Deep Exemplar node usually starts without it; pass -InstallCorrelationExtension to try building it.'
        }
    }
}

Invoke-Step 'Create model directories' {
    foreach ($dir in @('checkpoints','diffusion_models','loras','text_encoders','unet','vae','latent_upscale_models')) {
        Ensure-Directory (Join-Path $ComfyDir "models\$dir")
    }
}

Invoke-Step 'Install local FFmpeg tools' {
    Install-FfmpegIfMissing
}

if ($DownloadModels -and -not $SkipModelDownloads) {
    Invoke-Step 'Download LTX 2.3 models and outpainting LoRA' {
        Download-HfFile 'QuantStack/LTX-2.3-GGUF' 'LTX-2.3-distilled/LTX-2.3-distilled-Q4_K_M.gguf' (Join-Path $ComfyDir 'models\unet\LTX-2.3-distilled-Q4_K_M.gguf')
        Download-HfFile 'Lightricks/LTX-2.3-fp8' 'ltx-2.3-22b-dev-fp8.safetensors' (Join-Path $ComfyDir 'models\checkpoints\ltx-2.3-22b-dev-fp8.safetensors')
        Download-HfFile 'Comfy-Org/ltx-2' 'split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors' (Join-Path $ComfyDir 'models\text_encoders\gemma_3_12B_it_fp8_scaled.safetensors')
        Download-HfFile 'Kijai/LTX2.3_comfy' 'vae/LTX23_video_vae_bf16.safetensors' (Join-Path $ComfyDir 'models\vae\LTX23_video_vae_bf16.safetensors')
        Download-HfFile 'Kijai/LTX2.3_comfy' 'vae/LTX23_audio_vae_bf16.safetensors' (Join-Path $ComfyDir 'models\vae\LTX23_audio_vae_bf16.safetensors')
        Download-HfFile 'oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint' 'ltx-2.3-22b-ic-lora-outpaint.safetensors' (Join-Path $ComfyDir 'models\loras\ltx-2.3-22b-ic-lora-outpaint.safetensors')
    }

    Invoke-Step 'Download Qwen Image Edit 2511 GGUF Q4_K_M models and Lightning LoRA' {
        Download-HfFile 'unsloth/Qwen-Image-Edit-2511-GGUF' 'qwen-image-edit-2511-Q4_K_M.gguf' (Join-Path $ComfyDir 'models\diffusion_models\qwen-image-edit-2511-Q4_K_M.gguf')
        Download-HfFile 'Comfy-Org/Qwen-Image_ComfyUI' 'split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors' (Join-Path $ComfyDir 'models\text_encoders\qwen_2.5_vl_7b_fp8_scaled.safetensors')
        Download-HfFile 'Comfy-Org/Qwen-Image_ComfyUI' 'split_files/vae/qwen_image_vae.safetensors' (Join-Path $ComfyDir 'models\vae\qwen_image_vae.safetensors')
        Download-HfFile 'lightx2v/Qwen-Image-Edit-2511-Lightning' 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors' (Join-Path $ComfyDir 'models\loras\Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors')
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
    }
    $configPath = Join-Path $Root '.ai_remaster_config.json'
    ($config | ConvertTo-Json -Depth 4) | Set-Content -LiteralPath $configPath -Encoding UTF8
    Write-Host "Wrote: $configPath"
}

Write-Host "`nInstall complete." -ForegroundColor Green
Write-Host "ComfyUI: $ComfyDir"
Write-Host "Python environment: $PipelinePython"
Write-Host "Start ComfyUI with:"
Write-Host "  cd `"$ComfyDir`""
Write-Host "  `"$PipelinePython`" main.py --listen 127.0.0.1 --port 8188"

