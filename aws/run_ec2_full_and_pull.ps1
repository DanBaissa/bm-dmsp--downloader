param(
    [Parameter(Mandatory = $true)]
    [string]$HostName,

    [Parameter(Mandatory = $true)]
    [string]$KeyPath,

    [string]$SshUser = "ubuntu",
    [string]$RemoteDir = "~/bm-dmsp--downloader"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$sshTarget = "${SshUser}@${HostName}"
$sshArgs = @("-i", $KeyPath, "-o", "StrictHostKeyChecking=accept-new")

function Invoke-Remote {
    param([string]$Command)
    & ssh @sshArgs $sshTarget $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed: $Command"
    }
}

function Copy-ToRemote {
    param(
        [string]$Source,
        [string]$Destination
    )
    & scp @sshArgs -r $Source "${sshTarget}:$Destination"
    if ($LASTEXITCODE -ne 0) {
        throw "Upload failed: $Source"
    }
}

function Copy-FromRemote {
    param(
        [string]$Source,
        [string]$Destination
    )
    & scp @sshArgs -r "${sshTarget}:$Source" $Destination
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $Source"
    }
}

$localResultDir = Join-Path $repoRoot "ec2_full_run_from_ec2"
if (Test-Path $localResultDir) {
    Remove-Item -LiteralPath $localResultDir -Recurse -Force
}
New-Item -ItemType Directory -Path $localResultDir | Out-Null

Invoke-Remote "mkdir -p $RemoteDir"

Copy-ToRemote (Join-Path $repoRoot "data_sampler.py") $RemoteDir
Copy-ToRemote (Join-Path $repoRoot "requirements.txt") $RemoteDir
Copy-ToRemote (Join-Path $repoRoot "README.md") $RemoteDir
Copy-ToRemote (Join-Path $repoRoot ".env") $RemoteDir
Copy-ToRemote (Join-Path $repoRoot "aws") $RemoteDir
Copy-ToRemote (Join-Path $repoRoot "Data") $RemoteDir

Invoke-Remote "cd $RemoteDir && bash aws/bootstrap_ubuntu.sh"
Invoke-Remote "cd $RemoteDir && source .venv/bin/activate && bash aws/run_full_download.sh"

Copy-FromRemote "$RemoteDir/ec2_full_run" $localResultDir

Write-Host "Downloaded EC2 full results to $localResultDir"
