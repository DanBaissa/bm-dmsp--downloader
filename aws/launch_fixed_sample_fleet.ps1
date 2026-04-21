param(
    [int]$ShardCount = 10,
    [int]$SamplesPerBin = 2000,
    [int]$WarmupRowsPerShard = 25,
    [int]$PatchSize = 1000,
    [int]$MaxWorkers = 8,
    [string]$InstanceType = "m6i.2xlarge",
    [int]$VolumeSizeGiB = 2000,
    [int]$VolumeIops = 12000,
    [int]$VolumeThroughput = 500,
    [string]$VolumeType = "gp3",
    [string]$Region = "us-east-2",
    [string]$Profile,
    [string]$ImageId,
    [string]$SubnetId,
    [string[]]$SecurityGroupIds,
    [Parameter(Mandatory = $true)]
    [string]$KeyName,
    [Parameter(Mandatory = $true)]
    [string]$KeyPath,
    [string]$SshUser = "ubuntu",
    [string]$FleetName = "bm-dmsp-fixed-sample",
    [int]$SamplingSeed = 13492,
    [int]$DateSeed = 13492,
    [string]$RemoteDir = "~/bm-dmsp--downloader"
)

$ErrorActionPreference = "Stop"

function Invoke-AwsCli {
    param([string[]]$Arguments)
    $awsArgs = @()
    if ($Profile) {
        $awsArgs += @("--profile", $Profile)
    }
    $awsArgs += $Arguments
    $output = & aws @awsArgs
    if ($LASTEXITCODE -ne 0) {
        throw "AWS CLI command failed: aws $($awsArgs -join ' ')"
    }
    return $output
}

function Invoke-Remote {
    param(
        [string]$SshTarget,
        [string[]]$SshArgs,
        [string]$Command
    )
    & ssh @SshArgs $SshTarget $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed on ${SshTarget}: $Command"
    }
}

function Copy-ToRemote {
    param(
        [string[]]$SshArgs,
        [string]$SshTarget,
        [string]$Source,
        [string]$Destination
    )
    & scp @SshArgs -r $Source "${SshTarget}:$Destination"
    if ($LASTEXITCODE -ne 0) {
        throw "Upload failed: $Source"
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$fleetRoot = Join-Path $repoRoot ("ec2_fixed_sample_fleet_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
$sampleRoot = Join-Path $fleetRoot "master_sample"
$shardRoot = Join-Path $fleetRoot "shards"
$metadataPath = Join-Path $fleetRoot "instances.csv"

New-Item -ItemType Directory -Force -Path $fleetRoot | Out-Null
New-Item -ItemType Directory -Force -Path $sampleRoot | Out-Null
New-Item -ItemType Directory -Force -Path $shardRoot | Out-Null

Write-Host "Generating fixed sample CSV locally"
& py -3 data_sampler.py `
    --sample-only `
    --samples-per-bin $SamplesPerBin `
    --patch-size $PatchSize `
    --output-folder $sampleRoot `
    --sampling-seed $SamplingSeed `
    --date-seed $DateSeed
if ($LASTEXITCODE -ne 0) {
    throw "Failed to generate fixed sample CSV"
}

$masterCsv = Join-Path $sampleRoot "sampled_locations.csv"

Write-Host "Splitting fixed sample into $ShardCount shards"
& py -3 scripts\split_sample_csv.py `
    --input-csv $masterCsv `
    --output-dir $shardRoot `
    --shards $ShardCount `
    --warmup-rows-per-shard $WarmupRowsPerShard
if ($LASTEXITCODE -ne 0) {
    throw "Failed to split sampled_locations.csv"
}

$sshArgs = @("-i", $KeyPath, "-o", "StrictHostKeyChecking=accept-new")
$securityGroupArgs = @()
foreach ($securityGroupId in $SecurityGroupIds) {
    $securityGroupArgs += $securityGroupId
}

$imageIdToUse = $ImageId
if (-not $imageIdToUse) {
    $imageIdToUse = Invoke-AwsCli @(
        "ec2", "describe-images",
        "--region", $Region,
        "--owners", "099720109477",
        "--filters", "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*", "Name=state,Values=available",
        "--query", "reverse(sort_by(Images,&CreationDate))[0].ImageId",
        "--output", "text"
    )
    if (-not $imageIdToUse -or $imageIdToUse -eq "None") {
        throw "Could not resolve latest Ubuntu AMI for $Region"
    }
}

$fleetRecords = @()

for ($shardNumber = 1; $shardNumber -le $ShardCount; $shardNumber++) {
    $shardId = "shard_{0:d2}" -f $shardNumber
    $instanceName = "$FleetName-$shardId"
    $shardCsv = Join-Path $shardRoot "$shardId.csv"
    $warmupCsv = Join-Path $shardRoot "${shardId}_warmup.csv"

    $blockDeviceMapping = "DeviceName=/dev/sda1,Ebs={VolumeSize=$VolumeSizeGiB,VolumeType=$VolumeType,DeleteOnTermination=false,Iops=$VolumeIops,Throughput=$VolumeThroughput}"
    $tagSpecification = "ResourceType=instance,Tags=[{Key=Name,Value=$instanceName},{Key=Fleet,Value=$FleetName},{Key=ShardId,Value=$shardId}]"
    $runArgs = @(
        "ec2", "run-instances",
        "--region", $Region,
        "--image-id", $imageIdToUse,
        "--instance-type", $InstanceType,
        "--key-name", $KeyName,
        "--block-device-mappings", $blockDeviceMapping,
        "--tag-specifications", $tagSpecification,
        "--query", "Instances[0].InstanceId",
        "--output", "text"
    )
    if ($SubnetId) {
        $runArgs += @("--subnet-id", $SubnetId)
    }
    if ($securityGroupArgs.Count -gt 0) {
        $runArgs += "--security-group-ids"
        $runArgs += $securityGroupArgs
    }

    $instanceId = Invoke-AwsCli $runArgs
    if (-not $instanceId -or $instanceId -eq "None") {
        throw "Failed to launch instance for $shardId"
    }

    Write-Host "Launched $instanceId for $shardId"
    Invoke-AwsCli @("ec2", "wait", "instance-running", "--region", $Region, "--instance-ids", $instanceId) | Out-Null
    Invoke-AwsCli @("ec2", "wait", "instance-status-ok", "--region", $Region, "--instance-ids", $instanceId) | Out-Null

    $hostName = Invoke-AwsCli @(
        "ec2", "describe-instances",
        "--region", $Region,
        "--instance-ids", $instanceId,
        "--query", "Reservations[0].Instances[0].PublicDnsName",
        "--output", "text"
    )
    if (-not $hostName -or $hostName -eq "None") {
        throw "Launched $instanceId but no public DNS was assigned"
    }

    $sshTarget = "${SshUser}@${hostName}"
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        & ssh @sshArgs -o ConnectTimeout=10 $sshTarget "echo SSH_OK" | Out-Null
        if ($LASTEXITCODE -eq 0) {
            break
        }
        Start-Sleep -Seconds 10
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Instance $instanceId launched but SSH never became reachable"
    }

    Invoke-Remote $sshTarget $sshArgs "mkdir -p $RemoteDir $RemoteDir/shards && sudo mkdir -p /data /data/bm_dmsp_cache /data/bm_dmsp_runs && sudo chown -R ${SshUser}:${SshUser} /data"

    Copy-ToRemote $sshArgs $sshTarget (Join-Path $repoRoot "data_sampler.py") $RemoteDir
    Copy-ToRemote $sshArgs $sshTarget (Join-Path $repoRoot "requirements.txt") $RemoteDir
    Copy-ToRemote $sshArgs $sshTarget (Join-Path $repoRoot "README.md") $RemoteDir
    Copy-ToRemote $sshArgs $sshTarget (Join-Path $repoRoot ".env") $RemoteDir
    Copy-ToRemote $sshArgs $sshTarget (Join-Path $repoRoot "aws") $RemoteDir
    Copy-ToRemote $sshArgs $sshTarget (Join-Path $repoRoot "Data") $RemoteDir
    Copy-ToRemote $sshArgs $sshTarget $shardCsv "$RemoteDir/shards/"
    Copy-ToRemote $sshArgs $sshTarget $warmupCsv "$RemoteDir/shards/"

    Invoke-Remote $sshTarget $sshArgs "cd $RemoteDir && bash aws/bootstrap_ubuntu.sh"

    $remoteCmd = @"
cd $RemoteDir &&
source .venv/bin/activate &&
nohup bash -lc 'cd $RemoteDir && source .venv/bin/activate && bash aws/run_fixed_shard_download.sh $shardId shards/$shardId.csv shards/${shardId}_warmup.csv /data/bm_dmsp_runs /data/bm_dmsp_cache $PatchSize $MaxWorkers' > /data/${shardId}.log 2>&1 < /dev/null &
echo STARTED_$shardId
"@
    Invoke-Remote $sshTarget $sshArgs $remoteCmd

    $fleetRecords += [pscustomobject]@{
        ShardId = $shardId
        InstanceId = $instanceId
        HostName = $hostName
        Region = $Region
        InstanceType = $InstanceType
        VolumeSizeGiB = $VolumeSizeGiB
        VolumeType = $VolumeType
        VolumeIops = $VolumeIops
        VolumeThroughput = $VolumeThroughput
        MasterSampleCsv = $masterCsv
        ShardCsv = $shardCsv
        WarmupCsv = $warmupCsv
        RemoteDir = $RemoteDir
    }
}

$fleetRecords | Export-Csv -NoTypeInformation -Path $metadataPath
Write-Host "Fleet metadata written to $metadataPath"
