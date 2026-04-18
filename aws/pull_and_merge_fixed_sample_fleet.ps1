param(
    [Parameter(Mandatory = $true)]
    [string]$InstancesCsv,
    [Parameter(Mandatory = $true)]
    [string]$KeyPath,
    [string]$SshUser = "ubuntu"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$downloadRoot = Join-Path (Split-Path -Parent $InstancesCsv) "downloaded_shards"
$mergedRoot = Join-Path (Split-Path -Parent $InstancesCsv) "merged_dataset"
$sshArgs = @("-i", $KeyPath, "-o", "StrictHostKeyChecking=accept-new")

if (Test-Path $downloadRoot) {
    Remove-Item -LiteralPath $downloadRoot -Recurse -Force
}
if (Test-Path $mergedRoot) {
    Remove-Item -LiteralPath $mergedRoot -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $mergedRoot "bm") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $mergedRoot "dmsp") | Out-Null

$instances = Import-Csv $InstancesCsv
if (-not $instances) {
    throw "No instance metadata found in $InstancesCsv"
}

foreach ($instance in $instances) {
    $shardId = $instance.ShardId
    $hostName = $instance.HostName
    $sshTarget = "${SshUser}@${hostName}"
    $remoteShardDir = "/data/bm_dmsp_runs/$shardId"

    Write-Host "Pulling $shardId from $hostName"
    & scp @sshArgs -r "${sshTarget}:$remoteShardDir" $downloadRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to download $shardId from $hostName"
    }
}

$mergeScript = @'
from pathlib import Path
import shutil
import pandas as pd

download_root = Path(r"__DOWNLOAD_ROOT__")
merged_root = Path(r"__MERGED_ROOT__")
bm_out = merged_root / "bm"
dmsp_out = merged_root / "dmsp"

manifests = []
samples = []

for shard_dir in sorted(download_root.iterdir()):
    full_dir = shard_dir / "full"
    if not full_dir.exists():
        continue

    sample_csv = full_dir / "sampled_locations.csv"
    if sample_csv.exists():
        sample_df = pd.read_csv(sample_csv)
        sample_df.insert(0, "shard_id", shard_dir.name)
        samples.append(sample_df)

    manifest_csv = full_dir / "bm_dmsp_pairs.csv"
    if not manifest_csv.exists():
        continue

    manifest_df = pd.read_csv(manifest_csv)
    if manifest_df.empty:
        continue

    manifest_df.insert(0, "shard_id", shard_dir.name)
    for row in manifest_df.itertuples(index=False):
        tile_name = Path(row.bm_patch).name
        dmsp_name = Path(row.dmsp_patch).name
        target_bm_name = f"{shard_dir.name}_{tile_name}"
        target_dmsp_name = f"{shard_dir.name}_{dmsp_name}"
        bm_target = bm_out / target_bm_name
        dmsp_target = dmsp_out / target_dmsp_name
        shutil.copy2(Path(row.bm_patch), bm_target)
        shutil.copy2(Path(row.dmsp_patch), dmsp_target)
        manifest_df.loc[manifest_df["bm_patch"] == row.bm_patch, "bm_patch"] = str(bm_target)
        manifest_df.loc[manifest_df["dmsp_patch"] == row.dmsp_patch, "dmsp_patch"] = str(dmsp_target)
    manifests.append(manifest_df)

merged_manifest = pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame()
merged_samples = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()

merged_manifest.to_csv(merged_root / "bm_dmsp_pairs.csv", index=False)
merged_samples.to_csv(merged_root / "sampled_locations.csv", index=False)
print(f"merged_manifest_rows={len(merged_manifest)}")
print(f"merged_sample_rows={len(merged_samples)}")
'@

$mergeScript = $mergeScript.Replace("__DOWNLOAD_ROOT__", ($downloadRoot -replace '\\', '\\'))
$mergeScript = $mergeScript.Replace("__MERGED_ROOT__", ($mergedRoot -replace '\\', '\\'))
$mergeScript | py -3 -
if ($LASTEXITCODE -ne 0) {
    throw "Failed to merge shard outputs"
}

Write-Host "Merged dataset written to $mergedRoot"
