#!/usr/bin/env bash

# Run the complete production sweep, retain a coarse hardware trace, and only
# replace article analysis after every Hydra job has completed successfully.
set -Eeuo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

started_at=$(date -u +%Y-%m-%dT%H-%M-%SZ)
staging_dir="results/.running-scenario-decoder-${started_at}"
mkdir -p "$staging_dir"
run_log="$staging_dir/sweep.log"
hardware_log="$staging_dir/hardware_usage.csv"
hardware_summary="$staging_dir/hardware_summary.txt"

printf 'timestamp_utc,gpu_name,gpu_memory_total_mib,gpu_memory_used_mib,gpu_utilization_pct,gpu_memory_utilization_pct,gpu_temperature_c,gpu_power_w,ram_total_mib,ram_used_mib,ram_available_mib,load_1m,disk_available_gib\n' > "$hardware_log"
{
    date -u --iso-8601=seconds
    lscpu
    nvidia-smi -q
    free -h
    df -h .
} > "$hardware_summary"

sample_hardware() {
    local timestamp gpu ram disk
    timestamp=$(date -u --iso-8601=seconds)
    gpu=$(nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits | head -n 1 | sed 's/, */,/g')
    ram=$(free -m | awk '/^Mem:/ {print $2 "," $3 "," $7}')
    disk=$(df -BG --output=avail . | awk 'NR == 2 {gsub(/G/, "", $1); print $1}')
    printf '%s,%s,%s,%s,%s\n' "$timestamp" "$gpu" "$ram" "$(awk '{print $1}' /proc/loadavg)" "$disk" >> "$hardware_log"
}

make sweep-scenario-decoder > "$run_log" 2>&1 &
sweep_pid=$!
sample_hardware
while kill -0 "$sweep_pid" 2>/dev/null; do
    sleep 30
    sample_hardware
done

if ! wait "$sweep_pid"; then
    printf 'Sweep failed; logs retained in %s\n' "$staging_dir" >&2
    exit 1
fi
sample_hardware

result_config=$(find results -type f -path '*/.hydra/config.yaml' -newer "$run_log" -print | sort | tail -n 1)
if [[ -z "$result_config" ]]; then
    printf 'Could not locate the completed Hydra result directory; logs retained in %s\n' "$staging_dir" >&2
    exit 1
fi
result_dir=$(dirname "$(dirname "$result_config")")
mv "$hardware_log" "$hardware_summary" "$run_log" "$result_dir/"
rmdir "$staging_dir"

ANALYZE_DIR="$result_dir" make analysis > "$result_dir/analysis.log" 2>&1
printf 'Completed sweep and refreshed analysis from %s\n' "$result_dir"
