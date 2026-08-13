#!/usr/bin/env bash
#SBATCH --job-name=hase-device-time-analysis
#SBATCH --partition=cpu-genoa
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/th168408/workspace/haseonpu-alpakatune-runs/master-device-timing-isolation-20260813-r1/analysis-%j.out

set -euo pipefail
module purge
module load python/3.12.4

readonly source_root=/home/th168408/workspace/haseonpu-master-device-timing-isolation-20260813
readonly run_root=/home/th168408/workspace/haseonpu-alpakatune-runs/master-device-timing-isolation-20260813-r1
test -e "${run_root}/completed"
"${source_root}/.venv-device-timing-isolation/bin/python3" \
    "${source_root}/scripts/analyze_device_timing_isolation.py" "${run_root}"
touch "${run_root}/analysis-completed"
