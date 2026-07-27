#!/usr/bin/env bash
#SBATCH --job-name=hase-history-analysis
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/th168408/workspace/haseonpu-alpakatune-runs/forwardtet4-history-horizon-retries-gpu-20260724/analysis-slurm-%j.out

set -euo pipefail

module purge
module load python/3.12.4

readonly source_root=/home/th168408/workspace/haseonpu-alpakatune-forwardtet4-20260724
readonly run_root=/home/th168408/workspace/haseonpu-alpakatune-runs/forwardtet4-history-horizon-retries-gpu-20260724

"${source_root}/.venv/bin/python3" \
    "${source_root}/scripts/analyze_alpakatune_history_horizon.py" \
    --run-root "${run_root}"
