#!/usr/bin/env bash
#SBATCH --job-name=hase-tune-analysis
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/th168408/workspace/haseonpu-alpakatune-runs/forwardtet4-adaptive-gpu-20260723-v2/analysis-slurm-%j.out

set -euo pipefail

module purge
module load python/3.12.4

readonly source_root=/home/th168408/workspace/haseonpu-alpakatune-forwardtet4-20260723
readonly run_root=/home/th168408/workspace/haseonpu-alpakatune-runs/forwardtet4-adaptive-gpu-20260723-v2

"${source_root}/.venv/bin/python3" "${source_root}/scripts/analyze_alpakatune_campaign.py" \
    --run-root "${run_root}"
