#!/usr/bin/env bash
#SBATCH --job-name=hase-final-analysis
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/th168408/workspace/haseonpu-alpakatune-runs/final-a100-adaptive-offline-20260724/analysis-%j.out

set -euo pipefail

module purge
module load python/3.12.4

readonly source_root=/home/th168408/workspace/haseonpu-alpakatune-forwardtet4-final-20260724
readonly run_root=/home/th168408/workspace/haseonpu-alpakatune-runs/final-a100-adaptive-offline-20260724

test -f "${run_root}/completed"
"${source_root}/.venv/bin/python3" \
    "${source_root}/scripts/analyze_alpakatune_final.py" \
    --run-root "${run_root}"
touch "${run_root}/analyzed"
