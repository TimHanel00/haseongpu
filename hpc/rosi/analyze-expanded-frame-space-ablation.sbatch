#!/usr/bin/env bash
#SBATCH --job-name=hase-expanded-analysis
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/th168408/workspace/haseonpu-alpakatune-runs/expanded-frame-space-ablation-20260727/analysis-%j.out

set -euo pipefail

module purge
module load python/3.12.4

readonly source_root=/home/th168408/workspace/haseonpu-alpakatune-expanded-frame-space-20260727
readonly run_root=/home/th168408/workspace/haseonpu-alpakatune-runs/expanded-frame-space-ablation-20260727

test -e "${run_root}/completed"
"${source_root}/.venv/bin/python3" \
    "${source_root}/scripts/analyze_adaptive_horizon_ablation.py" \
    --run-root "${run_root}"
touch "${run_root}/analysis-completed"
