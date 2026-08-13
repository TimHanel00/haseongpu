#!/usr/bin/env bash
#SBATCH --job-name=hase-time-build
#SBATCH --partition=cpu-genoa
#SBATCH --nodes=1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/home/th168408/workspace/haseonpu-alpakatune-runs/master-timing-history-20260813-r1/build-%j.out

set -euo pipefail
readonly source_root=/home/th168408/workspace/haseonpu-master-timing-history-20260813
readonly run_root=/home/th168408/workspace/haseonpu-alpakatune-runs/master-timing-history-20260813-r1

test ! -e "${run_root}/build-completed"
export CMAKE_BUILD_PARALLEL_LEVEL=${SLURM_CPUS_PER_TASK}
bash "${source_root}/hpc/rosi/configure-timing-history-build.sh"
touch "${run_root}/build-completed"
