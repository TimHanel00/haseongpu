#!/usr/bin/env bash
#SBATCH --job-name=hase-two-kernel-analysis
#SBATCH --partition=cpu-genoa
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/th168408/workspace/haseonpu-alpakatune-runs/master-two-kernel-tuning-20260813-r1/analysis-%j.out

set -euo pipefail
module purge
module load python/3.12.4

readonly source_root=/home/th168408/workspace/haseonpu-master-two-kernel-tuning-20260813
readonly run_root=/home/th168408/workspace/haseonpu-alpakatune-runs/master-two-kernel-tuning-20260813-r1
readonly baseline_trace=/home/th168408/workspace/haseonpu-alpakatune-runs/master-all-kernel-timing-20260813-r1/raw/Cuda_NvidiaGpu_GpuCuda/timing_only/trace.jsonl
test -e "${run_root}/completed"
test -s "${baseline_trace}"
"${source_root}/.venv-two-kernel-tuning/bin/python3" \
    "${source_root}/scripts/analyze_two_kernel_tuning.py" "${run_root}" \
    --baseline-trace "${baseline_trace}"
touch "${run_root}/analysis-completed"
