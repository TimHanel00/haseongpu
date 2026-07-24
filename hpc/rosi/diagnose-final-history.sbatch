#!/usr/bin/env bash
#SBATCH --job-name=hase-history-check
#SBATCH --partition=gpu-a100,gpu-a100-haicu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/th168408/workspace/haseonpu-alpakatune-runs/final-a100-history-diagnostic-v2-20260724/slurm-%j.out

set -euo pipefail

module purge
module load gcc/14.2.0
module load cmake/4.0.3
module load cuda/12.8
module load python/3.12.4

readonly source_root=/home/th168408/workspace/haseonpu-alpakatune-forwardtet4-final-v2-20260724
readonly shared_root=/home/th168408/workspace/haseonpu-alpakatune-forwardtet4-20260724
readonly run_root=/home/th168408/workspace/haseonpu-alpakatune-runs/final-a100-history-diagnostic-v2-20260724
readonly output_dir=${run_root}/random_online
readonly history=${run_root}/history.json

test ! -e "${run_root}/started"
mkdir -p "${output_dir}"
touch "${run_root}/started"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export HASE_OPENPMD_PYTHONPATH="${shared_root}/build-baseline/hase-openpmd-provider/install/lib/python3.12/site-packages"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export HASE_CALCPHIASE="${source_root}/build-tuned/calcPhiASE"
export HASE_ALPAKATUNE_KERNELS=AccumulateForwardPhiAse
export ALPAKA_TUNE_CONFIG="${source_root}/hpc/rosi/configs/history_random_collect.yaml"
export HASE_ALPAKATUNE_HISTORY="${history}"
export HASE_ALPAKATUNE_TRACE="${output_dir}/trace.jsonl"
export HASE_ALPAKATUNE_METRICS="${output_dir}/metrics.jsonl"
unset HASE_ALPAKATUNE_MODEL HASE_ALPAKATUNE_BASELINE_ONLY

"${source_root}/.venv/bin/python3" "${source_root}/example/laserPumpCladding.py" \
    --backend Cuda_NvidiaGpu_GpuCuda \
    --timeSteps 5 \
    --pumpSteps 5 \
    --phi-ase-config "${source_root}/config/hase-phiase.yaml" \
    --openpmd-backend adios-sst \
    --rng-seed 5489 \
    --no-vtk \
    --timeout-seconds 300 \
    --summary-json "${output_dir}/summary.json" \
    >"${output_dir}/stdout.txt" 2>&1

test -s "${history}"
test -s "${output_dir}/trace.jsonl"
test -s "${output_dir}/metrics.jsonl"
"${source_root}/.venv/bin/python3" -c \
    'import json, pathlib, sys
history = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert history["schema_version"] == 1
assert history["contexts"]
assert all(context["configurations"] for context in history["contexts"].values())
for name in sys.argv[2:]:
    records = [json.loads(line) for line in pathlib.Path(name).read_text().splitlines() if line.strip()]
    assert records
    assert {record["kernel"] for record in records} == {"AccumulateForwardPhiAse"}' \
    "${history}" "${output_dir}/trace.jsonl" "${output_dir}/metrics.jsonl"
touch "${run_root}/completed"
