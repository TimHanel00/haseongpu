#!/usr/bin/env bash
#SBATCH --job-name=hase-tune-a100
#SBATCH --array=0-8
#SBATCH --partition=gpu-a100,gpu-a100-haicu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/th168408/workspace/haseonpu-alpakatune-runs/forwardtet4-adaptive-gpu-20260723-v2/slurm-%A_%a.out

set -euo pipefail

module purge
module load gcc/14.2.0
module load cmake/4.0.3
module load cuda/12.8
module load python/3.12.4

readonly source_root=/home/th168408/workspace/haseonpu-alpakatune-forwardtet4-20260723
readonly run_root=/home/th168408/workspace/haseonpu-alpakatune-runs/forwardtet4-adaptive-gpu-20260723-v2
readonly model=/home/th168408/workspace/alpakaTune-ml-runs/experiment-01/artifacts/rosi/experiment-01/model.atml
readonly expected_model_sha=150ce293375a618a236a288e323cc513fc380748eef3a9b23c103855f916f06b

readonly modes=(baseline random learned_hybrid)
readonly block=$((SLURM_ARRAY_TASK_ID / 3))
readonly repetition=$((SLURM_ARRAY_TASK_ID % 3))
readonly backend=Cuda_NvidiaGpu_GpuCuda
readonly mode=${modes[${block}]}
readonly output_dir=${run_root}/raw/${backend}/${mode}/rep-${repetition}

mkdir -p "${output_dir}"
{
    printf 'job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'array_task_id=%s\n' "${SLURM_ARRAY_TASK_ID}"
    printf 'node=%s\n' "${SLURMD_NODENAME}"
    printf 'backend=%s\n' "${backend}"
    printf 'mode=%s\n' "${mode}"
    printf 'repetition=%s\n' "${repetition}"
    printf 'source_dirty_snapshot=true\n'
    cat "${source_root}/hpc/rosi/source-manifest.json"
    lscpu
    nvidia-smi -L
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
    module list
} >"${output_dir}/metadata.txt" 2>&1

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export HASE_OPENPMD_PYTHONPATH="${source_root}/build-baseline/hase-openpmd-provider/install/lib/python3.12/site-packages"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

if [[ "${mode}" == baseline ]]; then
    export HASE_CALCPHIASE="${source_root}/build-baseline/calcPhiASE"
    unset ALPAKA_TUNE_CONFIG HASE_ALPAKATUNE_HISTORY HASE_ALPAKATUNE_TRACE HASE_ALPAKATUNE_MODEL
else
    export HASE_CALCPHIASE="${source_root}/build-tuned/calcPhiASE"
    export ALPAKA_TUNE_CONFIG="${source_root}/hpc/rosi/configs/${mode}.yaml"
    export HASE_ALPAKATUNE_HISTORY="${output_dir}/history.json"
    export HASE_ALPAKATUNE_TRACE="${output_dir}/trace.jsonl"
    if [[ "${mode}" == learned_hybrid ]]; then
        actual_model_sha=$(sha256sum "${model}" | awk '{print $1}')
        if [[ "${actual_model_sha}" != "${expected_model_sha}" ]]; then
            printf 'Model digest mismatch: expected %s, got %s\n' "${expected_model_sha}" "${actual_model_sha}" >&2
            exit 1
        fi
        export HASE_ALPAKATUNE_MODEL="${model}"
    else
        unset HASE_ALPAKATUNE_MODEL
    fi
fi

set +e
/usr/bin/time -v -o "${output_dir}/time.txt" \
    "${source_root}/.venv/bin/python3" "${source_root}/example/laserPumpCladding.py" \
    --backend "${backend}" \
    --timeSteps 150 \
    --pumpSteps 100 \
    --phi-ase-config "${source_root}/config/hase-phiase.yaml" \
    --openpmd-backend adios-sst \
    --rng-seed 5489 \
    --no-vtk \
    --timeout-seconds 1800 \
    --summary-json "${output_dir}/summary.json" \
    >"${output_dir}/stdout.txt" 2>&1
exit_code=$?
set -e

printf '%s\n' "${exit_code}" >"${output_dir}/exit-code.txt"
if [[ "${exit_code}" -ne 0 && "${exit_code}" -ne 124 ]]; then
    exit "${exit_code}"
fi

if [[ "${mode}" != baseline ]]; then
    test -s "${output_dir}/trace.jsonl"
    if grep -q '"learned_status":"fallback_' "${output_dir}/trace.jsonl"; then
        printf 'A learned strategy fallback appeared in %s\n' "${output_dir}/trace.jsonl" >&2
        exit 1
    fi
fi
