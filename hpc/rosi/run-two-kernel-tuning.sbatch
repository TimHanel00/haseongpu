#!/usr/bin/env bash
#SBATCH --job-name=hase-two-kernel-tune
#SBATCH --partition=gpu-a100,gpu-a100-haicu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/th168408/workspace/haseonpu-alpakatune-runs/master-two-kernel-tuning-20260813-r1/slurm-%j.out

set -euo pipefail
module purge
module load gcc/14.2.0
module load cmake/4.0.3
module load cuda/12.8
module load python/3.12.4

readonly source_root=/home/th168408/workspace/haseonpu-master-two-kernel-tuning-20260813
readonly shared_root=/home/th168408/workspace/haseonpu-alpakatune-forwardtet4-20260724
readonly run_root=/home/th168408/workspace/haseonpu-alpakatune-runs/master-two-kernel-tuning-20260813-r1
readonly backend=Cuda_NvidiaGpu_GpuCuda
readonly model=/home/th168408/workspace/alpakaTune-ml-runs/experiment-01/artifacts/rosi/experiment-01/model.atml
readonly expected_model_sha=150ce293375a618a236a288e323cc513fc380748eef3a9b23c103855f916f06b
readonly evaluation_steps=2001
readonly collection_steps=50
readonly history_root=${run_root}/histories

test -e "${run_root}/build-completed"
mkdir -p "${history_root}"
touch "${run_root}/started"
actual_model_sha=$(sha256sum "${model}" | awk '{print $1}')
test "${actual_model_sha}" = "${expected_model_sha}"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export HASE_OPENPMD_PYTHONPATH=${shared_root}/build-baseline/hase-openpmd-provider/install/lib/python3.12/site-packages
export HASE_OPENPMD_SNAPSHOT_RETENTION=last
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export HASE_CALCPHIASE=${source_root}/build-two-kernel-tuning/calcPhiASE
export HASE_ALPAKATUNE_KERNELS=AccumulateForwardPhiAse,TraceGeneralPump

run_mode()
{
    local mode=$1
    local steps=$2
    local config=$3
    local compact_history=$4
    local complete_history=$5
    local use_model=$6
    local benchmark_only=$7
    local output_dir=${run_root}/raw/${backend}/${mode}
    mkdir -p "${output_dir}"
    if [[ -s "${output_dir}/trace.jsonl" && -s "${output_dir}/time.txt" ]]; then
        printf 'reusing validated completed mode: %s\n' "${mode}"
        return
    fi
    test ! -e "${output_dir}/trace.jsonl"

    export HASE_ALPAKATUNE_MODE=${mode}
    export HASE_ALPAKATUNE_TRACE=${output_dir}/trace.jsonl
    export HASE_OPENPMD_SNAPSHOT_INTERVAL=${steps}
    if [[ "${benchmark_only}" == 1 ]]; then
        export HASE_ALPAKATUNE_BENCHMARK_ONLY=1
        unset ALPAKA_TUNE_CONFIG HASE_ALPAKATUNE_HISTORY HASE_ALPAKATUNE_COMPLETE_HISTORY HASE_ALPAKATUNE_MODEL
    else
        unset HASE_ALPAKATUNE_BENCHMARK_ONLY
        export ALPAKA_TUNE_CONFIG=${config}
        export HASE_ALPAKATUNE_HISTORY=${compact_history}
        export HASE_ALPAKATUNE_COMPLETE_HISTORY=${complete_history}
        if [[ "${use_model}" == 1 ]]; then
            export HASE_ALPAKATUNE_MODEL=${model}
        else
            unset HASE_ALPAKATUNE_MODEL
        fi
    fi

    {
        printf 'job_id=%s\nnode=%s\nmode=%s\nsteps=%s\n' \
            "${SLURM_JOB_ID}" "${SLURMD_NODENAME}" "${mode}" "${steps}"
        printf 'hase_master_commit=51850a2f3904151954ab07a3b9a87491684318eb\n'
        printf 'alpaka_tune_commit=997974047a9e6c8a4d5ee1efc97705f64810b446\n'
        printf 'alpaka_commit=176dcc921041328f891d58e80614585e7848ef95\n'
        printf 'model_sha256=%s\nkernels=%s\n' "${actual_model_sha}" "${HASE_ALPAKATUNE_KERNELS}"
        printf 'candidate_extents=32,64,128,256,512\ncoverage_preserved=true\n'
        sha256sum \
            "${source_root}/include/alpakaUtils/TimedEnqueue.hpp" \
            "${source_root}/include/core/calcForwardPhiAse.hpp" \
            "${source_root}/include/kernels/generalPump.hpp" \
            "${source_root}/hpc/rosi/configs/two_kernel_offline_collect.yaml" \
            "${source_root}/hpc/rosi/configs/two_kernel_offline.yaml" \
            "${source_root}/hpc/rosi/configs/two_kernel_online_fixed.yaml" \
            "${source_root}/hpc/rosi/configs/two_kernel_online_adaptive.yaml" \
            "${HASE_CALCPHIASE}"
        grep -E '^(HASE_ENABLE_ALPAKATUNE|HASE_ALPAKATUNE_GIT_TAG|HASE_ALPAKA_GIT_TAG):' \
            "${source_root}/build-two-kernel-tuning/CMakeCache.txt"
        nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
    } >"${output_dir}/metadata.txt"

    /usr/bin/time -v -o "${output_dir}/time.txt" \
        "${source_root}/.venv-two-kernel-tuning/bin/python3" "${source_root}/example/laserPumpCladding.py" \
        --backend "${backend}" \
        --timeSteps "${steps}" \
        --pumpSteps "${steps}" \
        --output-steps "${steps}" \
        --phi-ase-config "${source_root}/config/hase-phiase.yaml" \
        --openpmd-backend adios-sst \
        --rng-seed 5489 \
        --pump-rng-seed 5489 \
        --vtk-output-dir "${output_dir}/vtk" \
        --timings >"${output_dir}/stdout.txt" 2>&1

    test -s "${output_dir}/trace.jsonl"
    test -s "${output_dir}/time.txt"
}

readonly offline_compact=${history_root}/offline-compact.json
readonly offline_complete=${history_root}/offline-complete.json
readonly fixed_compact=${history_root}/online-fixed-compact.json
readonly fixed_complete=${history_root}/online-fixed-complete.json
readonly adaptive_compact=${history_root}/online-adaptive-compact.json
readonly adaptive_complete=${history_root}/online-adaptive-complete.json

run_mode \
    offline_collect "${collection_steps}" \
    "${source_root}/hpc/rosi/configs/two_kernel_offline_collect.yaml" \
    "${offline_compact}" "${offline_complete}" 0 0

python3 - "${offline_complete}" <<'PY'
import json
import pathlib
import sys

history = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert history["schema_version"] == 11
contexts = list(history["contexts"].values())
assert len(contexts) == 6
assert {context["metadata"]["kernel"].rsplit("::", 1)[-1] for context in contexts} == {
    "AccumulateForwardPhiAse",
    "TraceGeneralPump",
}
for context in contexts:
    assert context["candidate_count"] == 25
    measured = [samples for samples in context["candidate_samples"] if samples]
    assert len(measured) == 5
    assert {len(samples) for samples in measured} == {9}
    assert context["policy"]["queue"] is None
print("validated exhaustive history for six contexts with five candidates and nine samples each")
PY

run_mode \
    offline "${evaluation_steps}" \
    "${source_root}/hpc/rosi/configs/two_kernel_offline.yaml" \
    "${offline_compact}" "${offline_complete}" 0 0

cp "${offline_compact}" "${fixed_compact}"
cp "${offline_complete}" "${fixed_complete}"
run_mode \
    online_fixed "${evaluation_steps}" \
    "${source_root}/hpc/rosi/configs/two_kernel_online_fixed.yaml" \
    "${fixed_compact}" "${fixed_complete}" 1 0

cp "${fixed_compact}" "${adaptive_compact}"
cp "${fixed_complete}" "${adaptive_complete}"
run_mode \
    online_adaptive "${evaluation_steps}" \
    "${source_root}/hpc/rosi/configs/two_kernel_online_adaptive.yaml" \
    "${adaptive_compact}" "${adaptive_complete}" 1 0

touch "${run_root}/completed"
