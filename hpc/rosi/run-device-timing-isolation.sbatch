#!/usr/bin/env bash
#SBATCH --job-name=hase-device-time-run
#SBATCH --partition=gpu-a100,gpu-a100-haicu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/th168408/workspace/haseonpu-alpakatune-runs/master-device-timing-isolation-20260813-r1/slurm-%j.out

set -euo pipefail
module purge
module load gcc/14.2.0
module load cmake/4.0.3
module load cuda/12.8
module load python/3.12.4

readonly source_root=/home/th168408/workspace/haseonpu-master-device-timing-isolation-20260813
readonly tuner_root=/home/th168408/workspace/haseonpu-master-two-kernel-tuning-20260813
readonly shared_root=/home/th168408/workspace/haseonpu-alpakatune-forwardtet4-20260724
readonly run_root=/home/th168408/workspace/haseonpu-alpakatune-runs/master-device-timing-isolation-20260813-r1
readonly backend=Cuda_NvidiaGpu_GpuCuda
readonly steps=2001
readonly direct_binary=${source_root}/build-device-timing-isolation/calcPhiASE
readonly tuner_binary=${tuner_root}/build-two-kernel-tuning/calcPhiASE

test -e "${run_root}/build-completed"
test ! -e "${run_root}/started"
test -x "${direct_binary}"
test -x "${tuner_binary}"
test ! -e "${source_root}/dependencies/alpakaTune"
touch "${run_root}/started"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export HASE_OPENPMD_PYTHONPATH=${shared_root}/build-baseline/hase-openpmd-provider/install/lib/python3.12/site-packages
export HASE_OPENPMD_SNAPSHOT_RETENTION=last
export HASE_OPENPMD_SNAPSHOT_INTERVAL=${steps}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

run_mode()
{
    local mode=$1
    local binary=$2
    local frame_specs=$3
    local output_dir=${run_root}/raw/${backend}/${mode}
    mkdir -p "${output_dir}"

    export HASE_CALCPHIASE=${binary}
    if [[ "${mode}" == current_timing_default ]]; then
        export HASE_ALPAKATUNE_KERNELS=AccumulateForwardPhiAse,TraceGeneralPump
        export HASE_ALPAKATUNE_BENCHMARK_ONLY=1
        export HASE_ALPAKATUNE_MODE=${mode}
        export HASE_ALPAKATUNE_TRACE=${output_dir}/trace.jsonl
        unset ALPAKA_TUNE_CONFIG HASE_ALPAKATUNE_HISTORY HASE_ALPAKATUNE_COMPLETE_HISTORY HASE_ALPAKATUNE_MODEL
        unset HASE_DEVICE_TIMING_KERNELS HASE_DEVICE_TIMING_FRAME_SPECS HASE_DEVICE_TIMING_MODE HASE_DEVICE_TIMING_CSV
    else
        export HASE_DEVICE_TIMING_KERNELS=AccumulateForwardPhiAse,TraceGeneralPump
        export HASE_DEVICE_TIMING_FRAME_SPECS=${frame_specs}
        export HASE_DEVICE_TIMING_MODE=${mode}
        export HASE_DEVICE_TIMING_CSV=${output_dir}/timings.csv
        unset HASE_ALPAKATUNE_KERNELS HASE_ALPAKATUNE_BENCHMARK_ONLY HASE_ALPAKATUNE_MODE HASE_ALPAKATUNE_TRACE
        unset ALPAKA_TUNE_CONFIG HASE_ALPAKATUNE_HISTORY HASE_ALPAKATUNE_COMPLETE_HISTORY HASE_ALPAKATUNE_MODEL
    fi

    {
        printf 'job_id=%s\nnode=%s\nmode=%s\nsteps=%s\n' \
            "${SLURM_JOB_ID}" "${SLURMD_NODENAME}" "${mode}" "${steps}"
        printf 'hase_master_commit=51850a2f3904151954ab07a3b9a87491684318eb\n'
        printf 'alpaka_commit=176dcc921041328f891d58e80614585e7848ef95\n'
        printf 'alpaka_tune_linked=%s\nframe_specs=%s\n' \
            "$([[ "${mode}" == current_timing_default ]] && printf true || printf false)" "${frame_specs}"
        sha256sum "${binary}" "${source_root}/include/alpakaUtils/TimedEnqueue.hpp"
        grep -E '^(HASE_ENABLE_ALPAKATUNE|HASE_ENABLE_DEVICE_TIMING|HASE_ALPAKATUNE_GIT_TAG|HASE_ALPAKA_GIT_TAG):' \
            "$(dirname "${binary}")/CMakeCache.txt"
        nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
    } >"${output_dir}/metadata.txt"

    /usr/bin/time -v -o "${output_dir}/time.txt" \
        "${source_root}/.venv-device-timing-isolation/bin/python3" "${source_root}/example/laserPumpCladding.py" \
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

    test -s "${output_dir}/time.txt"
    if [[ "${mode}" == current_timing_default ]]; then
        test -s "${output_dir}/trace.jsonl"
    else
        test -s "${output_dir}/timings.csv"
    fi
}

run_mode current_timing_default "${tuner_binary}" default
run_mode no_tuner_selected "${direct_binary}" selected
run_mode no_tuner_default "${direct_binary}" default

touch "${run_root}/completed"
