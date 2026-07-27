#!/usr/bin/env bash
#SBATCH --job-name=hase-history-horizon
#SBATCH --partition=gpu-a100,gpu-a100-haicu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/th168408/workspace/haseonpu-alpakatune-runs/forwardtet4-history-horizon-retries-gpu-20260724/slurm-%j.out

set -euo pipefail

module purge
module load gcc/14.2.0
module load cmake/4.0.3
module load cuda/12.8
module load python/3.12.4

readonly source_root=/home/th168408/workspace/haseonpu-alpakatune-forwardtet4-20260724
readonly run_root=/home/th168408/workspace/haseonpu-alpakatune-runs/forwardtet4-history-horizon-retries-gpu-20260724
readonly backend=Cuda_NvidiaGpu_GpuCuda
readonly model=/home/th168408/workspace/alpakaTune-ml-runs/experiment-01/artifacts/rosi/experiment-01/model.atml
readonly expected_model_sha=150ce293375a618a236a288e323cc513fc380748eef3a9b23c103855f916f06b
readonly learned_history=${run_root}/histories/learned.json
readonly random_history=${run_root}/histories/random.json
readonly modes=(baseline learned_collect learned_resume random_collect offline_replay)

test ! -e "${run_root}/started"
mkdir -p "${run_root}/histories"
touch "${run_root}/started"

actual_model_sha=$(sha256sum "${model}" | awk '{print $1}')
if [[ "${actual_model_sha}" != "${expected_model_sha}" ]]; then
    printf 'Model digest mismatch: expected %s, got %s\n' \
        "${expected_model_sha}" "${actual_model_sha}" >&2
    exit 1
fi

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export HASE_OPENPMD_PYTHONPATH="${source_root}/build-baseline/hase-openpmd-provider/install/lib/python3.12/site-packages"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export HASE_ALPAKATUNE_KERNELS=AccumulateForwardPhiAse

for mode in "${modes[@]}"; do
    output_dir=${run_root}/raw/${backend}/${mode}
    mkdir -p "${output_dir}"
    {
        printf 'job_id=%s\n' "${SLURM_JOB_ID}"
        printf 'node=%s\n' "${SLURMD_NODENAME}"
        printf 'backend=%s\n' "${backend}"
        printf 'mode=%s\n' "${mode}"
        printf 'alpaka_tune_commit=175169bb83cba557a13c1228266bc8db7c1b1824\n'
        printf 'model_sha256=%s\n' "${actual_model_sha}"
        cat "${source_root}/hpc/rosi/history-horizon-manifest.json"
        nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
        module list
    } >"${output_dir}/metadata.txt" 2>&1

    export HASE_STEP_TIMINGS_JSONL="${output_dir}/step-timings.jsonl"
    if [[ "${mode}" == baseline ]]; then
        export HASE_CALCPHIASE="${source_root}/build-baseline/calcPhiASE"
        unset ALPAKA_TUNE_CONFIG HASE_ALPAKATUNE_HISTORY
        unset HASE_ALPAKATUNE_TRACE HASE_ALPAKATUNE_METRICS
        unset HASE_ALPAKATUNE_MODEL HASE_ALPAKATUNE_BASELINE_ONLY
    else
        export HASE_CALCPHIASE="${source_root}/build-tuned/calcPhiASE"
        export ALPAKA_TUNE_CONFIG="${source_root}/hpc/rosi/configs/history_${mode}.yaml"
        export HASE_ALPAKATUNE_TRACE="${output_dir}/trace.jsonl"
        export HASE_ALPAKATUNE_METRICS="${output_dir}/metrics.jsonl"
        unset HASE_ALPAKATUNE_BASELINE_ONLY

        case "${mode}" in
            learned_collect|learned_resume)
                export HASE_ALPAKATUNE_HISTORY="${learned_history}"
                export HASE_ALPAKATUNE_MODEL="${model}"
                ;;
            random_collect|offline_replay)
                export HASE_ALPAKATUNE_HISTORY="${random_history}"
                unset HASE_ALPAKATUNE_MODEL
                ;;
        esac
    fi

    set +e
    /usr/bin/time -v -o "${output_dir}/time.txt" \
        "${source_root}/.venv/bin/python3" "${source_root}/example/laserPumpCladding.py" \
        --backend "${backend}" \
        --timeSteps 1000 \
        --pumpSteps 1000 \
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
    if [[ "${exit_code}" -ne 0 ]]; then
        exit "${exit_code}"
    fi

    if [[ "${mode}" != baseline ]]; then
        test -s "${output_dir}/trace.jsonl"
        test -s "${output_dir}/metrics.jsonl"
        if grep -q '"learned_status":"fallback_' "${output_dir}/trace.jsonl"; then
            printf 'A learned strategy fallback appeared in %s\n' \
                "${output_dir}/trace.jsonl" >&2
            exit 1
        fi
    fi

    case "${mode}" in
        learned_collect)
            test -s "${learned_history}"
            sha256sum "${learned_history}" >"${run_root}/learned-before-resume.sha256"
            ;;
        learned_resume)
            sha256sum "${learned_history}" >"${run_root}/learned-after-resume.sha256"
            test "$(cut -d' ' -f1 "${run_root}/learned-before-resume.sha256")" = \
                 "$(cut -d' ' -f1 "${run_root}/learned-after-resume.sha256")"
            grep -q '"loaded_from_cache":true' "${output_dir}/trace.jsonl"
            ;;
        random_collect)
            test -s "${random_history}"
            sha256sum "${random_history}" >"${run_root}/random-before-offline.sha256"
            ;;
        offline_replay)
            sha256sum "${random_history}" >"${run_root}/random-after-offline.sha256"
            test "$(cut -d' ' -f1 "${run_root}/random-before-offline.sha256")" = \
                 "$(cut -d' ' -f1 "${run_root}/random-after-offline.sha256")"
            grep -q '"loaded_from_cache":true' "${output_dir}/trace.jsonl"
            ;;
    esac
done

touch "${run_root}/completed"
