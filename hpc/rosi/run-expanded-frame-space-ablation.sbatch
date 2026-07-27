#!/usr/bin/env bash
#SBATCH --job-name=hase-expanded-space
#SBATCH --partition=gpu-a100,gpu-a100-haicu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/th168408/workspace/haseonpu-alpakatune-runs/expanded-frame-space-ablation-20260727/slurm-%j.out

set -euo pipefail

module purge
module load gcc/14.2.0
module load cmake/4.0.3
module load cuda/12.8
module load python/3.12.4

readonly source_root=/home/th168408/workspace/haseonpu-alpakatune-expanded-frame-space-20260727
readonly shared_root=/home/th168408/workspace/haseonpu-alpakatune-forwardtet4-20260724
readonly run_root=/home/th168408/workspace/haseonpu-alpakatune-runs/expanded-frame-space-ablation-20260727
readonly backend=Cuda_NvidiaGpu_GpuCuda
readonly model=/home/th168408/workspace/alpakaTune-ml-runs/experiment-01/artifacts/rosi/experiment-01/model.atml
readonly expected_model_sha=150ce293375a618a236a288e323cc513fc380748eef3a9b23c103855f916f06b
readonly modes=(
    baseline_0
    random_horizon
    learned_continuous
    learned_horizon
    random_continuous
    baseline_1
)

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
export HASE_OPENPMD_PYTHONPATH="${shared_root}/build-baseline/hase-openpmd-provider/install/lib/python3.12/site-packages"
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
        printf 'hase_snapshot_commit=da949eda074e0007ab230ebd513685ffdfc59fd6\n'
        printf 'alpaka_tune_commit=58c2e0e4fce1e0e7d1860c8a9504ec9f6b06e448\n'
        printf 'model_sha256=%s\n' "${actual_model_sha}"
        printf 'instrumented_kernel=AccumulateForwardPhiAse\n'
        printf 'frame_extents=32,64,128,256,512,1024\n'
        printf 'num_frames_upper_multiplier=4\n'
        nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
        module list
    } >"${output_dir}/metadata.txt" 2>&1

    export HASE_STEP_TIMINGS_JSONL="${output_dir}/step-timings.jsonl"
    if [[ "${mode}" == baseline_* ]]; then
        export HASE_CALCPHIASE="${source_root}/build-baseline/calcPhiASE"
        unset ALPAKA_TUNE_CONFIG HASE_ALPAKATUNE_HISTORY
        unset HASE_ALPAKATUNE_TRACE HASE_ALPAKATUNE_METRICS
        unset HASE_ALPAKATUNE_MODEL HASE_ALPAKATUNE_BASELINE_ONLY
    else
        export HASE_CALCPHIASE="${source_root}/build-tuned/calcPhiASE"
        export ALPAKA_TUNE_CONFIG="${source_root}/hpc/rosi/configs/ablation_${mode}.yaml"
        export HASE_ALPAKATUNE_HISTORY="${run_root}/histories/${mode}.json"
        export HASE_ALPAKATUNE_TRACE="${output_dir}/trace.jsonl"
        export HASE_ALPAKATUNE_METRICS="${output_dir}/metrics.jsonl"
        unset HASE_ALPAKATUNE_BASELINE_ONLY
        if [[ "${mode}" == learned_* ]]; then
            export HASE_ALPAKATUNE_MODEL="${model}"
        else
            unset HASE_ALPAKATUNE_MODEL
        fi
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
        --timeout-seconds 3000 \
        --summary-json "${output_dir}/summary.json" \
        >"${output_dir}/stdout.txt" 2>&1
    exit_code=$?
    set -e
    printf '%s\n' "${exit_code}" >"${output_dir}/exit-code.txt"
    if [[ "${exit_code}" -ne 0 ]]; then
        exit "${exit_code}"
    fi

    if [[ "${mode}" != baseline_* ]]; then
        test -s "${output_dir}/trace.jsonl"
        test -s "${output_dir}/metrics.jsonl"
        test -s "${run_root}/histories/${mode}.json"
        if grep -q '"learned_status":"fallback_' "${output_dir}/trace.jsonl"; then
            printf 'A learned strategy fallback appeared in %s\n' \
                "${output_dir}/trace.jsonl" >&2
            exit 1
        fi
        "${source_root}/.venv/bin/python3" -c \
            'import json, pathlib, sys
records = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line.strip()]
assert records
assert {record["kernel"] for record in records} == {"AccumulateForwardPhiAse"}
assert {record["candidate_count"] for record in records} == {72, 84, 108, 126, 144}
assert {record["rejected_candidate_count"] for record in records} == {0}' \
            "${output_dir}/trace.jsonl"
    fi
done

touch "${run_root}/completed"
