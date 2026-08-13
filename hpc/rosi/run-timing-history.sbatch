#!/usr/bin/env bash
#SBATCH --job-name=hase-all-kernel-time
#SBATCH --partition=gpu-a100,gpu-a100-haicu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/th168408/workspace/haseonpu-alpakatune-runs/master-all-kernel-timing-20260813-r1/slurm-%j.out

set -euo pipefail
module purge
module load gcc/14.2.0
module load cmake/4.0.3
module load cuda/12.8
module load python/3.12.4

readonly source_root=/home/th168408/workspace/haseonpu-master-all-kernel-timing-20260813
readonly shared_root=/home/th168408/workspace/haseonpu-alpakatune-forwardtet4-20260724
readonly run_root=/home/th168408/workspace/haseonpu-alpakatune-runs/master-all-kernel-timing-20260813-r1
readonly output_dir=${run_root}/raw/Cuda_NvidiaGpu_GpuCuda/timing_only
readonly rounds=2000

test -e "${run_root}/build-completed"
test ! -e "${run_root}/started"
mkdir -p "${output_dir}/vtk"
touch "${run_root}/started"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export LC_ALL=C
export HASE_OPENPMD_PYTHONPATH=${shared_root}/build-baseline/hase-openpmd-provider/install/lib/python3.12/site-packages
export HASE_CALCPHIASE=${source_root}/build-timing-history/calcPhiASE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export ALPAKA_TUNE_CONFIG=${source_root}/hpc/rosi/configs/timing_full_history.yaml
export HASE_ALPAKATUNE_HISTORY=${output_dir}/compact-history.json
export HASE_ALPAKATUNE_COMPLETE_HISTORY=${output_dir}/full-history.json
export HASE_ALPAKATUNE_TRACE=${output_dir}/trace.jsonl

unset HASE_ALPAKATUNE_MODEL HASE_ALPAKATUNE_BASELINE_ONLY HASE_ALPAKATUNE_KERNELS

{
    printf 'job_id=%s\nnode=%s\nrounds=%s\n' "${SLURM_JOB_ID}" "${SLURMD_NODENAME}" "${rounds}"
    printf 'hase_master_commit=51850a2f3904151954ab07a3b9a87491684318eb\n'
    printf 'alpaka_tune_commit=997974047a9e6c8a4d5ee1efc97705f64810b446\n'
    printf 'alpaka_commit=176dcc921041328f891d58e80614585e7848ef95\n'
    printf 'instrumented_scope=all_explicit_hase_kernel_launches\nactive_tuning=false\n'
    printf 'tunable_parameter_count=0\nframe_spec=application_default_unchanged\n'
    sha256sum \
        "${source_root}/include/alpakaUtils/TimedEnqueue.hpp" \
        "${source_root}/include/core/calcForwardPhiAse.hpp" \
        "${source_root}/include/core/forwardSrm.hpp" \
        "${source_root}/include/core/timeSteppedSimulation.hpp" \
        "${source_root}/include/kernels/activePointMasks.hpp" \
        "${source_root}/include/kernels/derivativeComposition.hpp" \
        "${source_root}/include/kernels/forwardPhiAseMapping.hpp" \
        "${source_root}/include/kernels/generalPump.hpp" \
        "${source_root}/hpc/rosi/configs/timing_full_history.yaml" \
        "${HASE_CALCPHIASE}"
    grep -E '^(HASE_ENABLE_ALPAKATUNE|HASE_ALPAKATUNE_GIT_TAG|HASE_ALPAKA_GIT_TAG):' \
        "${source_root}/build-timing-history/CMakeCache.txt"
    ldd "${HASE_CALCPHIASE}"
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
} >"${output_dir}/metadata.txt"

/usr/bin/time -v -o "${output_dir}/time.txt" \
    "${source_root}/.venv-timing/bin/python3" "${source_root}/example/laserPumpCladding.py" \
    --backend Cuda_NvidiaGpu_GpuCuda \
    --timeSteps "${rounds}" \
    --pumpSteps "${rounds}" \
    --phi-ase-config "${source_root}/config/hase-phiase.yaml" \
    --openpmd-backend adios-sst \
    --rng-seed 5489 \
    --pump-rng-seed 5489 \
    --vtk-output-dir "${output_dir}/vtk" \
    --timings >"${output_dir}/stdout.txt" 2>&1

test -s "${output_dir}/compact-history.json"
test -s "${output_dir}/full-history.json"
test -s "${output_dir}/trace.jsonl"
test -s "${output_dir}/time.txt"

python3 - "${output_dir}/full-history.json" "${output_dir}/trace.jsonl" <<'PY'
import json
import pathlib
import sys

history = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert history["schema_version"] == 11
contexts = list(history["contexts"].values())
assert contexts
sample_count = 0
kernel_names = set()
for context in contexts:
    assert context["candidate_count"] == 1
    assert context["metadata"]["mode"] == "online_adaptive"
    assert context["policy"]["queue"] is None
    assert context["candidate_configurations"] == [{}]
    assert context["metadata"]["model_context"]["dimensions"] == []
    kernel_names.add(context["metadata"]["kernel"])
    samples = context["candidate_samples"]
    assert len(samples) == 1 and samples[0]
    sample_count += len(samples[0])
assert sample_count >= 15000
assert len(kernel_names) >= 8

trace = [json.loads(line) for line in pathlib.Path(sys.argv[2]).read_text().splitlines() if line]
assert len(trace) == sample_count
assert all(row["measured"] for row in trace)
assert all(row["runtime_measurement_source"] == "device_event" for row in trace)
assert all(row["candidate_count"] == 1 for row in trace)
assert all(row["tunable_parameter_count"] == 0 for row in trace)
assert all(not row["active_queue"] for row in trace)
assert {row["kernel"] for row in trace} == {name.rsplit("::", 1)[-1] for name in kernel_names}
print(
    f"validated {sample_count} device-event samples across "
    f"{len(kernel_names)} kernels and {len(contexts)} fixed-launch contexts"
)
PY

touch "${run_root}/completed"
