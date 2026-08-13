#!/usr/bin/env bash
set -euo pipefail

readonly source_root=/home/th168408/workspace/haseonpu-master-two-kernel-tuning-20260813
readonly run_root=/home/th168408/workspace/haseonpu-alpakatune-runs/master-two-kernel-tuning-20260813-r1

mkdir -p "${run_root}"
build_job=$(sbatch --parsable "${source_root}/hpc/rosi/build-two-kernel-tuning.sbatch")
build_job=${build_job%%;*}
[[ "${build_job}" =~ ^[0-9]+$ ]]
run_job=$(sbatch --parsable --dependency="afterok:${build_job}" "${source_root}/hpc/rosi/run-two-kernel-tuning.sbatch")
run_job=${run_job%%;*}
[[ "${run_job}" =~ ^[0-9]+$ ]]
analysis_job=$(
    sbatch --parsable --dependency="afterok:${run_job}" "${source_root}/hpc/rosi/analyze-two-kernel-tuning.sbatch"
)
analysis_job=${analysis_job%%;*}
[[ "${analysis_job}" =~ ^[0-9]+$ ]]
printf 'build_job=%s\nrun_job=%s\nanalysis_job=%s\n' \
    "${build_job}" "${run_job}" "${analysis_job}" | tee "${run_root}/job-ids.txt"
