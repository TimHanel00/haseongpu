#!/usr/bin/env bash
set -euo pipefail

readonly source_root=/home/th168408/workspace/haseonpu-master-all-kernel-timing-20260813
readonly run_root=/home/th168408/workspace/haseonpu-alpakatune-runs/master-all-kernel-timing-20260813-r1

mkdir -p "${run_root}"
build_job=$(sbatch --parsable "${source_root}/hpc/rosi/build-timing-history.sbatch")
build_job=${build_job%%;*}
[[ "${build_job}" =~ ^[0-9]+$ ]]
run_job=$(sbatch --parsable --dependency="afterok:${build_job}" "${source_root}/hpc/rosi/run-timing-history.sbatch")
run_job=${run_job%%;*}
[[ "${run_job}" =~ ^[0-9]+$ ]]
printf 'build_job=%s\nrun_job=%s\n' "${build_job}" "${run_job}" | tee "${run_root}/job-ids.txt"
