#!/usr/bin/env bash
set -euo pipefail

readonly source_root=/home/th168408/workspace/haseonpu-alpakatune-forwardtet4-20260723

submit_job()
{
    local response
    local job_id
    response=$(sbatch --parsable "$@")
    job_id=${response%%;*}
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        printf 'Could not parse numeric Slurm job ID from: %s\n' "${response}" >&2
        return 1
    fi
    printf '%s\n' "${job_id}"
}

smoke_job=$(submit_job --array=0,3,6,9,12 "${source_root}/hpc/rosi/run-overhead-campaign.sbatch")
campaign_job=$(submit_job --array=1-2,4-5,7-8,10-11,13-14 --dependency="afterok:${smoke_job}" \
    "${source_root}/hpc/rosi/run-overhead-campaign.sbatch")
analysis_job=$(submit_job --dependency="afterok:${campaign_job}" \
    "${source_root}/hpc/rosi/analyze-overhead-campaign.sbatch")
printf 'smoke_job=%s\ncampaign_job=%s\nanalysis_job=%s\n' "${smoke_job}" "${campaign_job}" "${analysis_job}"
