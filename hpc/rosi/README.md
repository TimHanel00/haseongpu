# Rosi HASEonGPU alpakaTune validation

This campaign compares the uninstrumented `forwardTet4` HASEonGPU path with
the online-adaptive random and learned alpakaTune strategies on the same `laserPumpCladding`
workload. The source base is `TimHanel00/haseongpu:forwardTet4` commit
`b674bb817a199bcbde84cc05124e5079c5c35c37`; alpakaTune is a FetchContent
dependency pinned to `revivalTuner` commit
`175169bb83cba557a13c1228266bc8db7c1b1824`.

That tuner snapshot contains the history-aware restart and bounded adaptive
retry behavior directly. HASE does not patch tuner semantics locally. When one
adaptive refill exhausts its retry budget with an empty queue, the tuner
reopens the best measured history entry for one measured natural launch and
tries the strategy again on the next refill.

The tuned build intercepts every direct HASE-owned `FrameSpec` launch in the
Tet4 forward/SRM, pump, derivative, state-integration, mask, and mapping paths.
For each natural launch it uses revivalTuner's generated, coverage-preserving
`FrameSpec` alternatives: the original decomposition and, where possible, a
half-frame-extent / double-frame-count alternative. No new kernel parameter is
introduced and no kernel invocation is replayed.

The Slurm array contains nine tasks: one CUDA backend, three modes, and three
repetitions. Every run requests one A100, eight host CPUs, and 64 GB from either
Rosi A100 partition. Each application invocation requests 150 outer time steps, 100
pumped steps, reflections, fixed RNG seed 5489, no VTK callback, and a 1800 s
Python-side deadline checked between completed outer steps. Slurm remains the
hard stop if a single backend step does not return. Timed-out runs retain partial
traces and are treated as censored observations.

The random and learned strategy configurations explicitly use `online_adaptive`, zero
warmups, one scored natural invocation per candidate, a ten-sample rolling
history, and a 40,000-execution admission/cooling horizon. Adaptation continues
after the horizon. The learned run uses the held-out ensemble at the absolute
campaign path and terminates immediately if alpakaTune reports any fallback
state.

`analyze-laser-campaign.sbatch` checks finite results and compares completed
PhiASE and beta integrals against the pristine CUDA baseline with a 5%
tolerance. It writes run and timeout tables plus kernel-runtime,
recommendation-latency, and application-wall-time plots under `evaluations/`.

The login-node setup builds both runtime variants through the supported
`python3 -m pip install .` interface. The baseline and tuned native artifacts
remain in separate durable runtime directories; the final installed Python
frontend points at the tuned runtime. A separate provider venv retains the
baseline install contract, while the matching bundled `openpmd_api` module is
consumed directly from the durable provider prefix by both runtime variants.

No command on Rosi uses Git. Source and third-party snapshots are transferred
with targeted RSYNC calls, and every remote write stays below
`/home/th168408/workspace`. This campaign uses the isolated source snapshot
`/home/th168408/workspace/haseonpu-alpakatune-forwardtet4-20260723`; the older
Rosi checkout is left untouched.

## History-aware horizon comparison

`run-history-horizon-campaign.sbatch` is a paired, single-allocation follow-up
using alpakaTune `revivalTuner` commit
`175169bb83cba557a13c1228266bc8db7c1b1824`. It executes these five modes
sequentially during one continuous A100 allocation:

1. the uninstrumented HASE baseline;
2. a fresh learned adaptive collection (`read: false`, `write: true`);
3. learned adaptive continuation from that history (`read: true`,
   `write: false`);
4. a fresh random adaptive collection (`read: false`, `write: true`);
5. offline replay of that random history (`read: true`, `write: false`).

Both adaptive collection configurations use `horizon: 40000`,
`horizon_offset_with_active_history: 0.8`, no warm-up, and three consecutive
measured launches per complete candidate residency. Every application run uses
1000 time steps and 1000 pump steps. The learned continuation therefore starts
its unchanged sigmoid/Boltzmann policy at normalized position 0.8 and stretches
the remaining interval across a full new 40,000-launch horizon for each tuner
context. All online modes set
`maximum_consecutive_strategy_retries: 20`; rejected proposals are retried
synchronously. An empty adaptive scheduler uses the measured-incumbent
progress fallback described above instead of entering terminal replay.

The job hashes both history files immediately before and after their read-only
consumer. A changed hash fails the job. Metrics and traces remain enabled in
all instrumented modes even when history writing is disabled. The dependent
`analyze-history-horizon-campaign.sbatch` validates numerics, reports
instrumentation totals and residual-adapter persistence, and writes cumulative
time and baseline-delta plots.
