# HASEonGPU + alpakaTune

An experimental integration of [alpakaTune](https://github.com/TimHanel00/alpaka3-tuner)
with the [`forwardTet4` HASEonGPU branch](https://github.com/TimHanel00/haseongpu/tree/forwardTet4).
It applies runtime launch-configuration tuning to HASEonGPU's Alpaka kernels
and includes a reproducible validation campaign for comparing tuning strategies
against an untuned baseline.

This repository is a modified HASEonGPU distribution intended for tuning
research. It is not the upstream HASEonGPU repository or an official
HASEonGPU release.

## What this repository adds

- An opt-in `HASE_ENABLE_ALPAKATUNE` CMake option.
- alpakaTune integration at every direct HASE-owned `FrameSpec` launch in the
  Tet4 forward ray/SRM, pump, derivative, state-integration, active-mask, and
  beta-mapping paths (17 stable kernel identities).
- Exhaustive, random, simulated-annealing, Bayesian-optimization, and learned
  hybrid tuning configurations.
- JSON tuning history and JSON Lines traces containing selected launch
  configurations, kernel runtimes, recommendation latency, and cache state.
- A baseline-versus-tuned Slurm campaign for the `laserPumpCladding` workload
  on the Rosi HPC cluster.
- Analysis tooling for numerical validation and performance plots.

The integration changes only the frame decomposition of a natural HASE kernel
launch. It uses revivalTuner's coverage-preserving `makeFrameSpecTuning`
generator: the original launch and, where possible, a half-frame-extent /
double-frame-count alternative. Kernels are not replayed and no physics
parameter is tuned.

## Requirements

- Linux
- Python 3.10 or newer
- CMake 3.24 or newer
- A C++20 compiler
- An Alpaka-supported CPU or GPU backend
- CUDA or HIP/ROCm when targeting the corresponding GPU backend
- Internet access during the default CMake build, unless all dependencies are
  supplied locally

MPI is optional. The Python frontend communicates with the standalone
`calcPhiASE` backend through openPMD-api.

## Quick start

Create an environment and build an alpakaTune-enabled installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip

CMAKE_ARGS="-DHASE_ENABLE_ALPAKATUNE=ON -DDISABLE_MPI=ON" \
  python3 -m pip install -v .
```

By default, CMake fetches the pinned alpakaTune revision declared in
`CMakeLists.txt`. An existing source checkout can instead be supplied with:

```bash
-DFETCHCONTENT_SOURCE_DIR_ALPAKATUNE=/absolute/path/to/alpaka3-tuner
```

Choose a strategy and provide writable history and trace paths:

```bash
mkdir -p runs/random

export ALPAKA_TUNE_CONFIG="$PWD/hpc/rosi/configs/random.yaml"
export HASE_ALPAKATUNE_HISTORY="$PWD/runs/random/history.json"
export HASE_ALPAKATUNE_TRACE="$PWD/runs/random/trace.jsonl"

python3 example/laserPumpCladding.py \
  --backend Host_Cpu_CpuSerial \
  --timeSteps 2 \
  --pumpSteps 2 \
  --no-vtk
```

Use a backend that was enabled in your Alpaka build. Available backend names
can be inspected with:

```bash
python3 utils/listOpenPmdBackends.py
python3 -c "from HASEonGPU import AlpakaBackends; print(AlpakaBackends.all())"
```

The first command lists openPMD transport backends; the second lists Alpaka
compute backends. These are separate runtime choices.

## Build options

The integration is disabled by default so an ordinary build retains the
upstream launch behavior.

| Option | Default | Purpose |
|---|---:|---|
| `HASE_ENABLE_ALPAKATUNE` | `OFF` | Route HASE-owned kernel launches through alpakaTune. |
| `HASE_ALPAKATUNE_GIT_REPOSITORY` | alpaka3-tuner repository | Dependency fetched by CMake. |
| `HASE_ALPAKATUNE_GIT_TAG` | pinned commit | Reproducible alpakaTune revision. |
| `HASE_SELECT_BACKEND_ALPAKA` | `OFF` | Select Alpaka dependencies and executors explicitly. |
| `HASE_CUDA_ARCHITECTURES` | `native` | CUDA architecture used by the build. |
| `DISABLE_MPI` | `AUTO` | `ON` disables MPI, `OFF` requires it, and `AUTO` detects it. |

For example, a manually selected CUDA build can be configured with:

```bash
cmake -S . -B build-tuned \
  -DHASE_ENABLE_ALPAKATUNE=ON \
  -DHASE_SELECT_BACKEND_ALPAKA=ON \
  -Dalpaka_DEP_CUDA=ON \
  -Dalpaka_EXEC_GpuCuda=ON \
  -DHASE_CUDA_ARCHITECTURES=80 \
  -DDISABLE_MPI=ON

cmake --build build-tuned -j
```

See the [HASEonGPU compilation guide](docs/source/compilation.rst) for the
openPMD provider, accelerator, MPI, and packaging options inherited from
HASEonGPU.

## Runtime configuration

An alpakaTune-enabled executable requires these variables:

| Variable | Required | Meaning |
|---|---:|---|
| `ALPAKA_TUNE_CONFIG` | yes | YAML strategy configuration. |
| `HASE_ALPAKATUNE_HISTORY` | yes | Persistent alpakaTune history file. |
| `HASE_ALPAKATUNE_TRACE` | yes | Append-only per-launch JSONL trace. |
| `HASE_ALPAKATUNE_MODEL` | learned strategy only | Trained `.atml` model used by `learned_hybrid`. |
| `HASE_ALPAKATUNE_METRICS` | no | Append-only per-kernel aggregate of host-call, recommendation, measured-runtime, and estimated synchronization/control time. |
| `HASE_ALPAKATUNE_KERNELS` | no | Comma-separated kernel identities to tune; unset or empty tunes every instrumented kernel. |
| `HASE_ALPAKATUNE_BASELINE_ONLY=1` | instrumented baseline only | Restrict each kernel's tuning space to its original `FrameSpec`, while retaining the normal measured tuner launch path. |
| `HASE_STEP_TIMINGS_JSONL` | no | Write cumulative timestamps from inside the compiled backend after each completed time step. |

The integration deliberately rejects learned-strategy fallback states so a
learned validation run cannot silently turn into a random or heuristic run.

Offline replay uses the same application interface and compatible history as
online tuning; only the YAML `tuning.mode` changes to `offline`. The tuner then
launches the recorded winner without a strategy, warm-up, timing
synchronization, or history update. A history is compatible only with the same
kernel, launch space, and physical device model.

Ready-to-use strategy files are stored in [`hpc/rosi/configs`](hpc/rosi/configs).
The adaptive campaign configurations use a fixed random seed, no warm-up
launches, rolling timing windows, and a 40,000-execution admission/cooling
horizon. The basic comparison measures one natural invocation per residency;
the paired history experiment measures three. Reaching the horizon signals
policy completion but does not stop adaptation, measurement, or the
surrounding HASE simulation.

## Validation campaign

The primary Rosi campaign compares an untuned CUDA baseline with adaptive
random and learned-hybrid tuning, using three repetitions of each mode on an
exclusive NVIDIA A100 node. The configuration suite additionally contains
exhaustive, simulated-annealing, and Bayesian-optimization strategies. The
campaign runs the `laserPumpCladding` example with a fixed physics RNG seed and
retains partial traces when a run reaches its time budget.

The workflow is:

1. Build separate baseline and tuned executables with
   `hpc/rosi/configure-build.sh`.
2. Submit the smoke task, full Slurm array, and dependent analysis job with
   `hpc/rosi/submit-chain.sh`.
3. Inspect the generated report, tables, and plots below the campaign's
   `evaluations/` directory.

`hpc/rosi/submit-overhead-chain.sh` runs the longer break-even campaign. It
adds a native baseline, an `online_fixed` single-candidate instrumented
baseline, and offline history replay to the online random and learned modes.
Every run records cumulative outer-step times; tuned runs also record aggregate
tuner-side host-call accounting.

`hpc/rosi/run-history-horizon-campaign.sbatch` exercises the separate
history-read and history-write controls with the new adaptive `horizon` API.
It pairs a fresh learned collection with a read-only learned continuation, and
a fresh random collection with read-only offline replay. The collection runs
record three measurements per complete candidate residency; the learned
continuation uses `horizon_offset_with_active_history: 0.8` while retaining a
full fresh horizon. See `hpc/rosi/README.md` for the exact five-run contract.

The campaign validates finite output and requires completed tuned runs to keep
the PhiASE and beta integrals within 5% of the baseline medians. It also plots
kernel runtime, recommendation latency, and total application wall time.

The files in `hpc/rosi` are site-specific research scripts. They contain Rosi
module versions and absolute workspace, result, and model paths. Adapt those
paths before using them under another account or on another cluster. See
[`hpc/rosi/README.md`](hpc/rosi/README.md) for the exact pinned campaign
contract.

## Repository layout

| Path | Contents |
|---|---|
| `include/alpakaUtils/TunedEnqueue.hpp` | HASE-to-alpakaTune launch adapter and trace writer. |
| `hpc/rosi/configs/` | Tuning strategy YAML files. |
| `hpc/rosi/` | Rosi build and Slurm campaign scripts. |
| `scripts/analyze_alpakatune_campaign.py` | Validation, CSV generation, and plotting. |
| `example/laserPumpCladding.py` | Campaign workload and local runnable example. |
| `docs/` | HASEonGPU user and developer documentation. |

## Upstream project and attribution

This work is based on HASEonGPU, an open-source HPC application for calculating
amplified spontaneous emission flux in laser gain media. The integration branch
starts from `TimHanel00/haseongpu:forwardTet4` commit
`d6df76b5dbc860f3810420cc9a5d9b8c0c38ca1b`; the campaign dependency snapshot
pins the `alpaka3-tuner:revivalTuner` tip
`30d54a71971a1653976c75610424c3b7db6f50aa`.

For the original software, documentation, authors, and issue tracker, visit the
[upstream HASEonGPU project](https://github.com/ComputationalRadiationPhysics/haseongpu).

If you use HASEonGPU in scientific work, cite the metadata in
[`CITATION.cff`](CITATION.cff) and the associated publication:

> C. H. J. Eckert, E. Zenker, M. Bussmann, and D. Albach,
> *HASEonGPU—An adaptive, load-balanced MPI/GPU-code for calculating the
> amplified spontaneous emission in high power laser media*, Computer Physics
> Communications 207 (2016), 362–374.
> <https://doi.org/10.1016/j.cpc.2016.05.019>

## License

This modified distribution remains licensed under the GNU General Public
License v3.0 or later. See [`COPYING`](COPYING) and [`LICENSE.md`](LICENSE.md).
Individual files retain their original copyright notices.
