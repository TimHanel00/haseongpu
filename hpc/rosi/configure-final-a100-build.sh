#!/usr/bin/env bash
# Run only inside an interactive Rosi login shell.
set -euo pipefail

module purge
module load gcc/14.2.0
module load cmake/4.0.3
module load cuda/12.8
module load python/3.12.4

readonly source_root=/home/th168408/workspace/haseonpu-alpakatune-forwardtet4-final-v2-20260724
readonly shared_root=/home/th168408/workspace/haseonpu-alpakatune-forwardtet4-20260724
readonly shared_dependencies=${shared_root}/dependencies
readonly provider_prefix=${shared_root}/build-baseline/hase-openpmd-provider/install
readonly provider_python=${provider_prefix}/lib/python3.12/site-packages
readonly baseline_venv=${source_root}/.venv-baseline
readonly tuned_venv=${source_root}/.venv

test -d "${shared_dependencies}/alpaka"
test -d "${shared_dependencies}/yaml_cpp"
test -d "${shared_dependencies}/nlohmann_json"
test -d "${source_root}/dependencies/alpakaTune"
test -d "${provider_python}/openpmd_api"

python3 -m venv "${baseline_venv}"
python3 -m venv "${tuned_venv}"
"${baseline_venv}/bin/python3" -m pip install --upgrade pip
"${tuned_venv}/bin/python3" -m pip install --upgrade pip

readonly common_cmake_args="\
-DHASE_BUILD_RELEASE=ON \
-DHASE_NATIVE_OPTIMIZATIONS=OFF \
-DDISABLE_MPI=ON \
-DHASE_SELECT_BACKEND_ALPAKA=ON \
-Dalpaka_DEP_CUDA=ON \
-Dalpaka_DEP_OMP=OFF \
-Dalpaka_EXEC_GpuCuda=ON \
-Dalpaka_EXEC_CpuOmpBlocks=OFF \
-Dalpaka_EXEC_CpuSerial=OFF \
-DHASE_CUDA_ARCHITECTURES=80 \
-DHASE_OPENPMD_PROVIDER=system \
-DHASE_OPENPMD_PYTHON_PACKAGE_DIR=${provider_python} \
-DCMAKE_PREFIX_PATH=${provider_prefix} \
-DopenPMD_DIR=${provider_prefix}/lib/cmake/openPMD \
-DFETCHCONTENT_SOURCE_DIR_ALPAKA=${shared_dependencies}/alpaka"

(
    cd "${source_root}"
    CMAKE_ARGS="${common_cmake_args} \
        -DHASE_RUNTIME_DIR=${source_root}/build-baseline \
        -DHASE_ENABLE_ALPAKATUNE=OFF" \
        "${baseline_venv}/bin/python3" -m pip install .

    CMAKE_ARGS="${common_cmake_args} \
        -DHASE_RUNTIME_DIR=${source_root}/build-tuned \
        -DHASE_ENABLE_ALPAKATUNE=ON \
        -DFETCHCONTENT_SOURCE_DIR_ALPAKATUNE=${source_root}/dependencies/alpakaTune \
        -DFETCHCONTENT_SOURCE_DIR_YAML_CPP=${shared_dependencies}/yaml_cpp \
        -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=${shared_dependencies}/nlohmann_json" \
        "${tuned_venv}/bin/python3" -m pip install .
)

test -x "${source_root}/build-baseline/calcPhiASE"
test -x "${source_root}/build-tuned/calcPhiASE"
PYTHONPATH="${provider_python}" \
    "${tuned_venv}/bin/python3" -c \
    'import HASEonGPU, matplotlib, numpy, openpmd_api, scipy'
if strings "${source_root}/build-tuned/calcPhiASE" | grep -q 'CpuSerial'; then
    printf 'CpuSerial appears in the tuned binary\n' >&2
    exit 1
fi
