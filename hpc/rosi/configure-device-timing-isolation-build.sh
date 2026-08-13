#!/usr/bin/env bash
set -euo pipefail

module purge
module load gcc/14.2.0
module load cmake/4.0.3
module load cuda/12.8
module load python/3.12.4

readonly source_root=/home/th168408/workspace/haseonpu-master-device-timing-isolation-20260813
readonly shared_root=/home/th168408/workspace/haseonpu-alpakatune-forwardtet4-20260724
readonly shared_dependencies=${shared_root}/dependencies
readonly provider_prefix=${shared_root}/build-baseline/hase-openpmd-provider/install
readonly provider_python=${provider_prefix}/lib/python3.12/site-packages
readonly campaign_venv=${source_root}/.venv-device-timing-isolation

test -d "${source_root}/dependencies/alpaka"
test ! -e "${source_root}/dependencies/alpakaTune"
test -d "${shared_dependencies}/yaml_cpp"
test -d "${shared_dependencies}/nlohmann_json"
test -d "${provider_python}/openpmd_api"

python3 -m venv "${campaign_venv}"
"${campaign_venv}/bin/python3" -m pip install --upgrade pip

readonly cmake_args="\
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
-DCMAKE_CUDA_FLAGS=--maxrregcount=64 \
-DHASE_OPENPMD_PROVIDER=system \
-DHASE_OPENPMD_PYTHON_PACKAGE_DIR=${provider_python} \
-DCMAKE_PREFIX_PATH=${provider_prefix} \
-DopenPMD_DIR=${provider_prefix}/lib/cmake/openPMD \
-DHASE_RUNTIME_DIR=${source_root}/build-device-timing-isolation \
-DHASE_ENABLE_ALPAKATUNE=OFF \
-DHASE_ENABLE_DEVICE_TIMING=ON \
-DFETCHCONTENT_SOURCE_DIR_ALPAKA=${source_root}/dependencies/alpaka \
-DFETCHCONTENT_SOURCE_DIR_YAML_CPP=${shared_dependencies}/yaml_cpp \
-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=${shared_dependencies}/nlohmann_json"

(
    cd "${source_root}"
    CMAKE_ARGS="${cmake_args}" "${campaign_venv}/bin/python3" -m pip install .
)

readonly build_root=${source_root}/build-device-timing-isolation
test -x "${build_root}/calcPhiASE"
grep -q '^HASE_ENABLE_ALPAKATUNE:BOOL=OFF$' "${build_root}/CMakeCache.txt"
grep -q '^HASE_ENABLE_DEVICE_TIMING:BOOL=ON$' "${build_root}/CMakeCache.txt"
test -z "$(find "${build_root}" -iname '*alpakatune*' -print -quit)"
