#!/usr/bin/env bash
# Run only inside the single interactive Rosi login-node dependency session.
set -euo pipefail

module purge
module load gcc/14.2.0
module load cmake/4.0.3
module load cuda/12.8
module load python/3.12.4

readonly source_root=/home/th168408/workspace/haseonpu-alpakatune-forwardtet4-20260724
readonly dependencies=${source_root}/dependencies
readonly provider_prefix=${source_root}/build-baseline/hase-openpmd-provider/install
readonly venv=${source_root}/.venv
readonly provider_venv=${source_root}/.venv-provider

python3 -m venv "${venv}"
python3 -m venv "${provider_venv}"
"${venv}/bin/python3" -m pip install --upgrade pip
"${provider_venv}/bin/python3" -m pip install --upgrade pip

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
-DFETCHCONTENT_SOURCE_DIR_ALPAKA=${dependencies}/alpaka"

(
    cd "${source_root}"
    CMAKE_ARGS="${common_cmake_args} \
        -DHASE_RUNTIME_DIR=${source_root}/build-baseline \
        -DHASE_ENABLE_ALPAKATUNE=OFF \
        -DHASE_OPENPMD_PROVIDER=bundled \
        -DHASE_OPENPMD_USE_ADIOS2=ON \
        -DHASE_OPENPMD_USE_SST=ON \
        -DHASE_OPENPMD_USE_HDF5=OFF \
        -DHASE_OFFLINE_DEPENDENCY_ROOT=${dependencies}" \
        "${provider_venv}/bin/python3" -m pip install .

    provider_python="${provider_prefix}/lib/python3.12/site-packages"
    test -d "${provider_python}/openpmd_api"

    CMAKE_ARGS="${common_cmake_args} \
        -DHASE_RUNTIME_DIR=${source_root}/build-tuned \
        -DHASE_ENABLE_ALPAKATUNE=ON \
        -DHASE_OPENPMD_PROVIDER=system \
        -DHASE_OPENPMD_PYTHON_PACKAGE_DIR=${provider_python} \
        -DCMAKE_PREFIX_PATH=${provider_prefix} \
        -DopenPMD_DIR=${provider_prefix}/lib/cmake/openPMD \
        -DFETCHCONTENT_SOURCE_DIR_ALPAKATUNE=${dependencies}/alpakaTune \
        -DFETCHCONTENT_SOURCE_DIR_YAML_CPP=${dependencies}/yaml_cpp \
        -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=${dependencies}/nlohmann_json" \
        "${venv}/bin/python3" -m pip install .
)

test -x "${source_root}/build-baseline/calcPhiASE"
test -x "${source_root}/build-tuned/calcPhiASE"
provider_python="${provider_prefix}/lib/python3.12/site-packages"
PYTHONPATH="${provider_python}" \
    "${venv}/bin/python3" -c 'import HASEonGPU, matplotlib, numpy, openpmd_api, scipy'
if strings "${source_root}/build-tuned/calcPhiASE" | grep -q 'CpuSerial'; then
    printf 'CpuSerial appears in the tuned binary\n' >&2
    exit 1
fi
