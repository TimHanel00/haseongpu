include_guard(GLOBAL)

set(HASE_OPENPMD_GIT_REPOSITORY "https://github.com/openPMD/openPMD-api.git")
set(HASE_OPENPMD_GIT_TAG "0.17.0")
set(HASE_ADIOS2_GIT_REPOSITORY "https://github.com/ornladios/ADIOS2.git")
set(HASE_ADIOS2_GIT_TAG "v2.10.2")

if(DEFINED openPMD_SUPERBUILD)
    set(HASE_OPENPMD_SUPERBUILD_DEFAULT ${openPMD_SUPERBUILD})
else()
    set(HASE_OPENPMD_SUPERBUILD_DEFAULT ON)
endif()
option(HASE_OPENPMD_SUPERBUILD
    "Allow openPMD-api to fetch/build its bundled helper dependencies"
    ${HASE_OPENPMD_SUPERBUILD_DEFAULT}
)
option(HASE_OPENPMD_BUILD_PYTHON_BINDINGS
    "Build openPMD-api Python bindings as part of the HASE CMake build"
    OFF
)

message(STATUS "Fetching pinned ADIOS2 for the HASE openPMD transport")
message(STATUS "Fetching pinned openPMD-api for the HASE openPMD transport")

set(ADIOS2_USE_Fortran
    OFF
    CACHE BOOL
    "Disable ADIOS2 Fortran bindings in the HASE superbuild"
    FORCE
)
set(ADIOS2_USE_Python
    OFF
    CACHE BOOL
    "Disable ADIOS2 Python bindings in the HASE superbuild"
    FORCE
)
set(ADIOS2_BUILD_EXAMPLES
    OFF
    CACHE BOOL
    "Disable ADIOS2 examples in the HASE superbuild"
    FORCE
)
set(ADIOS2_BUILD_TESTING
    OFF
    CACHE BOOL
    "Disable ADIOS2 tests in the HASE superbuild"
    FORCE
)
set(BUILD_TESTING
    OFF
    CACHE BOOL
    "Disable third-party tests while configuring the HASE superbuild dependencies"
    FORCE
)
set(ADIOS2_INSTALL_GENERATE_CONFIG
    ON
    CACHE BOOL
    "Generate ADIOS2 CMake configs required by openPMD's find_package(ADIOS2)"
    FORCE
)

# Keep the ADIOS2 superbuild narrow. HASE's openPMD transport uses ADIOS2
# BP/SST-style openPMD series and does not need HDF5, compression plugins,
# remote/cloud transports, visualization hooks, or profiling infrastructure.
foreach(HASE_ADIOS2_DISABLED_OPTION IN ITEMS
    BZip2
    Blosc2
    Campaign
    Catalyst
    DAOS
    DataMan
    DataSpaces
    Endian_Reverse
    HDF5
    HDF5_VOL
    IME
    LIBPRESSIO
    MGARD
    MHS
    PNG
    Profiling
    SZ
    Sodium
    SysVShMem
    UCX
    ZFP
    ZeroMQ
)
    set(ADIOS2_USE_${HASE_ADIOS2_DISABLED_OPTION}
        OFF
        CACHE STRING
        "Disable unused ADIOS2 ${HASE_ADIOS2_DISABLED_OPTION} support in the HASE superbuild"
        FORCE
    )
endforeach()
if(MPI_FOUND)
    set(ADIOS2_USE_MPI
        ON
        CACHE BOOL
        "Enable MPI in ADIOS2 when HASE MPI is available"
        FORCE
    )
else()
    set(ADIOS2_USE_MPI
        OFF
        CACHE BOOL
        "Disable MPI in ADIOS2 when HASE MPI is unavailable"
        FORCE
    )
endif()

include(FetchContent)
FetchContent_Declare(
    ADIOS2
    GIT_REPOSITORY "${HASE_ADIOS2_GIT_REPOSITORY}"
    GIT_TAG "${HASE_ADIOS2_GIT_TAG}"
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(ADIOS2)
if(EXISTS "${ADIOS2_BINARY_DIR}/adios2-config.cmake")
    set(ADIOS2_DIR
        "${ADIOS2_BINARY_DIR}"
        CACHE PATH
        "ADIOS2 CMake config directory produced by the HASE FetchContent build"
        FORCE
    )
endif()
if(NOT DEFINED ADIOS2_VERSION OR "${ADIOS2_VERSION}" STREQUAL "")
    string(REGEX REPLACE "^v" "" ADIOS2_VERSION "${HASE_ADIOS2_GIT_TAG}")
    set(ADIOS2_VERSION
        "${ADIOS2_VERSION}"
        CACHE STRING
        "ADIOS2 version provided by the HASE FetchContent build"
        FORCE
    )
endif()

set(openPMD_USE_ADIOS2
    ON
    CACHE STRING
    "Enable ADIOS2 backend for the HASE openPMD transport"
    FORCE
)
set(openPMD_USE_HDF5
    OFF
    CACHE STRING
    "Disable HDF5 backend for the HASE openPMD transport"
    FORCE
)
set(openPMD_HAVE_PKGCONFIG
    OFF
    CACHE BOOL
    "Do not generate pkg-config metadata in the HASE superbuild"
    FORCE
)
set(openPMD_USE_VERIFY
    OFF
    CACHE BOOL
    "Disable openPMD internal VERIFY checks in the HASE superbuild"
    FORCE
)
set(openPMD_SUPERBUILD
    ${HASE_OPENPMD_SUPERBUILD}
    CACHE BOOL
    "Allow openPMD-api to fetch/build its bundled helper dependencies"
    FORCE
)
foreach(HASE_OPENPMD_INTERNAL_DEP IN ITEMS CATCH JSON TOML11 PYBIND11)
    set(openPMD_USE_INTERNAL_${HASE_OPENPMD_INTERNAL_DEP}
        ${HASE_OPENPMD_SUPERBUILD}
        CACHE BOOL
        "Use openPMD-api bundled ${HASE_OPENPMD_INTERNAL_DEP} dependency"
        FORCE
    )
endforeach()
set(openPMD_USE_PYTHON
    ${HASE_OPENPMD_BUILD_PYTHON_BINDINGS}
    CACHE STRING
    "Build openPMD-api Python bindings from the HASE CMake build"
    FORCE
)
set(openPMD_BUILD_TESTING
    OFF
    CACHE BOOL
    "Disable openPMD-api tests in the HASE superbuild"
    FORCE
)
set(openPMD_BUILD_EXAMPLES
    OFF
    CACHE BOOL
    "Disable openPMD-api examples in the HASE superbuild"
    FORCE
)
set(openPMD_BUILD_CLI_TOOLS
    OFF
    CACHE BOOL
    "Disable openPMD-api CLI tools in the HASE superbuild"
    FORCE
)
set(openPMD_INSTALL
    OFF
    CACHE BOOL
    "Do not install openPMD-api from the HASE superbuild"
    FORCE
)
if(MPI_FOUND)
    set(openPMD_USE_MPI
        ON
        CACHE STRING
        "Enable MPI in openPMD-api when HASE MPI is available"
        FORCE
    )
else()
    set(openPMD_USE_MPI
        OFF
        CACHE STRING
        "Disable MPI in openPMD-api when HASE MPI is unavailable"
        FORCE
    )
endif()

FetchContent_Declare(
    openPMD
    GIT_REPOSITORY "${HASE_OPENPMD_GIT_REPOSITORY}"
    GIT_TAG "${HASE_OPENPMD_GIT_TAG}"
)
FetchContent_MakeAvailable(openPMD)

if(NOT TARGET openPMD::openPMD)
    message(FATAL_ERROR "openPMD::openPMD target was not created")
endif()

if(TARGET openPMD.py)
    add_custom_target(hase_openpmd_python DEPENDS openPMD.py)
endif()

if(HASE_OPENPMD_BUILD_PYTHON_BINDINGS AND DEFINED openPMD_BINARY_DIR AND DEFINED openPMD_INSTALL_PYTHONDIR)
    set(HASE_OPENPMD_PYTHONPATH
        "${openPMD_BINARY_DIR}/${openPMD_INSTALL_PYTHONDIR}"
        CACHE PATH
        "PYTHONPATH entry for the openPMD-api Python module built by HASE"
        FORCE
    )
    message(
        STATUS
        "HASE openPMD Python module path: ${HASE_OPENPMD_PYTHONPATH}"
    )
endif()
