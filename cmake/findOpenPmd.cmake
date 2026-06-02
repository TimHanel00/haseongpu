include_guard(GLOBAL)

set(HASE_OPENPMD_GIT_REPOSITORY "https://github.com/openPMD/openPMD-api.git")
set(HASE_OPENPMD_GIT_TAG "0.17.0")
set(HASE_ADIOS2_GIT_REPOSITORY "https://github.com/ornladios/ADIOS2.git")
set(HASE_ADIOS2_GIT_TAG "v2.10.2")

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
set(ADIOS2_INSTALL_GENERATE_CONFIG
    OFF
    CACHE BOOL
    "Do not generate ADIOS2 install configs in the HASE superbuild"
    FORCE
)
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
set(openPMD_USE_PYTHON
    ON
    CACHE STRING
    "Build openPMD-api Python bindings from the HASE CMake build"
    FORCE
)
set(openPMD_USE_INTERNAL_PYBIND11
    ON
    CACHE BOOL
    "Fetch pybind11 for openPMD-api Python bindings"
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

if(DEFINED openPMD_BINARY_DIR AND DEFINED openPMD_INSTALL_PYTHONDIR)
    set(HASE_OPENPMD_PYTHONPATH
        "${openPMD_BINARY_DIR}/${openPMD_INSTALL_PYTHONDIR}"
        CACHE PATH
        "PYTHONPATH entry for the openPMD-api Python module built by HASE"
        FORCE
    )
    message(STATUS "HASE openPMD Python module path: ${HASE_OPENPMD_PYTHONPATH}")
endif()
