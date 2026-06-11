include_guard(GLOBAL)

if(POLICY CMP0057)
    cmake_policy(SET CMP0057 NEW)
endif()

set(HASE_OPENPMD_BACKEND
    "bp"
    CACHE STRING
    "openPMD backend used by the HASE transport: adios, adios-sst, bp, or hdf5"
)
set_property(
    CACHE HASE_OPENPMD_BACKEND
    PROPERTY STRINGS adios adios-sst bp hdf5
)

set(HASE_OPENPMD_BACKEND_ALLOWED adios adios-sst bp hdf5)
string(TOLOWER "${HASE_OPENPMD_BACKEND}" HASE_OPENPMD_BACKEND_NORMALIZED)
if(NOT HASE_OPENPMD_BACKEND STREQUAL HASE_OPENPMD_BACKEND_NORMALIZED)
    set(HASE_OPENPMD_BACKEND
        "${HASE_OPENPMD_BACKEND_NORMALIZED}"
        CACHE STRING
        "openPMD backend used by the HASE transport: adios, adios-sst, bp, or hdf5"
        FORCE
    )
endif()

if(NOT HASE_OPENPMD_BACKEND IN_LIST HASE_OPENPMD_BACKEND_ALLOWED)
    message(
        FATAL_ERROR
        "Unsupported HASE_OPENPMD_BACKEND='${HASE_OPENPMD_BACKEND}'. "
        "Expected one of: adios, adios-sst, bp, hdf5."
    )
endif()

set(HASE_OPENPMD_USE_ADIOS2 OFF)
set(HASE_OPENPMD_USE_HDF5 OFF)
set(HASE_OPENPMD_FILE_EXTENSION "bp")
if(HASE_OPENPMD_BACKEND STREQUAL "hdf5")
    set(HASE_OPENPMD_USE_HDF5 ON)
    set(HASE_OPENPMD_FILE_EXTENSION "h5")
else()
    set(HASE_OPENPMD_USE_ADIOS2 ON)
endif()

message(STATUS "HASE openPMD backend: ${HASE_OPENPMD_BACKEND}")
