set(HASE_PYTHON_RUNTIME_DIR "${CMAKE_BINARY_DIR}/python/pyInclude")
if(TARGET hase-cpp AND TARGET hase_openpmd_python)
    add_dependencies(hase-cpp hase_openpmd_python)
endif()
if(HASE_FORWARD_LOGGING)
    set(HASE_FORWARD_LOGGING_PY True)
else()
    set(HASE_FORWARD_LOGGING_PY False)
endif()
# HASE wheels do not vendor openPMD-api runtime libraries or Python bindings.
# Even when CMake fetches openPMD-api for the build, the installed package uses
# the runtime environment's openpmd_api module and dynamic libraries.
set(HASE_USE_SYSTEM_OPENPMD_PY True)
if(NOT DEFINED HASE_OPENPMD_PYTHON_PACKAGE_DIR)
    set(HASE_OPENPMD_PYTHON_PACKAGE_DIR "")
endif()

set(HASE_CONFIGURED_RUNTIME_DIR "${HASE_RUNTIME_DIR}")
if(NOT HASE_CONFIGURED_RUNTIME_DIR AND SKBUILD)
    set(HASE_CONFIGURED_RUNTIME_DIR "${CMAKE_BINARY_DIR}")
endif()

string(
    REPLACE
    "\\"
    "\\\\"
    HASE_OPENPMD_PYTHON_PACKAGE_DIR_ESCAPED
    "${HASE_OPENPMD_PYTHON_PACKAGE_DIR}"
)
string(
    REPLACE
    "\""
    "\\\""
    HASE_OPENPMD_PYTHON_PACKAGE_DIR_ESCAPED
    "${HASE_OPENPMD_PYTHON_PACKAGE_DIR_ESCAPED}"
)
string(
    REPLACE
    "\\"
    "\\\\"
    HASE_RUNTIME_DIR_ESCAPED
    "${HASE_CONFIGURED_RUNTIME_DIR}"
)
string(
    REPLACE
    "\""
    "\\\""
    HASE_RUNTIME_DIR_ESCAPED
    "${HASE_RUNTIME_DIR_ESCAPED}"
)
file(MAKE_DIRECTORY "${HASE_PYTHON_RUNTIME_DIR}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/pyInclude/_native_config.py.in"
    "${HASE_PYTHON_RUNTIME_DIR}/_native_config.py"
    @ONLY
)

add_custom_target(hase_python_package)

if(TARGET hase-cpp)
    install(
        TARGETS hase-cpp
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime
    )
endif()
if(TARGET HaseAlpakaBackendNames AND TARGET HaseOpenPmdBackendProbe)
    install(
        TARGETS HaseAlpakaBackendNames HaseOpenPmdBackendProbe
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT runtime
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT runtime
    )
endif()
install(
    FILES "${HASE_PYTHON_RUNTIME_DIR}/_native_config.py"
    DESTINATION pyInclude
    COMPONENT python
)

install(FILES HASEonGPU.py DESTINATION . COMPONENT python)
