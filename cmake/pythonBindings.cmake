set(HASE_PYTHON_RUNTIME_DIR "${CMAKE_BINARY_DIR}/python/pyInclude/_native")

if(TARGET hase_openpmd_python)
    add_dependencies(calcPhiASE hase_openpmd_python)
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
file(MAKE_DIRECTORY "${HASE_PYTHON_RUNTIME_DIR}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/pyInclude/_native_config.py.in"
    "${CMAKE_BINARY_DIR}/python/pyInclude/_native_config.py"
    @ONLY
)
add_custom_command(
    TARGET HaseAlpakaBackendNames
    POST_BUILD
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:HaseAlpakaBackendNames>"
        "${HASE_PYTHON_RUNTIME_DIR}/$<TARGET_FILE_NAME:HaseAlpakaBackendNames>"
    COMMENT
        "Copying Alpaka backend-name library to build package ${HASE_PYTHON_RUNTIME_DIR}"
    VERBATIM
)
add_custom_command(
    TARGET calcPhiASE
    POST_BUILD
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:calcPhiASE>"
        "${HASE_PYTHON_RUNTIME_DIR}/$<TARGET_FILE_NAME:calcPhiASE>"
    COMMENT
        "Copying calcPhiASE executable to build package ${HASE_PYTHON_RUNTIME_DIR}"
    VERBATIM
)
install(TARGETS calcPhiASE RUNTIME DESTINATION pyInclude/_native)
install(TARGETS HaseAlpakaBackendNames LIBRARY DESTINATION pyInclude/_native)
install(
    FILES "${CMAKE_BINARY_DIR}/python/pyInclude/_native_config.py"
    DESTINATION pyInclude
)
install(FILES HASEonGPU.py DESTINATION .)
