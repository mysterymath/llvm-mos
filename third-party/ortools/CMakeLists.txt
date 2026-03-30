# LLVM-compatible CMakeLists.txt for OR-Tools CP-SAT solver.
#
# This replaces OR-Tools' upstream CMake so that the CP-SAT solver
# integrates with LLVM's build and export system (via add_llvm_library).
#
# Maintained by llvm-mos. If upstream changes, run update_ortools.sh
# to re-vendor sources.

# No cmake_minimum_required here — we inherit LLVM's.

# ======================================================================
# Undo LLVM compile definitions that conflict with third-party code.
# LLVM sets _LIBCPP_HARDENING_MODE globally which breaks abseil/protobuf
# when they're compiled against libstdc++ instead of libc++.
# ======================================================================
remove_definitions(-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE)
get_directory_property(_DEFS COMPILE_DEFINITIONS)
list(FILTER _DEFS EXCLUDE REGEX "_LIBCPP_HARDENING_MODE")
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "${_DEFS}")

# ======================================================================
# Always build in release mode. The CP-SAT solver is unusably slow in
# debug mode, and we never need to debug into third-party code.
# ======================================================================
if(NOT MSVC)
  add_compile_options(-O2 -DNDEBUG)
else()
  add_compile_options(/O2 /DNDEBUG)
endif()

# ======================================================================
# Suppress warnings from third-party code under LLVM's strict flags.
# ======================================================================
if(NOT MSVC)
  add_compile_options(
    -Wno-unused-parameter
    -Wno-sign-compare
    -Wno-missing-field-initializers
    -Wno-unused-private-field
    -Wno-covered-switch-default
    -Wno-shorten-64-to-32
    -Wno-deprecated-declarations
    -Wno-implicit-int-float-conversion
    -Wno-unused-variable
    -Wno-shadow
    -Wno-extra-semi
  )
endif()

# ======================================================================
# Dependencies: build via their own CMakeLists.txt
# ======================================================================

# zlib and bzip2 not needed (only used by base/file.cc and
# base/recordio.cc which we exclude).

# abseil-cpp
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ABSL_BUILD_TEST_HELPERS OFF CACHE BOOL "" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
# Prevent abseil's per-target source dirs from being added as -I paths,
# which shadows system headers (absl/time/time.h shadows <time.h>).
set(CMAKE_INCLUDE_CURRENT_DIR OFF)
add_subdirectory(abseil-cpp EXCLUDE_FROM_ALL)

# protobuf (runtime only — no compiler)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_LIBPROTOC OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_LIBUPB OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
set(protobuf_WITH_ZLIB OFF CACHE BOOL "" FORCE)
set(protobuf_ABSL_PROVIDER "package" CACHE STRING "" FORCE)
add_subdirectory(protobuf EXCLUDE_FROM_ALL)

# ======================================================================
# Add all abseil/protobuf targets to the LLVM export set.
# LLVMMOSCodeGen (an exported LLVM library) depends on ortools, which
# depends on these. CMake requires all transitive deps of exported
# targets to also be exported.
# ======================================================================
function(export_directory_targets_to_llvm dir)
  get_property(_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
  foreach(_tgt IN LISTS _targets)
    get_target_property(_type ${_tgt} TYPE)
    get_target_property(_aliased ${_tgt} ALIASED_TARGET)
    get_target_property(_imported ${_tgt} IMPORTED)
    if(_aliased OR _imported)
      continue()
    endif()
    # Only export real library targets (STATIC, OBJECT, INTERFACE).
    if(_type MATCHES "STATIC_LIBRARY|OBJECT_LIBRARY|INTERFACE_LIBRARY")
      set_property(GLOBAL APPEND PROPERTY LLVM_EXPORTS ${_tgt})
      install(TARGETS ${_tgt} EXPORT LLVMExports
        ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}")
    endif()
  endforeach()
  # Recurse into subdirectories.
  get_property(_subdirs DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
  foreach(_sub IN LISTS _subdirs)
    export_directory_targets_to_llvm(${_sub})
  endforeach()
endfunction()

export_directory_targets_to_llvm(${CMAKE_CURRENT_SOURCE_DIR}/abseil-cpp)
export_directory_targets_to_llvm(${CMAKE_CURRENT_SOURCE_DIR}/protobuf)

# ======================================================================
# OR-Tools CP-SAT sources
# ======================================================================

# Collect source files for each OR-Tools component.
file(GLOB ORTOOLS_SAT_SRCS ortools/sat/*.cc)
list(FILTER ORTOOLS_SAT_SRCS EXCLUDE REGEX "(sat_runner|circuit_main|swig_helper|_benchmark)\\.cc$")
file(GLOB ORTOOLS_SAT_CAPI_SRCS ortools/sat/c_api/*.cc)
file(GLOB ORTOOLS_GLOP_SRCS ortools/glop/*.cc)
# Exclude file readers not available under __PORTABLE_PLATFORM__.
file(GLOB ORTOOLS_LP_DATA_SRCS ortools/lp_data/*.cc)
list(FILTER ORTOOLS_LP_DATA_SRCS EXCLUDE REGEX "(mps_reader|model_reader|sol_reader)")
# solve_flow_model.cc uses File I/O.
file(GLOB ORTOOLS_GRAPH_SRCS ortools/graph/*.cc)
list(FILTER ORTOOLS_GRAPH_SRCS EXCLUDE REGEX "solve_flow_model\\.cc$")
# graph_base only exists in OR-Tools main, not v9.12.
file(GLOB ORTOOLS_GRAPH_BASE_SRCS ortools/graph_base/*.cc)
# set_cover_reader.cc uses File I/O.
file(GLOB ORTOOLS_ALGORITHMS_SRCS ortools/algorithms/*.cc)
list(FILTER ORTOOLS_ALGORITHMS_SRCS EXCLUDE REGEX "set_cover_reader\\.cc$")
# set_cover only exists in OR-Tools main, not v9.12.
file(GLOB ORTOOLS_SET_COVER_SRCS ortools/set_cover/*.cc)
file(GLOB ORTOOLS_BASE_SRCS ortools/base/*.cc)
# Exclude file I/O and related utilities (not available under __PORTABLE_PLATFORM__).
list(FILTER ORTOOLS_BASE_SRCS EXCLUDE REGEX "(file|filesystem|gzipfile|gzipstring|zipfile|recordio|temp_path)\\.cc$")
# Exclude file I/O utilities not available under __PORTABLE_PLATFORM__.
file(GLOB ORTOOLS_UTIL_SRCS ortools/util/*.cc)
list(FILTER ORTOOLS_UTIL_SRCS EXCLUDE REGEX "(file_util|qap_reader)\\.cc$")
file(GLOB ORTOOLS_PORT_SRCS ortools/port/*.cc)

# Exclude benchmarks and tests from all components.
foreach(_VAR
    ORTOOLS_SAT_SRCS ORTOOLS_GLOP_SRCS ORTOOLS_LP_DATA_SRCS
    ORTOOLS_GRAPH_SRCS ORTOOLS_GRAPH_BASE_SRCS ORTOOLS_ALGORITHMS_SRCS
    ORTOOLS_SET_COVER_SRCS ORTOOLS_BASE_SRCS ORTOOLS_UTIL_SRCS
    ORTOOLS_PORT_SRCS)
  list(FILTER ${_VAR} EXCLUDE REGEX "_benchmark[s]?\\.cc$")
  list(FILTER ${_VAR} EXCLUDE REGEX "_test\\.cc$")
  list(FILTER ${_VAR} EXCLUDE REGEX "_fuzz\\.cc$")
endforeach()

# Generated protobuf sources.
file(GLOB_RECURSE ORTOOLS_PROTO_SRCS ortools/generated/*.pb.cc)

# Build as a static library, exported alongside LLVM libraries.
add_library(ortools STATIC
  ${ORTOOLS_SAT_SRCS}
  ${ORTOOLS_SAT_CAPI_SRCS}
  ${ORTOOLS_GLOP_SRCS}
  ${ORTOOLS_LP_DATA_SRCS}
  ${ORTOOLS_GRAPH_SRCS}
  ${ORTOOLS_GRAPH_BASE_SRCS}
  ${ORTOOLS_ALGORITHMS_SRCS}
  ${ORTOOLS_SET_COVER_SRCS}
  ${ORTOOLS_BASE_SRCS}
  ${ORTOOLS_UTIL_SRCS}
  ${ORTOOLS_PORT_SRCS}
  ${ORTOOLS_PROTO_SRCS}
)

# Include paths:
# - ortools/ for #include "ortools/sat/cp_model.h"
# - ortools/generated/ for #include "ortools/sat/cp_model.pb.h"
#   (generated protos mirror the ortools/ directory structure)
target_include_directories(ortools PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/ortools/generated
)

# Compile definitions matching upstream OR-Tools cpp.cmake.
target_compile_definitions(ortools PUBLIC
  OR_PROTO_DLL=    # DLL export (empty on non-Windows)
  USE_GLOP         # Enable GLOP LP solver (used by CP-SAT)
  USE_BOP          # Enable BOP solver (used by CP-SAT)
  OR_TOOLS_MAJOR=9
  OR_TOOLS_MINOR=12
  OR_TOOLS_PATCH=0
  __PORTABLE_PLATFORM__  # Disable file I/O (avoids zlib dependency)
)

# Match upstream compile options.
if(NOT MSVC)
  target_compile_options(ortools PRIVATE -fwrapv)
endif()

# C++17 required.
target_compile_features(ortools PUBLIC cxx_std_17)
set_target_properties(ortools PROPERTIES
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)


# Link dependencies. PRIVATE so they don't leak into LLVM's export set.
# Add ortools to the LLVM export set.
set_property(GLOBAL APPEND PROPERTY LLVM_EXPORTS ortools)
install(TARGETS ortools EXPORT LLVMExports
  ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}")

# PUBLIC because ortools headers include protobuf/abseil headers.
target_link_libraries(ortools PUBLIC
  absl::base
  absl::flags
  absl::flags_parse
  absl::flat_hash_map
  absl::flat_hash_set
  absl::hash
  absl::log
  absl::log_initialize
  absl::random_random
  absl::span
  absl::status
  absl::statusor
  absl::strings
  absl::str_format
  absl::synchronization
  absl::time
  absl::btree
  absl::cleanup
  absl::any_invocable
  absl::bind_front
  absl::function_ref
  absl::inlined_vector
  absl::node_hash_map
  protobuf::libprotobuf
)
