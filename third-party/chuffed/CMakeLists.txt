# LLVM-compatible CMakeLists.txt for Chuffed CP solver.
#
# This replaces Chuffed's upstream CMakeLists.txt so that Chuffed
# integrates with LLVM's build and export system (via add_llvm_library).
#
# Maintained by llvm-mos. If upstream adds new source files, this file
# needs to be updated to match. The update_chuffed.sh script checks for
# upstream CMakeLists.txt changes and warns if this may be needed.

# No cmake_minimum_required here — we inherit LLVM's.

# Suppress warnings that fire under LLVM's strict warning flags.
if(NOT MSVC)
  add_compile_options(
    -Wno-unused-parameter
    -Wno-sign-compare
    -Wno-missing-field-initializers
    -Wno-unused-private-field
    -Wno-covered-switch-default
  )
endif()

# Build without FlatZinc support (avoids RTTI/exceptions requirement).
add_compile_definitions(CHUFFED_NO_FLATZINC)

# Thirdparty: MurmurHash3 (public domain)
add_library(thirdparty_murmur OBJECT
  thirdparty/MurmurHash3/MurmurHash3.cpp
)
target_include_directories(thirdparty_murmur PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Core Chuffed library. Uses add_llvm_library (without BUILDTREE_ONLY)
# so it goes into LLVM_EXPORTS alongside LLVMMOSCodeGen, which depends
# on it. CMake requires all dependencies of exported targets to be in
# the same export set.
add_llvm_library(chuffed
  chuffed/vars/int-var.cpp
  chuffed/vars/int-var-el.cpp
  chuffed/vars/modelling.cpp
  chuffed/vars/int-var-sl.cpp
  chuffed/vars/bool-view.cpp
  chuffed/vars/int-var-ll.cpp
  chuffed/ldsb/ldsb.cpp
  chuffed/globals/subcircuit.cpp
  chuffed/globals/mddglobals.cpp
  chuffed/globals/sym-break.cpp
  chuffed/globals/linear-bool.cpp
  chuffed/globals/linear-bool-decomp.cpp
  chuffed/globals/well-founded.cpp
  chuffed/globals/circuit.cpp
  chuffed/globals/minimum.cpp
  chuffed/globals/bool_arg_max.cpp
  chuffed/globals/alldiff.cpp
  chuffed/globals/template.cpp
  chuffed/globals/directives.cpp
  chuffed/globals/cumulative.cpp
  chuffed/globals/cumulativeCalendar.cpp
  chuffed/globals/disjunctive.cpp
  chuffed/globals/regular.cpp
  chuffed/globals/lex.cpp
  chuffed/globals/table.cpp
  chuffed/globals/edit_distance.cpp
  chuffed/globals/EdExplFinder.cpp
  chuffed/globals/value-precede.cpp
  chuffed/globals/graph.cpp
  chuffed/globals/tree.cpp
  chuffed/globals/minimum_weight_tree.cpp
  chuffed/globals/mst.cpp
  chuffed/globals/dconnected.cpp
  chuffed/globals/dtree.cpp
  chuffed/globals/dag.cpp
  chuffed/globals/bounded_path.cpp
  chuffed/mdd/MDD.cpp
  chuffed/mdd/mdd_prop.cpp
  chuffed/mdd/mdd_to_lgraph.cpp
  chuffed/mdd/opcache.cpp
  chuffed/mdd/weighted_dfa.cpp
  chuffed/mdd/wmdd_prop.cpp
  chuffed/mip/mip.cpp
  chuffed/mip/recalc.cpp
  chuffed/mip/simplex.cpp
  chuffed/primitives/element.cpp
  chuffed/primitives/bool.cpp
  chuffed/primitives/linear.cpp
  chuffed/primitives/arithmetic.cpp
  chuffed/primitives/domain.cpp
  chuffed/primitives/binary.cpp
  chuffed/branching/branching.cpp
  chuffed/core/init.cpp
  chuffed/core/stats.cpp
  chuffed/core/engine.cpp
  chuffed/core/options.cpp
  chuffed/core/sat.cpp
  chuffed/core/conflict.cpp
  chuffed/support/union_find.cpp
  chuffed/support/lengauer_tarjan.cpp
  chuffed/support/dijkstra.cpp
  chuffed/support/kosaraju_scc.cpp

  $<TARGET_OBJECTS:thirdparty_murmur>
)
target_include_directories(chuffed PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
