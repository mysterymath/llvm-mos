#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ORTOOLS_TAG="v9.12"
DEST="${SCRIPT_DIR}/ortools"
TMPDIR=$(mktemp -d)

echo "This script vendors OR-Tools CP-SAT and its dependencies into"
echo "${DEST}/"
echo
echo "It will:"
echo "  1. Clone OR-Tools at ${ORTOOLS_TAG}"
echo "  2. Apply patches and build (to generate proto files)"
echo "  3. Copy needed sources + generated protos + deps"
echo "  4. Install LLVM-compatible CMakeLists.txt"
echo
read -p "Press a key to continue, or Ctrl+C to cancel"

cleanup() {
    echo "Cleaning up temp dir..."
    rm -rf "${TMPDIR}"
}
trap cleanup EXIT

# =========================================================================
echo "****************************************"
echo "Step 1: Clone OR-Tools"
echo "****************************************"
git clone --depth 1 --branch "${ORTOOLS_TAG}" \
    https://github.com/google/or-tools.git "${TMPDIR}/or-tools"

# =========================================================================
echo "****************************************"
echo "Step 2: Apply patches"
echo "****************************************"

# Fix: missing #ifdef USE_GUROBI guards in model_builder_helper.cc
# (causes link errors when building with -DUSE_GUROBI=OFF on main branch)
# No patches needed — we only build the proto generation target,
# which doesn't link and therefore doesn't hit Gurobi/Xpress issues.

# =========================================================================
echo "****************************************"
echo "Step 3: Configure and build OR-Tools"
echo "****************************************"
cmake -B "${TMPDIR}/build" -S "${TMPDIR}/or-tools" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_DEPS=ON \
    -DUSE_SCIP=OFF \
    -DUSE_COINOR=OFF \
    -DUSE_GLPK=OFF \
    -DUSE_HIGHS=OFF \
    -DUSE_PDLP=OFF \
    -DUSE_GUROBI=OFF \
    -DUSE_CPLEX=OFF \
    -DBUILD_MATH_OPT=OFF \
    -DBUILD_FLATZINC=OFF \
    -DBUILD_SAMPLES=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_PYTHON=OFF \
    -DBUILD_JAVA=OFF \
    -DBUILD_DOTNET=OFF

# Only build the proto generation target — we don't need the full
# OR-Tools library (which would fail to link without Gurobi anyway).
ninja -C "${TMPDIR}/build" ortools_proto
echo "Proto generation succeeded."

# =========================================================================
echo "****************************************"
echo "Step 4: Copy vendored sources"
echo "****************************************"
rm -rf "${DEST}"
mkdir -p "${DEST}/ortools"

# OR-Tools source directories needed by CP-SAT
for dir in sat glop lp_data graph algorithms base util port; do
    echo "  Copying ortools/${dir}/"
    cp -r "${TMPDIR}/or-tools/ortools/${dir}" "${DEST}/ortools/${dir}"
    # Remove test files and main files
    find "${DEST}/ortools/${dir}" -name "*_test.cc" -o -name "*_test.h" \
        -o -name "*_main.cc" -o -name "*_fuzz*" | xargs rm -f 2>/dev/null || true
done

# Also need some headers from other modules (included but not compiled)
for dir in bop constraint_solver linear_solver packing routing scheduling; do
    echo "  Copying ortools/${dir}/ (headers only)"
    mkdir -p "${DEST}/ortools/${dir}"
    cp "${TMPDIR}/or-tools/ortools/${dir}"/*.h "${DEST}/ortools/${dir}/" 2>/dev/null || true
done

# =========================================================================
echo "****************************************"
echo "Step 5: Copy generated proto files"
echo "****************************************"
mkdir -p "${DEST}/ortools/generated"

# Copy all generated .pb.h and .pb.cc from build, preserving the
# ortools/ prefix so #include "ortools/sat/cp_model.pb.h" resolves.
for pb in $(find "${TMPDIR}/build/ortools" -name "*.pb.h" -o -name "*.pb.cc"); do
    relpath="${pb#${TMPDIR}/build/}"
    destdir="${DEST}/ortools/generated/$(dirname "${relpath}")"
    mkdir -p "${destdir}"
    cp "${pb}" "${destdir}/"
done
echo "  Copied $(find "${DEST}/ortools/generated" -name "*.pb.*" | wc -l) proto files."

# =========================================================================
echo "****************************************"
echo "Step 6: Copy dependencies"
echo "****************************************"

# Abseil
echo "  Copying abseil-cpp..."
cp -r "${TMPDIR}/build/_deps/absl-src" "${DEST}/abseil-cpp"
rm -rf "${DEST}/abseil-cpp/.git"*
rm -rf "${DEST}/abseil-cpp/ci"
find "${DEST}/abseil-cpp" -name "*_test.cc" -o -name "*_testing*" \
    -o -name "*_benchmark*" | xargs rm -f 2>/dev/null || true

# Protobuf (runtime only, no compiler)
echo "  Copying protobuf..."
cp -r "${TMPDIR}/build/_deps/protobuf-src" "${DEST}/protobuf"
rm -rf "${DEST}/protobuf/.git"*
rm -rf "${DEST}/protobuf/docs"
rm -rf "${DEST}/protobuf/benchmarks"
rm -rf "${DEST}/protobuf/conformance"
rm -rf "${DEST}/protobuf/examples"
rm -rf "${DEST}/protobuf/csharp"
rm -rf "${DEST}/protobuf/java"
rm -rf "${DEST}/protobuf/python"
rm -rf "${DEST}/protobuf/ruby"
rm -rf "${DEST}/protobuf/objectivec"
rm -rf "${DEST}/protobuf/php"
rm -rf "${DEST}/protobuf/rust"
find "${DEST}/protobuf" -name "*_test*" -o -name "*unittest*" \
    -o -name "*mock*" | xargs rm -rf 2>/dev/null || true

# zlib and bzip2 not needed — only used by ortools/base/file.cc
# and base/recordio.cc which we exclude from the build.

# =========================================================================
echo "****************************************"
echo "Step 7: Install LLVM CMakeLists.txt"
echo "****************************************"
cp "${SCRIPT_DIR}/ortools_CMakeLists.txt" "${DEST}/CMakeLists.txt"

# =========================================================================
echo "****************************************"
echo "Step 8: Compute sizes"
echo "****************************************"
echo "  Total vendored size: $(du -sh "${DEST}" | cut -f1)"
echo "  ortools sources: $(du -sh "${DEST}/ortools" | cut -f1)"
echo "  abseil-cpp: $(du -sh "${DEST}/abseil-cpp" | cut -f1)"
echo "  protobuf: $(du -sh "${DEST}/protobuf" | cut -f1)"

echo
echo "****************************************"
echo "Done. OR-Tools CP-SAT vendored at ${DEST}/"
echo "****************************************"
