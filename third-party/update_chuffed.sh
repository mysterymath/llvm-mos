#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

echo "This script deletes ${SCRIPT_DIR}/chuffed and re-clones it from GitHub."
echo "It replaces Chuffed's CMakeLists.txt with an LLVM-compatible version"
echo "and patches engine.cpp to remove dynamic_cast (RTTI) usage."
echo
read -p "Press a key to continue, or Ctrl+C to cancel"

echo "****************************************"
echo "Cloning Chuffed"
echo "****************************************"
rm -rf "${SCRIPT_DIR}/chuffed"
git clone https://github.com/chuffed/chuffed.git "${SCRIPT_DIR}/chuffed"
rm -rf "${SCRIPT_DIR}/chuffed/.git"*

echo "****************************************"
echo "Removing unnecessary files"
echo "****************************************"
rm -rf "${SCRIPT_DIR}/chuffed/build"
rm -rf "${SCRIPT_DIR}/chuffed/.github"

echo "****************************************"
echo "Checking upstream CMakeLists.txt"
echo "****************************************"
UPSTREAM_HASH=$(sha256sum "${SCRIPT_DIR}/chuffed/CMakeLists.txt" | cut -d' ' -f1)
STAMP_FILE="${SCRIPT_DIR}/chuffed_cmake_upstream_hash"

if [ -f "${STAMP_FILE}" ]; then
    STORED_HASH=$(cat "${STAMP_FILE}")
    if [ "${UPSTREAM_HASH}" != "${STORED_HASH}" ]; then
        echo "WARNING: Upstream CMakeLists.txt has changed!"
        echo "  Stored hash: ${STORED_HASH}"
        echo "  New hash:    ${UPSTREAM_HASH}"
        echo "  You may need to update chuffed_CMakeLists.txt."
        echo
        read -p "Press a key to continue anyway, or Ctrl+C to cancel"
    else
        echo "Upstream CMakeLists.txt unchanged."
    fi
fi

echo "${UPSTREAM_HASH}" > "${STAMP_FILE}"

echo "****************************************"
echo "Replacing CMakeLists.txt with LLVM version"
echo "****************************************"
mv "${SCRIPT_DIR}/chuffed/CMakeLists.txt" "${SCRIPT_DIR}/chuffed/CMakeLists.txt.upstream"
cp "${SCRIPT_DIR}/chuffed_CMakeLists.txt" "${SCRIPT_DIR}/chuffed/CMakeLists.txt"

echo "****************************************"
echo "Patching engine.cpp to remove RTTI usage"
echo "****************************************"
# When CHUFFED_NO_FLATZINC is defined (set in our CMakeLists.txt), replace
# dynamic_cast<FlatZincSpace*> with nullptr and use a stub header instead
# of the real flatzinc.h. This lets Chuffed compile with -fno-rtti.
patch -d "${SCRIPT_DIR}/chuffed" -p1 <<'PATCH'
--- a/chuffed/core/engine.cpp
+++ b/chuffed/core/engine.cpp
@@ -6,7 +6,11 @@
 #include "chuffed/core/propagator.h"
 #include "chuffed/core/sat-types.h"
 #include "chuffed/core/sat.h"
+#ifdef CHUFFED_NO_FLATZINC
+#include "chuffed/flatzinc/flatzinc_stub.h"
+#else
 #include "chuffed/flatzinc/flatzinc.h"
+#endif
 #include "chuffed/ldsb/ldsb.h"
 #include "chuffed/mip/mip.h"
 #include "chuffed/support/misc.h"
@@ -333,7 +337,11 @@
 	if (so.sbps) {
 		saveCurrentSolution();
 	}
+#ifdef CHUFFED_NO_FLATZINC
+	FlatZinc::FlatZincSpace* fzn = nullptr;
+#else
 	auto* fzn = dynamic_cast<FlatZinc::FlatZincSpace*>(problem);
+#endif
 	if (fzn != nullptr) {
 		fzn->storeSolution();
 	}
@@ -669,7 +677,11 @@
 		profilerConnector->start(problemLabel, so.cpprofiler_id, true);
 	}
 #endif
+#ifdef CHUFFED_NO_FLATZINC
+	FlatZinc::FlatZincSpace* fzn = nullptr;
+#else
 	auto* fzn = dynamic_cast<FlatZinc::FlatZincSpace*>(problem);
+#endif
 	if ((fzn != nullptr) && fzn->restart_status >= 0) {
 		const Lit p = fzn->iv[fzn->restart_status]->getLit(1, LR_EQ);  // START
 		assumptions.push(toInt(p));
@@ -960,9 +972,11 @@

 			if (di == nullptr) {
 				solutions++;
+#ifndef CHUFFED_NO_FLATZINC
 				if (auto* oss = dynamic_cast<std::stringstream*>(output_stream)) {
 					oss->str("");
 				}
+#endif
 				if (so.print_sol) {
 					problem->print(*output_stream);
 					(*output_stream) << "\n----------\n";
@@ -1055,7 +1069,11 @@
 			if (doProfiling()) {
 				std::string info;
 				if (so.send_domains) {
+#ifdef CHUFFED_NO_FLATZINC
+					FlatZinc::FlatZincSpace* fzs = nullptr;
+#else
 					auto* fzs = dynamic_cast<FlatZinc::FlatZincSpace*>(problem);
+#endif
 					if (fzs != nullptr) {
 						info = fzs->getDomainsString();
 					}
PATCH

echo "****************************************"
echo "Installing FlatZinc stub header"
echo "****************************************"
# The stub provides a minimal FlatZincSpace definition so engine.cpp
# compiles without the real FlatZinc headers (which need RTTI/exceptions).
# All methods are no-ops since fzn is always nullptr with the stub.
cat > "${SCRIPT_DIR}/chuffed/chuffed/flatzinc/flatzinc_stub.h" << 'STUB'
// Stub FlatZincSpace for builds without FlatZinc (CHUFFED_NO_FLATZINC).
// Provides type definitions so engine.cpp compiles; all code paths using
// FlatZinc are dead (fzn/fzs are always nullptr).

#ifndef CHUFFED_FLATZINC_STUB_H
#define CHUFFED_FLATZINC_STUB_H

#include "chuffed/core/engine.h"
#include "chuffed/support/vec.h"
#include "chuffed/vars/int-var.h"

#include <iostream>
#include <string>

namespace FlatZinc {

class FlatZincSpace : public Problem {
public:
	vec<IntVar*> iv;
	int restart_status = -1;
	bool solution_found = false;
	bool enable_on_restart = false;

	void print(std::ostream&) override {}
	void storeSolution() {}
	bool onRestart(Engine*) { return false; }
	void printDomains(std::ostream& = std::cout) {}
	std::string getDomainsString() { return ""; }
};

}  // namespace FlatZinc

#endif
STUB

echo "****************************************"
echo "Done. Chuffed vendored at ${SCRIPT_DIR}/chuffed/"
echo "****************************************"
