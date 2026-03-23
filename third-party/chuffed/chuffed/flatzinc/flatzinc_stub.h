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
