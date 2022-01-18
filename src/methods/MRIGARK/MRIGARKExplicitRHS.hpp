#ifndef MRIGARKEXPLICITRHS_DEFINED__
#define MRIGARKEXPLICITRHS_DEFINED__

#include "RHS.hpp"
#include "MRIGARKInnerRHSFunctions.hpp"

using namespace arma;

class MRIGARKExplicitRHS: public RHS {
public:
	MRIGARKInnerRHSFunctions* inner_rhs_funcs;

	MRIGARKExplicitRHS(MRIGARKInnerRHSFunctions* inner_rhs_funcs_) {
		inner_rhs_funcs = inner_rhs_funcs_;
	}
	
	void evaluate(double theta, vec* v, vec* f) {
		inner_rhs_funcs->explicit_solve_rhs(theta, v, f);
	}
};

#endif