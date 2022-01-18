#ifndef MRIGARKEXPLICITRHSJACOBIAN_DEFINED__
#define MRIGARKEXPLICITRHSJACOBIAN_DEFINED__

#include "RHSJacobian.hpp"
#include "MRIGARKInnerRHSFunctions.hpp"

using namespace arma;

class MRIGARKExplicitRHSJacobian: public RHSJacobian {
public:
	MRIGARKInnerRHSFunctions* inner_rhs_funcs;

	MRIGARKExplicitRHSJacobian(MRIGARKInnerRHSFunctions* inner_rhs_funcs_) {
		inner_rhs_funcs = inner_rhs_funcs_;
	}
	
	void evaluate(double theta, vec* v, mat* jac) {
		inner_rhs_funcs->explicit_solve_rhsjacobian(theta, v, jac);
	}
};

#endif