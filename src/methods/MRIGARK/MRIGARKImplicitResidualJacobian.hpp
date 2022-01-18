#ifndef MRIGARKIMPLICITRESIDUALJACOBIAN_DEFINED__
#define MRIGARKIMPLICITRESIDUALJACOBIAN_DEFINED__

#include "ResidualJacobian.hpp"
#include "RHSJacobian.hpp"
#include "MRIGARKInnerRHSFunctions.hpp"

using namespace arma;

class MRIGARKImplicitResidualJacobian: public ResidualJacobian {
public:
	MRIGARKInnerRHSFunctions* inner_rhs_funcs;

	MRIGARKImplicitResidualJacobian(MRIGARKInnerRHSFunctions* inner_rhs_funcs_) {
		inner_rhs_funcs = inner_rhs_funcs_;
	}

	void evaluate(double t, vec* y, mat* jac) {
		inner_rhs_funcs->implicit_solve_residual_jacobian(y, jac);
	}	
};

#endif