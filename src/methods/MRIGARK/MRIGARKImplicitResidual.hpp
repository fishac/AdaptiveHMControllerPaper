#ifndef MRIGARKIMPLICITRESIDUAL_DEFINED__
#define MRIGARKIMPLICITRESIDUAL_DEFINED__

#include "Residual.hpp"
#include "MRIGARKInnerRHSFunctions.hpp"

using namespace arma;

class MRIGARKImplicitResidual: public Residual {
public:
	MRIGARKInnerRHSFunctions* inner_rhs_funcs;

	MRIGARKImplicitResidual(MRIGARKInnerRHSFunctions* inner_rhs_funcs_) {
		inner_rhs_funcs = inner_rhs_funcs_;
	}

	void evaluate(double t, vec* explicit_data, vec* y_prev, vec* y, vec* f) {
		inner_rhs_funcs->implicit_solve_residual(explicit_data, y_prev, y, f);
	}

	void evaluate_explicit_data(vec* explicit_data) {
		inner_rhs_funcs->implicit_set_previous_terms();
		*explicit_data = inner_rhs_funcs->implicit_get_previous_terms();
	}
};

#endif