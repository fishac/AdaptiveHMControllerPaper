#ifndef DIRKRESIDUALJACOBIAN_DEFINED__
#define DIRKRESIDUALJACOBIAN_DEFINED__

#include "SingleRateMethodCoefficients.hpp"
#include "ResidualJacobian.hpp"
#include "RHSJacobian.hpp"

using namespace arma;

class DIRKResidualJacobian: public ResidualJacobian {
public:
	SingleRateMethodCoefficients* coeffs;
	RHSJacobian* rhsjac;
	mat jac_temp;
	mat I;
	int problem_dimension;
	double h;
	int stage_index;

	DIRKResidualJacobian(SingleRateMethodCoefficients* coeffs_, RHSJacobian* rhsjac_, int problem_dimension_) {
		coeffs = coeffs_;
		rhsjac = rhsjac_;
		problem_dimension = problem_dimension_;
		jac_temp = mat(problem_dimension, problem_dimension, fill::zeros);
		I = eye(problem_dimension,problem_dimension);
	}

	void evaluate(double t, vec* y, mat* jac) {
		jac->zeros();
		rhsjac->evaluate(t+(coeffs->c(stage_index))*h, y, &jac_temp);
		*jac = I - h*(coeffs->A(stage_index,stage_index))*jac_temp;
	}

	void set_problem_dependent_data(double h_) {
		h = h_;
	}

	void set_function_dependent_data(int stage_index_) {
		stage_index = stage_index_;
	}

};

#endif