#ifndef DIRKRESIDUAL_DEFINED__
#define DIRKRESIDUAL_DEFINED__

#include "SingleRateMethodCoefficients.hpp"
#include "Residual.hpp"
#include "RHS.hpp"

using namespace arma;

class DIRKResidual: public Residual {
public:
	SingleRateMethodCoefficients* coeffs;
	RHS* rhsfunc;
	vec y_temp;
	int problem_dimension;
	vec* explicit_data;
	double h;
	int stage_index;

	DIRKResidual(SingleRateMethodCoefficients* coeffs_, RHS* rhsfunc_, int problem_dimension_) {
		coeffs = coeffs_;
		rhsfunc = rhsfunc_;
		problem_dimension = problem_dimension_;
		y_temp = vec(problem_dimension, fill::zeros);
	}

	void evaluate(double t, vec* explicit_data, vec* y_0, vec* y, vec* f) {
		//printf("DIRKResidual h: %.16f\n",h);
		f->zeros();
		rhsfunc->evaluate(t+(coeffs->c(stage_index))*h, y, &y_temp);
		//printf("DIRKResidual y_temp 2-norm: %.16f\n",norm(y_temp,2));
		*f = *y - *y_0 - h*(*explicit_data + (coeffs->A(stage_index,stage_index))*y_temp);
		//printf("DIRKResidual f 2-norm: %.16f\n",norm(*f,2));
	}

	void evaluate_explicit_data(vec* explicit_data_) {
		*explicit_data_ = *explicit_data;
	}

	void set_problem_dependent_data(double h_) {
		h = h_;
	}

	void set_explicit_data_pointer(vec* explicit_data_) {
		explicit_data = explicit_data_;
	}

	void set_function_dependent_data(int stage_index_) {
		stage_index = stage_index_;
	}

};

#endif