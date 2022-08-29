#ifndef TYPE1MRGARKFIXEDMETHOD_DEFINED__
#define TYPE1MRGARKFIXEDMETHOD_DEFINED__

#include "FixedStepMultiRateMethod.hpp"
#include "Type1MRGARKFixedStep.hpp"
#include "MRGARKCoefficients.hpp"
#include "FixedStepMultiRateStep.hpp"

using namespace arma;

class Type1MRGARKFixedMethod : public FixedStepMultiRateMethod {

public:
	int problem_dimension;
	double effective_H;
	int total_output_points;
	vec y;
	vec y_hat;
	vec y_hat_s;
	vec y_hat_f;
	mat Y;

	Type1MRGARKFixedMethod(int problem_dimension_) 	{
		problem_dimension = problem_dimension_;
	}

	mat solve(double t_0, double H, int M, vec* y_0, vec* output_tspan, FixedStepMultiRateStep* type1_mrgark_step) {
		prepare_solve(H, M, output_tspan, type1_mrgark_step);
		y = *y_0;

		int output_index = 0;
		if((*output_tspan)(0) == t_0) {
			Y.col(0) = *y_0;
			output_index++;
		}

		double t = t_0;
		while(output_index < total_output_points) {
			if(t + H - (*output_tspan)(output_index) > 0.0) {
				effective_H = (*output_tspan)(output_index) - t;
			} else {
				effective_H = H;
			}
			type1_mrgark_step->step_solution(t, effective_H, M, &y, &y);
			if (t + effective_H == (*output_tspan)(output_index)) {
				Y.col(output_index) = y;
				output_index++;
			}
			t += effective_H;
		}
		return Y;
	}
	
	void prepare_solve(double H_, int M_, vec* output_tspan, FixedStepMultiRateStep* type1_mrgark_step) {
		type1_mrgark_step->set_coeffs();
		type1_mrgark_step->refresh_coupling_coeffs(M_);

		effective_H = H_;
		total_output_points = output_tspan->n_elem;

		Y = mat(problem_dimension, total_output_points, fill::zeros);

		y = vec(problem_dimension, fill::zeros);
		y_hat = vec(problem_dimension, fill::zeros);
		y_hat_s = vec(problem_dimension, fill::zeros);
		y_hat_f = vec(problem_dimension, fill::zeros);
	}
};

#endif