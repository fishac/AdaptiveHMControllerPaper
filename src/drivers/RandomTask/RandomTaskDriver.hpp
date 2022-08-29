#ifndef RANDOMTASKDRIVER_DEFINED__
#define RANDOMTASKDRIVER_DEFINED__

#include <armadillo>
#include <math.h>

#include "EX4_EX4_3_2_A_Coefficients.hpp"
#include "EX2_EX2_2_1_A_Coefficients.hpp"
#include "MRIGARKERK33Coefficients.hpp"
#include "MRIGARKIRK21aCoefficients.hpp"
#include "MRIGARKERK45aCoefficients.hpp"
#include "MRIGARKESDIRK34aCoefficients.hpp"
#include "MRIGARKERK22aCoefficients.hpp"
#include "MRIGARKTEST1Coefficients.hpp"
#include "MRIGARKTEST2Coefficients.hpp"
#include "IMEXMRI3aCoefficients.hpp"
#include "IMEXMRI4Coefficients.hpp"
#include "MERK32aCoefficients.hpp"
#include "Type1MRGARKFixedMethod.hpp"
#include "MRIGARKFixedMethod.hpp"
#include "Type1MRGARKFixedStep.hpp"
#include "MRIGARKFixedStep.hpp"
#include "Problem.hpp"
#include "FixedStepMultiRateMethod.hpp"
#include "DormandPrinceERKCoefficients.hpp"
#include "BogackiShampineERKCoefficients.hpp"
#include "HeunEulerERKCoefficients.hpp"
#include "WeightedErrorNorm.hpp"

using namespace std;
using namespace arma;

void save_stats(const char* problem_name, const char* method_name, int M, vec* errs, vec* H_vec, vec* fast_function_evals, vec* slow_function_evals, vec* implicit_function_evals, vec* explicit_function_evals, vec* fast_jacobian_evals, vec* slow_jacobian_evals, vec* implicit_jacobian_evals) {
	char filename[75];
	sprintf(filename, "./output/%s/%s_Fixed_%s_M%d_stats.csv",problem_name, problem_name,method_name,M);
	mat output = join_rows(*errs, *H_vec, *fast_function_evals, *slow_function_evals);
	output = join_rows(output, *implicit_function_evals, *explicit_function_evals);
	output = join_rows(output, *fast_jacobian_evals, *slow_jacobian_evals, *implicit_jacobian_evals);
	output.save(filename, csv_ascii);
}

class RandomTaskDriver {
public:
	void run(Problem* problem, vec* H_vec, vec* M_vec, vec* output_tspan, mat* Y_true) {
		vec atol(problem->problem_dimension,fill::ones);
		atol = 1e-12*atol;
		double rtol = 1e-12;
		WeightedErrorNorm err_norm(&atol, rtol);

		
		MRIGARKTEST2Coefficients mrigarktest2;
		MRICoefficients* mricoeffs_ptr = &mrigarktest2;

		MRIGARKFixedMethod mrigark_method(
			problem,
			problem->problem_dimension
		);
		
		BogackiShampineERKCoefficients inner_coeffs;
		SingleRateMethodCoefficients* inner_coeffs_ptr = &inner_coeffs;
		
		
		MRIGARKFixedStep mrigark_step(
			mricoeffs_ptr, 
			inner_coeffs_ptr,
			problem,
			problem->problem_dimension, 
			&err_norm
		);
		
		MRIGARKFixedStep mrigark_step_embedding(
			mricoeffs_ptr, 
			inner_coeffs_ptr,
			problem,
			problem->problem_dimension, 
			&err_norm,
			true
		);
		
		for(int im=0; im<M_vec->n_elem; im++) {
			int M = (*M_vec)(im);
			for(int ih=0; ih<H_vec->n_elem; ih++) {
				double H = (*H_vec)(ih);

				mat Y = mrigark_method.solve(problem->t_0, H, M, &(problem->y_0), output_tspan, &mrigark_step);
				mat Y_hat = mrigark_method.solve(problem->t_0, H, M, &(problem->y_0), output_tspan, &mrigark_step_embedding);
				
				double err1 = abs(Y-*Y_true).max();
				double err2 = abs(Y_hat-*Y_true).max();
				double diff = abs(Y-Y_hat).max();
	
				printf("H = %.16f, M = %d, ||Y-Y_true|| = %.16f, ||Y_hat-Y_true|| = %.16f, ||Y-Y_hat|| = %.16f\n", H, M, err1, err2, diff);
			}
		}
	}

};

#endif