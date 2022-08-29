#ifndef FIXEDSTEPDRIVER_DEFINED__
#define FIXEDSTEPDRIVER_DEFINED__

#include <armadillo>
#include <math.h>

#include "EX4_EX4_3_2_A_Coefficients.hpp"
#include "EX2_EX2_2_1_A_Coefficients.hpp"
#include "MRIGARKERK33Coefficients.hpp"
#include "MRIGARKIRK21aCoefficients.hpp"
#include "MRIGARKERK45aCoefficients.hpp"
#include "MRIGARKESDIRK34aCoefficients.hpp"
#include "MRIGARKSDIRK33aCoefficients.hpp"
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
#include "ARK54ESDIRKCoefficients.hpp"
#include "ARK43ESDIRKCoefficients.hpp"
#include "ARK32ESDIRKCoefficients.hpp"
#include "SDIRK21Coefficients.hpp"
#include "DormandPrinceERKCoefficients.hpp"
#include "ZonneveldERKCoefficients.hpp"
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

class FixedStepDriver {
public:
	void run(Problem* problem, vec* H_vec, vec* M_vec, vec* output_tspan, mat* Y_true) {
		vec atol(problem->problem_dimension,fill::ones);
		atol = 1e-14*atol;
		double rtol = 1e-14;
		WeightedErrorNorm err_norm(&atol, rtol);

		EX4_EX4_3_2_A_Coefficients ex4_ex4_3_2_A;
		EX2_EX2_2_1_A_Coefficients ex2_ex2_2_1_A;
		MRIGARKERK33Coefficients mrigarkerk33;
		MRIGARKIRK21aCoefficients mrigarkirk21a;
		MRIGARKERK45aCoefficients mrigarkerk45a;
		MRIGARKESDIRK34aCoefficients mrigarkesdirk34a;
		MRIGARKSDIRK33aCoefficients mrigarksdirk33a;
		MRIGARKERK22aCoefficients mrigarkerk22a;
		IMEXMRI3aCoefficients imexmri3a;
		IMEXMRI4Coefficients imexmri4;
		MERK32aCoefficients merk32a;
		
		MRIGARKTEST1Coefficients mrigarktest1;
		MRIGARKTEST2Coefficients mrigarktest2;

		Type1MRGARKFixedMethod type1_mrgark_method(
			problem->problem_dimension
		);

		MRIGARKFixedMethod mrigark_method(
			problem,
			problem->problem_dimension
		);

		//run_single_mrgark_method(problem, &type1_mrgark_method, &ex4_ex4_3_2_A, H_vec, M_vec, output_tspan, Y_true);
		//run_single_mrgark_method(problem, &type1_mrgark_method, &ex2_ex2_2_1_A, H_vec, M_vec, output_tspan, Y_true);
		
		//run_single_mrigark_method(problem, &mrigark_method, &mrigarkerk33, H_vec, M_vec, output_tspan, Y_true, &err_norm);
		//run_single_mrigark_method(problem, &mrigark_method, &mrigarkerk45a, H_vec, M_vec, output_tspan, Y_true, &err_norm);
		//run_single_mrigark_method(problem, &mrigark_method, &mrigarkerk22a, H_vec, M_vec, output_tspan, Y_true, &err_norm);
		//run_single_mrigark_method(problem, &mrigark_method, &merk32a, H_vec, M_vec, output_tspan, Y_true, &err_norm);
		//run_single_mrigark_method(problem, &mrigark_method, &mrigarktest1, H_vec, M_vec, output_tspan, Y_true, &err_norm);
		//run_single_mrigark_method(problem, &mrigark_method, &mrigarktest2, H_vec, M_vec, output_tspan, Y_true, &err_norm);
		if (!problem->explicit_only) {
			//run_single_mrigark_method(problem, &mrigark_method, &mrigarkirk21a, H_vec, M_vec, output_tspan, Y_true, &err_norm);
			run_single_mrigark_method(problem, &mrigark_method, &mrigarksdirk33a, H_vec, M_vec, output_tspan, Y_true, &err_norm);
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkesdirk34a, H_vec, M_vec, output_tspan, Y_true, &err_norm);
			//run_single_mrigark_method(problem, &mrigark_method, &imexmri3a, H_vec, M_vec, output_tspan, Y_true, &err_norm);
			//run_single_mrigark_method(problem, &mrigark_method, &imexmri4, H_vec, M_vec, output_tspan, Y_true, &err_norm);
		}

		SDIRK21Coefficients sdirk21;
		ARK32ESDIRKCoefficients ark32esdirk;
		ARK43ESDIRKCoefficients ark43esdirk;
		ARK54ESDIRKCoefficients ark54esdirk;
		HeunEulerERKCoefficients heuneuler;
		BogackiShampineERKCoefficients bogackishampine;
		ZonneveldERKCoefficients zonneveld;
		DormandPrinceERKCoefficients dormandprince;

		/*
		run_single_dirk_method(problem, &heuneuler, H_vec, output_tspan, Y_true, &err_norm);
		run_single_dirk_method(problem, &bogackishampine, H_vec, output_tspan, Y_true, &err_norm);
		run_single_dirk_method(problem, &zonneveld, H_vec, output_tspan, Y_true, &err_norm);
		run_single_dirk_method(problem, &dormandprince, H_vec, output_tspan, Y_true, &err_norm);
		if (!problem->explicit_only) {
			run_single_dirk_method(problem, &sdirk21, H_vec, output_tspan, Y_true, &err_norm);
			run_single_dirk_method(problem, &ark32esdirk, H_vec, output_tspan, Y_true, &err_norm);
			run_single_dirk_method(problem, &ark43esdirk, H_vec, output_tspan, Y_true, &err_norm);
			run_single_dirk_method(problem, &ark54esdirk, H_vec, output_tspan, Y_true, &err_norm);
		}
		*/
	}

	void run_single_dirk_method(Problem* problem, SingleRateMethodCoefficients* coeffs, vec* H_vec, vec* output_tspan, mat* Y_true, WeightedErrorNorm* err_norm) {
		FixedDIRKMethod method(coeffs, problem, problem->problem_dimension, err_norm);

		int total_Hs = H_vec->n_elem;

		vec errs(total_Hs, fill::zeros);
		vec fast_function_evals(total_Hs, fill::zeros);
		vec slow_function_evals(total_Hs, fill::zeros);
		vec implicit_function_evals(total_Hs, fill::zeros);
		vec explicit_function_evals(total_Hs, fill::zeros);
		vec fast_jacobian_evals(total_Hs, fill::zeros);
		vec slow_jacobian_evals(total_Hs, fill::zeros);
		vec implicit_jacobian_evals(total_Hs, fill::zeros);

		vec p;

		printf("\n%s.\n",coeffs->name);
		for(int ih=0; ih<total_Hs; ih++) {
			double H = (*H_vec)(ih);

			problem->reset_eval_counts();
			mat Y = method.solve(problem->t_0, H, &(problem->y_0), output_tspan, true);
			errs(ih) = abs(Y-*Y_true).max();
			fast_function_evals(ih) = problem->fast_function_evals;
			slow_function_evals(ih) = problem->slow_function_evals;
			implicit_function_evals(ih) = problem->implicit_function_evals;
			explicit_function_evals(ih) = problem->explicit_function_evals;
			fast_jacobian_evals(ih) = problem->fast_jacobian_evals;
			slow_jacobian_evals(ih) = problem->slow_jacobian_evals;
			implicit_jacobian_evals(ih) = problem->implicit_jacobian_evals;

			printf("H = %.16f, Err = %.16f\n", H, errs(ih));
		}
		
		p = polyfit(arma::log10(*H_vec),arma::log10(errs),1);
		printf("\tOrder estimate: %f\n",p(0));

		//save_stats(problem->name, instance_name, M, &errs, H_vec, &fast_function_evals, &slow_function_evals, &implicit_function_evals, &explicit_function_evals, &fast_jacobian_evals, &slow_jacobian_evals, &implicit_jacobian_evals);
		
		errs.zeros();
		fast_function_evals.zeros();
		slow_function_evals.zeros();
		implicit_function_evals.zeros();
		explicit_function_evals.zeros();
		fast_jacobian_evals.zeros();
		slow_jacobian_evals.zeros();
		implicit_jacobian_evals.zeros();
	}

	void run_single_mrgark_method(Problem* problem, Type1MRGARKFixedMethod* method, MRGARKCoefficients* coeffs, vec* H_vec, vec* M_vec, vec* output_tspan, mat* Y_true) {
		Type1MRGARKFixedStep type1_mrgark_step(
			coeffs, 
			problem,
			problem->problem_dimension
		);

		run_single_method(problem, method, &type1_mrgark_step, coeffs->name, H_vec, M_vec, output_tspan, Y_true);
	}

	void run_single_mrigark_method(Problem* problem, MRIGARKFixedMethod* method, MRICoefficients* coeffs, vec* H_vec, vec* M_vec, vec* output_tspan, mat* Y_true, WeightedErrorNorm* err_norm) {
		if (coeffs->primary_order == 1 || coeffs->primary_order == 2) {
			HeunEulerERKCoefficients inner_coeffs;
			//printf("Using Heun-Euler\n");
			MRIGARKFixedStep mrigark_step(
				coeffs, 
				&inner_coeffs,
				problem,
				problem->problem_dimension, 
				err_norm
			);
			run_single_method(problem, method, &mrigark_step, coeffs->name, H_vec, M_vec, output_tspan, Y_true);
		} else if (coeffs->primary_order == 3) {
			BogackiShampineERKCoefficients inner_coeffs;
			//printf("Using Bogacki-Shampine\n");
			MRIGARKFixedStep mrigark_step(
				coeffs, 
				&inner_coeffs,
				problem,
				problem->problem_dimension, 
				err_norm
			);
			run_single_method(problem, method, &mrigark_step, coeffs->name, H_vec, M_vec, output_tspan, Y_true);
		} else if (coeffs->primary_order == 4 || coeffs->primary_order == 5) {
			DormandPrinceERKCoefficients inner_coeffs;
			//printf("Using Dormand-Prince\n");
			MRIGARKFixedStep mrigark_step(
				coeffs, 
				&inner_coeffs,
				problem,
				problem->problem_dimension, 
				err_norm
			);
			run_single_method(problem, method, &mrigark_step, coeffs->name, H_vec, M_vec, output_tspan, Y_true);
		}
	}

	void run_single_method(Problem* problem, FixedStepMultiRateMethod* method, FixedStepMultiRateStep* step, const char* instance_name, vec* H_vec, vec* M_vec, vec* output_tspan, mat* Y_true) {
		int total_Ms = M_vec->n_elem;
		int total_Hs = H_vec->n_elem;

		vec errs(total_Hs, fill::zeros);
		vec fast_function_evals(total_Hs, fill::zeros);
		vec slow_function_evals(total_Hs, fill::zeros);
		vec implicit_function_evals(total_Hs, fill::zeros);
		vec explicit_function_evals(total_Hs, fill::zeros);
		vec fast_jacobian_evals(total_Hs, fill::zeros);
		vec slow_jacobian_evals(total_Hs, fill::zeros);
		vec implicit_jacobian_evals(total_Hs, fill::zeros);

		vec p;

		printf("\n%s.\n",instance_name);
		for(int im=0; im<M_vec->n_elem; im++) {
			int M = (*M_vec)(im);
			for(int ih=0; ih<total_Hs; ih++) {
				double H = (*H_vec)(ih);
				double h = H/M;

				problem->reset_eval_counts();
				mat Y = method->solve(problem->t_0, H, M, &(problem->y_0), output_tspan, step);
				errs(ih) = abs(Y-*Y_true).max();
				fast_function_evals(ih) = problem->fast_function_evals;
				slow_function_evals(ih) = problem->slow_function_evals;
				implicit_function_evals(ih) = problem->implicit_function_evals;
				explicit_function_evals(ih) = problem->explicit_function_evals;
				fast_jacobian_evals(ih) = problem->fast_jacobian_evals;
				slow_jacobian_evals(ih) = problem->slow_jacobian_evals;
				implicit_jacobian_evals(ih) = problem->implicit_jacobian_evals;
	
				printf("H = %.16f, M = %d, Err = %.16f\n", H, M, errs(ih));
			}
			
			p = polyfit(arma::log10(*H_vec),arma::log10(errs),1);
			printf("\tOrder estimate: %f\n",p(0));

			//save_stats(problem->name, instance_name, M, &errs, H_vec, &fast_function_evals, &slow_function_evals, &implicit_function_evals, &explicit_function_evals, &fast_jacobian_evals, &slow_jacobian_evals, &implicit_jacobian_evals);
			
			errs.zeros();
			fast_function_evals.zeros();
			slow_function_evals.zeros();
			implicit_function_evals.zeros();
			explicit_function_evals.zeros();
			fast_jacobian_evals.zeros();
			slow_jacobian_evals.zeros();
			implicit_jacobian_evals.zeros();
		}
	}
};

#endif