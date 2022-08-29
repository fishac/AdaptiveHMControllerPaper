#include <armadillo>
#include <math.h>
#include <stdio.h>

#include "Brusselator1DProblem.hpp"
#include "VernerERKCoefficients.hpp"
#include "FixedDIRKMethod.hpp"
#include "AdaptiveDIRKMethod.hpp"
#include "PIDController.hpp"
#include "MRIGARKERK45aCoefficients.hpp"
#include "ZonneveldERKCoefficients.hpp"
#include "MRIGARKAdaptiveMethod.hpp"
#include "MRIGARKAdaptiveStepSlowMeasurement.hpp"
#include "ConstantConstantController.hpp"

using namespace std;
using namespace arma;

void run_fixed_dirk(int num_steps) {
	Brusselator1DProblem problem(100);
	vec atol(problem.problem_dimension,fill::ones);
	atol = 1e-12*atol;
	double rtol = 1e-12;
	WeightedErrorNorm err_norm(&atol, rtol);
	VernerERKCoefficients coeffs;

	double H = (problem.t_f - problem.t_0)/num_steps;
	FixedDIRKMethod method(
		&coeffs,
		&problem,
		problem.problem_dimension,
		&err_norm
	);
	vec output_tspan = linspace(problem.t_0, problem.t_f, num_steps+1);

	printf("Beginning solve process.\n");
	mat Y = method.solve(problem.t_0, H, &(problem.y_0), &output_tspan);

	printf("Beginning computating of fs(Y) and ff(Y).\n");
	mat fsY(problem.problem_dimension, output_tspan.n_elem, fill::zeros);
	mat ffY(problem.problem_dimension, output_tspan.n_elem, fill::zeros);
	for(int i=0; i<output_tspan.n_elem; i++) {
		vec ytemp = Y.col(i);
		vec ftemp(problem.problem_dimension, fill::zeros);
		double ttemp = i*H;
		
		problem.slow_rhs(ttemp, &ytemp, &ftemp);
		fsY.col(i) = ftemp;

		problem.fast_rhs(ttemp, &ytemp, &ftemp);
		ffY.col(i) = ftemp;
	}

	printf("Beginning to save to disk.\n");
	mat output_mat_T = join_cols(Y,fsY,ffY);
	mat output_time_mat(output_tspan.n_elem, 1, fill::zeros);
	output_time_mat.col(0) = output_tspan;
	mat output_mat = join_rows(output_time_mat, trans(output_mat_T));
	output_mat.save("./output/Brusselator1D/Brusselator1D_fixed_t_Y_fsY_ffY.csv",csv_ascii);

	printf("Done.\n");
}

void run_adaptive_dirk(int num_steps) {
	Brusselator1DProblem problem(100);
	vec atol(problem.problem_dimension,fill::ones);
	atol = 1e-5*atol;
	double rtol = 1e-5;
	WeightedErrorNorm err_norm(&atol, rtol);
	VernerERKCoefficients coeffs;

	double H = (problem.t_f - problem.t_0)/num_steps;
	AdaptiveDIRKMethod method(
		&coeffs,
		&problem,
		problem.problem_dimension,
		&err_norm,
		true
	);
	vec output_tspan = linspace(problem.t_0, problem.t_f, num_steps+1);
	double k1_pid[3] = { 0.49, 0.34, 0.1 };
	double k2_pid[3] = { 0.0, 0.0, 0.0 }; 
	PIDController pidcontroller(
		1.0,
		1.0,
		1.0,
		0.85,
		k1_pid,
		k2_pid
	);

	printf("Beginning solve process.\n");
	AdaptiveSingleRateMethodReturnValue ret;
	method.solve(problem.t_0, H, &(problem.y_0), &output_tspan, &pidcontroller, &ret);
	mat Y = ret.Y;

	printf("Beginning computating of fs(Y) and ff(Y).\n");
	mat fsY(problem.problem_dimension, output_tspan.n_elem, fill::zeros);
	mat ffY(problem.problem_dimension, output_tspan.n_elem, fill::zeros);
	for(int i=0; i<output_tspan.n_elem; i++) {
		vec ytemp = Y.col(i);
		vec ftemp(problem.problem_dimension, fill::zeros);
		double ttemp = i*H;
		
		problem.slow_rhs(ttemp, &ytemp, &ftemp);
		fsY.col(i) = ftemp;

		problem.fast_rhs(ttemp, &ytemp, &ftemp);
		ffY.col(i) = ftemp;
	}

	printf("Beginning to save to disk.\n");
	mat output_mat_T = join_cols(Y,fsY,ffY);
	mat output_time_mat(output_tspan.n_elem, 1, fill::zeros);
	output_time_mat.col(0) = output_tspan;
	mat output_mat = join_rows(output_time_mat, trans(output_mat_T));
	output_mat.save("./output/Brusselator1D/Brusselator1D_fixed_t_Y_fsY_ffY.csv",csv_ascii);

	printf("Done.\n");
}

void run_adaptive_mri(int num_steps, double tol) {
	Brusselator1DProblem problem(100);
	vec atol(problem.problem_dimension,fill::ones);
	atol = tol*atol;
	double rtol = tol;
	WeightedErrorNorm err_norm(&atol, rtol);
	MRIGARKERK45aCoefficients coeffs;
	ZonneveldERKCoefficients inner_coeffs;

	double H = (problem.t_f - problem.t_0)/num_steps;
	int M = 10;
	vec output_tspan = linspace(problem.t_0, problem.t_f, num_steps+1);
	double k1_CC[1] = { 0.54 }; // 10-all
	double k2_CC[1] = { 0.74 }; // 10-all
	ConstantConstantController CCcontroller(
		1.0,
		1.0,
		1.0,
		0.85,
		k1_CC,
		k2_CC
	);
	MRIGARKAdaptiveMethod mrigark_method(
		&problem,
		problem.problem_dimension,
		&err_norm
	);
	AdaptiveMultiRateMethodReturnValue ret;
	MRIGARKAdaptiveStepSlowMeasurement mrigark_step_sm(
		&coeffs, 
		&inner_coeffs, 
		&problem,
		problem.problem_dimension, 
		&err_norm
	);
	const char* measurement_type = "LASA-mean";

	printf("Beginning solve process.\n");
	mrigark_method.solve(problem.t_0, H, M, &(problem.y_0), &output_tspan, &mrigark_step_sm, &CCcontroller, measurement_type, &ret);
	mat Y = ret.Y;
	printf("\tStatus: %d\n",ret.status);

	printf("Beginning computating of fs(Y) and ff(Y).\n");
	mat fsY(problem.problem_dimension, output_tspan.n_elem, fill::zeros);
	mat ffY(problem.problem_dimension, output_tspan.n_elem, fill::zeros);
	for(int i=0; i<output_tspan.n_elem; i++) {
		vec ytemp = Y.col(i);
		vec ftemp(problem.problem_dimension, fill::zeros);
		double ttemp = i*H;
		
		problem.slow_rhs(ttemp, &ytemp, &ftemp);
		fsY.col(i) = ftemp;

		problem.fast_rhs(ttemp, &ytemp, &ftemp);
		ffY.col(i) = ftemp;
	}

	printf("Beginning to save to disk.\n");
	mat output_mat_T = join_cols(Y,fsY,ffY);
	mat output_time_mat(output_tspan.n_elem, 1, fill::zeros);
	output_time_mat.col(0) = output_tspan;
	mat output_mat = join_rows(output_time_mat, trans(output_mat_T));
	output_mat.save("./output/Brusselator1D/Brusselator1D_fixed_t_Y_fsY_ffY.csv",csv_ascii);

	printf("Done.\n");
}

int main(int argc, char* argv[]) {
	if(argc != 3) {
		printf("Error: Requires 1 command-line argument. Ex: B.exe <NumSteps> <tol>\n");
		return 1;
	} else {
		const char* arg_num_steps = argv[1];
		const char* arg_tol = argv[2];
		int num_steps = 0;
		double tol = 0.0;
		sscanf(arg_num_steps, "%d", &num_steps);
		sscanf(arg_num_steps, "%lf", &tol);

		run_adaptive_mri(num_steps,tol);
		return 0;
	}
}