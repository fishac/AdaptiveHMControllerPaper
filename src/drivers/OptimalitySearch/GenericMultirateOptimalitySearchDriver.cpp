#include <armadillo>
#include <math.h>

#include "MultirateOptimalitySearchDriver.hpp"
#include "BicouplingProblem.hpp"
#include "BicouplingLNProblem.hpp"
#include "BicouplingDLProblem.hpp"
#include "BrusselatorProblem.hpp"
#include "BrusselatorDLProblem.hpp"
#include "FourBody3dProblem.hpp"
#include "KapsProblem.hpp"
#include "KapsLNProblem.hpp"
#include "KapsDLProblem.hpp"
#include "KPRProblem.hpp"
#include "KPRDLProblem.hpp"
#include "LienardProblem.hpp"
#include "LienardLNProblem.hpp"
#include "LienardDLProblem.hpp"
#include "OregonatorProblem.hpp"
#include "OregonatorDLProblem.hpp"
#include "PleiadesProblem.hpp"
#include "Brusselator1DProblem.hpp"
#include "Problem.hpp"
#include "MRIGARKFixedMethod.hpp"
#include "MRIGARKERK22aCoefficients.hpp"
#include "MRIGARKERK33Coefficients.hpp"
#include "MRIGARKIRK21aCoefficients.hpp"
#include "MRIGARKERK45aCoefficients.hpp"
#include "MRIGARKESDIRK34aCoefficients.hpp"
#include "MERK32aCoefficients.hpp"
#include "HeunEulerERKCoefficients.hpp"
#include "BogackiShampineERKCoefficients.hpp"
#include "ZonneveldERKCoefficients.hpp"
#include "VernerERKCoefficients.hpp"
#include "WeightedErrorNorm.hpp"

using namespace std;
using namespace arma;

void setup_and_run_with_problem(Problem* problem, const char* method_name, const char* tol_string, double tol, const char* spf_string, int slow_penalty_factor, double H_fine, double H_tol, double H_interval, int M_max_iter, int M_min_iter, double eff_rtol) {
	MultirateOptimalitySearchDriver driver;
	vec atol(problem->problem_dimension, fill::ones);
	atol *= tol/10.0;
	double rtol = tol/10.0;
	WeightedErrorNorm err_norm(&atol, rtol);
	VernerERKCoefficients reference_coeffs;
	FixedDIRKMethod reference_method(
		&reference_coeffs,
		problem,
		problem->problem_dimension,
		&err_norm
	);
	
	if(strcmp("MRIGARKERK33",method_name) == 0) {
		BogackiShampineERKCoefficients inner_coeffs;
		MRIGARKERK33Coefficients coeffs;
		MRIGARKFixedMethod method(
			problem,
			problem->problem_dimension
		);
		MRIGARKFixedStep step(
			&coeffs,
			&inner_coeffs,
			problem,
			problem->problem_dimension,
			&err_norm
		);
		driver.run(problem, &method, &step, coeffs.name, tol_string, spf_string,&reference_method, &(problem->y_0), problem->t_0, problem->t_f, tol, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
	} else if(strcmp("MRIGARKERK22a",method_name) == 0) {
		HeunEulerERKCoefficients inner_coeffs;
		MRIGARKERK22aCoefficients coeffs;
		MRIGARKFixedMethod method(
			problem,
			problem->problem_dimension
		);
		MRIGARKFixedStep step(
			&coeffs,
			&inner_coeffs,
			problem,
			problem->problem_dimension,
			&err_norm
		);
		driver.run(problem, &method, &step, coeffs.name, tol_string, spf_string, &reference_method, &(problem->y_0), problem->t_0, problem->t_f, tol, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
	} else if(strcmp("MRIGARKIRK21a",method_name) == 0) {
		HeunEulerERKCoefficients inner_coeffs;
		MRIGARKIRK21aCoefficients coeffs;
		MRIGARKFixedMethod method(
			problem,
			problem->problem_dimension
		);
		MRIGARKFixedStep step(
			&coeffs,
			&inner_coeffs,
			problem,
			problem->problem_dimension,
			&err_norm
		);
		driver.run(problem, &method, &step, coeffs.name, tol_string, spf_string, &reference_method, &(problem->y_0), problem->t_0, problem->t_f, tol, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
	} else if(strcmp("MRIGARKERK45a",method_name) == 0) {
		ZonneveldERKCoefficients inner_coeffs;
		MRIGARKERK45aCoefficients coeffs;
		MRIGARKFixedMethod method(
			problem,
			problem->problem_dimension
		);
		MRIGARKFixedStep step(
			&coeffs,
			&inner_coeffs,
			problem,
			problem->problem_dimension,
			&err_norm
		);
		driver.run(problem, &method, &step, coeffs.name, tol_string, spf_string, &reference_method, &(problem->y_0), problem->t_0, problem->t_f, tol, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
	} else if(strcmp("MRIGARKESDIRK34a",method_name) == 0) {
		BogackiShampineERKCoefficients inner_coeffs;
		MRIGARKESDIRK34aCoefficients coeffs;
		MRIGARKFixedMethod method(
			problem,
			problem->problem_dimension
		);
		MRIGARKFixedStep step(
			&coeffs,
			&inner_coeffs,
			problem,
			problem->problem_dimension,
			&err_norm
		);
		driver.run(problem, &method, &step, coeffs.name, tol_string, spf_string, &reference_method, &(problem->y_0), problem->t_0, problem->t_f, tol, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
	} else if(strcmp("MERK32a",method_name) == 0) {
		BogackiShampineERKCoefficients inner_coeffs;
		MERK32aCoefficients coeffs;
		MRIGARKFixedMethod method(
			problem,
			problem->problem_dimension
		);
		MRIGARKFixedStep step(
			&coeffs,
			&inner_coeffs,
			problem,
			problem->problem_dimension,
			&err_norm
		);
		driver.run(problem, &method, &step, coeffs.name, tol_string, spf_string, &reference_method, &(problem->y_0), problem->t_0, problem->t_f, tol, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
	} else {
		printf("Error: Did not recognize method name: %s\n",method_name);
	}
}

int main(int argc, char* argv[]) {
	if(argc != 11) {
		printf("Error: Requires 11 command-line arguments.\n");
		printf("Ex: ./exe/GenericMultiRateOptimalitySearchDriver.exe <ProblemName> <MethodName> <tol> <slow_penalty_factor> <H_fine> <H_tol> <H_interval> <M_max_iter> <M_min_iter> <eff_rtol>\n"); 
		return 1;
	} else {
		double tol = 0.0;
		int slow_penalty_factor = 0;
		double H_fine = 0.0;
		double H_tol = 0.0;
		double H_interval = 0.0;
		int M_max_iter = 0;
		int M_min_iter = 0;
		double eff_rtol = 0.0;

		sscanf(argv[3], "%lf", &tol);
		sscanf(argv[4], "%d", &slow_penalty_factor);
		sscanf(argv[5], "%lf", &H_fine);
		sscanf(argv[6], "%lf", &H_tol);
		sscanf(argv[7], "%lf", &H_interval);
		sscanf(argv[8], "%d", &M_max_iter);
		sscanf(argv[9], "%d", &M_min_iter);
		sscanf(argv[10], "%lf", &eff_rtol);

		const char* problem_name = argv[1];
		const char* method_name = argv[2];
		const char* tol_string = argv[3];
		const char* spf_string = argv[4];

		if(strcmp("Bicoupling",problem_name) == 0) {
			BicouplingProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
		} else if(strcmp("BicouplingLN",problem_name) == 0) {
			BicouplingLNProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
		} else if(strcmp("BicouplingDL",problem_name) == 0) {
			BicouplingDLProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
		} else if(strcmp("Brusselator",problem_name) == 0) {
			BrusselatorProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
		} else if(strcmp("BrusselatorDL",problem_name) == 0) {
			BrusselatorDLProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
		} else if(strcmp("FourBody3d",problem_name) == 0) {
			FourBody3dProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
		} else if(strcmp("Kaps",problem_name) == 0) {
			KapsProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
		} else if(strcmp("KapsLN",problem_name) == 0) {
			KapsLNProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
		} else if(strcmp("KapsDL",problem_name) == 0) {
			KapsDLProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
		} else if(strcmp("KPR",problem_name) == 0) {
			KPRProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
		} else if(strcmp("KPRDL",problem_name) == 0) {
			KPRDLProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
		} else if(strcmp("Lienard",problem_name) == 0) {
			LienardProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
		} else if(strcmp("LienardLN",problem_name) == 0) {
			LienardLNProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
		} else if(strcmp("LienardDL",problem_name) == 0) {
			LienardDLProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
		} else if(strcmp("Oregonator",problem_name) == 0) {
			OregonatorProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
		} else if(strcmp("OregonatorDL",problem_name) == 0) {
			OregonatorDLProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
		} else if(strcmp("Pleiades",problem_name) == 0) {
			PleiadesProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
		} else if(strcmp("Brusselator1D",problem_name) == 0) {
			Brusselator1DProblem problem(10);
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
		} else {
			printf("Error: Did not recognize problem name: %s\n", problem_name);
			return 1;
		}
	}
	return 0;
}