#include <armadillo>
#include <math.h>

#include "SingleRateOptimalitySearchDriver.hpp"
#include "BicouplingProblem.hpp"
#include "BicouplingLNProblem.hpp"
#include "BrusselatorProblem.hpp"
#include "FourBody3dProblem.hpp"
#include "KapsProblem.hpp"
#include "KapsLNProblem.hpp"
#include "KPRProblem.hpp"
#include "LienardProblem.hpp"
#include "LienardLNProblem.hpp"
#include "OregonatorProblem.hpp"
#include "OregonatorLNProblem.hpp"
#include "PleiadesProblem.hpp"
#include "Problem.hpp"
#include "HeunEulerERKCoefficients.hpp"
#include "BogackiShampineERKCoefficients.hpp"
#include "DormandPrinceERKCoefficients.hpp"
#include "VernerERKCoefficients.hpp"
#include "WeightedErrorNorm.hpp"

using namespace std;
using namespace arma;

void setup_and_run_with_problem(Problem* problem, const char* method_name, const char* tol_string, double tol, double H_fine, double H_tol, double H_interval, double eff_rtol) {
	SingleRateOptimalitySearchDriver driver;
	vec atol(problem->problem_dimension, fill::ones);
	atol *= tol/10.0;
	double rtol = tol/10.0;
	WeightedErrorNorm err_norm(&atol, rtol);
	VernerERKCoefficients reference_coeffs;
	FixedDIRKMethod reference_method(
		&reference_coeffs,
		&(problem->full_rhs),
		&(problem->full_rhsjacobian),
		problem->problem_dimension,
		&err_norm
	);
	
	if(strcmp("HeunEulerERK",method_name) == 0) {
		HeunEulerERKCoefficients coeffs;
		FixedDIRKMethod method(
			&coeffs,
			&(problem->full_rhs),
			&(problem->full_rhsjacobian),
			problem->problem_dimension,
			&err_norm
		);
		driver.run(problem, &method, coeffs.name, tol_string, &reference_method, &(problem->y_0), problem->t_0, problem->t_f, tol, H_fine, H_tol, H_interval, eff_rtol);
	} else if(strcmp("BogackiShampineERK",method_name) == 0) {
		BogackiShampineERKCoefficients coeffs;
		FixedDIRKMethod method(
			&coeffs,
			&(problem->full_rhs),
			&(problem->full_rhsjacobian),
			problem->problem_dimension,
			&err_norm
		);
		driver.run(problem, &method, coeffs.name, tol_string, &reference_method, &(problem->y_0), problem->t_0, problem->t_f, tol, H_fine, H_tol, H_interval, eff_rtol);
	} else if(strcmp("DormandPrinceERK",method_name) == 0) {
		DormandPrinceERKCoefficients coeffs;
		FixedDIRKMethod method(
			&coeffs,
			&(problem->full_rhs),
			&(problem->full_rhsjacobian),
			problem->problem_dimension,
			&err_norm
		);
		driver.run(problem, &method, coeffs.name, tol_string, &reference_method, &(problem->y_0), problem->t_0, problem->t_f, tol, H_fine, H_tol, H_interval, eff_rtol);
	} else {
		printf("Error: Did not recognize method name: %s\n",method_name);
	}
}

int main(int argc, char* argv[]) {
	printf("argc: %d\n",argc);
	if(argc != 8) {
		printf("Error: Requires 8 command-line arguments.\n");
		printf("Ex: ./exe/GenericSingleRateOptimalitySearchDriver.exe <ProblemName> <MethodName> <tol> <H_fine> <H_tol> <H_interval> <eff_rtol>\n"); 
		return 1;
	} else {
		double tol = 0.0;
		double H_fine = 0.0;
		double H_tol = 0.0;
		double H_interval = 0.0;
		double eff_rtol = 0.0;

		sscanf(argv[3], "%lf", &tol);
		sscanf(argv[4], "%lf", &H_fine);
		sscanf(argv[5], "%lf", &H_tol);
		sscanf(argv[6], "%lf", &H_interval);
		sscanf(argv[7], "%lf", &eff_rtol);

		const char* problem_name = argv[1];
		const char* method_name = argv[2];
		const char* tol_string = argv[3];

		if(strcmp("Bicoupling",problem_name) == 0) {
			BicouplingProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, H_fine, H_tol, H_interval, eff_rtol);
		} else if(strcmp("BicouplingLN",problem_name) == 0) {
			BicouplingLNProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, H_fine, H_tol, H_interval, eff_rtol);
		} else if(strcmp("Brusselator",problem_name) == 0) {
			BrusselatorProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, H_fine, H_tol, H_interval, eff_rtol);
		} else if(strcmp("FourBody3d",problem_name) == 0) {
			FourBody3dProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, H_fine, H_tol, H_interval, eff_rtol);
		} else if(strcmp("Kaps",problem_name) == 0) {
			KapsProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, H_fine, H_tol, H_interval, eff_rtol);
		} else if(strcmp("KapsLN",problem_name) == 0) {
			KapsLNProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, H_fine, H_tol, H_interval, eff_rtol);
		} else if(strcmp("KPR",problem_name) == 0) {
			KPRProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, H_fine, H_tol, H_interval, eff_rtol);
		} else if(strcmp("Lienard",problem_name) == 0) {
			LienardProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, H_fine, H_tol, H_interval, eff_rtol);
		} else if(strcmp("LienardLN",problem_name) == 0) {
			LienardLNProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, H_fine, H_tol, H_interval, eff_rtol);
		} else if(strcmp("Oregonator",problem_name) == 0) {
			OregonatorProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, H_fine, H_tol, H_interval, eff_rtol);
		} else if(strcmp("OregonatorLN",problem_name) == 0) {
			OregonatorLNProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, H_fine, H_tol, H_interval, eff_rtol);
		} else if(strcmp("Pleiades",problem_name) == 0) {
			PleiadesProblem problem;
			setup_and_run_with_problem(&problem, method_name, tol_string, tol, H_fine, H_tol, H_interval, eff_rtol);
		} else {
			printf("Error: Did not recognize problem name: %s\n", problem_name);
			return 1;
		}
	}
	return 0;
}