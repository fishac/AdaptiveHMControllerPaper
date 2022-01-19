#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>
#include <math.h>

#include "BicouplingProblem.hpp"
#include "BrusselatorProblem.hpp"
#include "FourBody3dProblem.hpp"
#include "KapsProblem.hpp"
#include "KPRProblem.hpp"
#include "ForcedVanderPolProblem.hpp"
#include "PleiadesProblem.hpp"
#include "Problem.hpp"
#include "ControllerTestsDriver.hpp"

using namespace std;
using namespace arma;

mat load_true_sol(const char* problem_name) {
	char filename[100];
	sprintf(filename,"./resources/%s/%s_fixed_truesol_11.csv",problem_name,problem_name);
	mat Y_true;
	bool success = Y_true.load(filename, csv_ascii);
	if (!success) {
		printf("Failed to load true solution.\n");
	}
	return Y_true;
}

mat get_true_sol(vec* output_tspan, Problem* problem) {
	if (problem->has_true_solution) {
		mat Y_true(problem->problem_dimension, output_tspan->n_elem, fill::zeros);
		vec y_true(problem->problem_dimension, fill::zeros);
		double t = 0.0;
		for(int it=0; it<output_tspan->n_elem; it++) {
			t = (*output_tspan)(it);
			(problem->true_solution).evaluate(t, &y_true);
			Y_true.col(it) = y_true;
		}
		return Y_true;
	} else {
		return load_true_sol(problem->name);
	}
}

void setup_and_run(Problem* problem, double tol, const char* tol_string) {
	ControllerTestsDriver driver;
	double H_0 = problem->default_H*std::pow(2,-5.0);
	int M_0 = 10;
	vec atol = tol*vec(problem->problem_dimension,fill::ones);
	double rtol = tol;
	vec output_tspan = linspace(problem->t_0,problem->t_f,11);
	mat Y_true = get_true_sol(&output_tspan, problem);

	printf("\n%s Problem.\n", problem->name);
	driver.run(problem, H_0, M_0, &atol, rtol, &Y_true, &output_tspan, tol_string);
}

int main(int argc, char* argv[]) {
	if(argc != 3) {
		printf("Error: Requires 2 command-line arguments. Ex: GenericControllerTestsDriver.exe <ProblemName> <tol>\n");
		return 1;
	} else {
		double tol = 0.0;

		const char* input_problem_name = argv[1];
		const char* tol_string = argv[2];
		sscanf(argv[2], "%lf", &tol);

		if(strcmp("Bicoupling",input_problem_name) == 0) {
			BicouplingProblem problem;
			setup_and_run(&problem, tol, tol_string);
		} else if(strcmp("Brusselator",input_problem_name) == 0) {
			BrusselatorProblem problem;
			setup_and_run(&problem, tol, tol_string);
		} else if(strcmp("FourBody3d",input_problem_name) == 0) {
			FourBody3dProblem problem;
			setup_and_run(&problem, tol, tol_string);
		} else if(strcmp("Kaps",input_problem_name) == 0) {
			KapsProblem problem;
			setup_and_run(&problem, tol, tol_string);
		} else if(strcmp("KPR",input_problem_name) == 0) {
			KPRProblem problem;
			setup_and_run(&problem, tol, tol_string);
		} else if(strcmp("ForcedVanderPol",input_problem_name) == 0) {
			ForcedVanderPolProblem problem;
			setup_and_run(&problem, tol, tol_string);
		} else if(strcmp("Pleiades",input_problem_name) == 0) {
			PleiadesProblem problem;
			setup_and_run(&problem, tol, tol_string);
		} else {
			printf("Error: Did not recognize problem name: %s\n", input_problem_name);
			return 1;
		}
		return 0;
	}
}
