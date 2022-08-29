#include <armadillo>
#include <math.h>
#include <stdio.h>

#include "BicouplingProblem.hpp"
#include "BicouplingDLProblem.hpp"
#include "BicouplingLNProblem.hpp"
#include "BrusselatorProblem.hpp"
#include "BrusselatorDLProblem.hpp"
#include "FourBody3dProblem.hpp"
#include "KapsProblem.hpp"
#include "KapsDLProblem.hpp"
#include "KapsLNProblem.hpp"
#include "KPRProblem.hpp"
#include "KPRDLProblem.hpp"
#include "LienardProblem.hpp"
#include "LienardDLProblem.hpp"
#include "LienardLNProblem.hpp"
#include "OregonatorProblem.hpp"
#include "OregonatorDLProblem.hpp"
#include "PleiadesProblem.hpp"
#include "DecoupledLinearProblem.hpp"
#include "Problem.hpp"
#include "RandomTaskDriver.hpp"

using namespace std;
using namespace arma;

mat load_true_sol(const char* problem_name) {
	char filename[100];
	sprintf(filename,"./resources/%s/%s_truesol.csv",problem_name,problem_name);
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
			problem->true_solution(t, &y_true);
			Y_true.col(it) = y_true;
		}
		return Y_true;
	} else {
		return load_true_sol(problem->name);
	}
}

vec generate_H_vec(int total_Hs, double default_H) {
	vec H_vec(total_Hs, fill::zeros);
	for(int ih=0; ih<total_Hs; ih++) {
		H_vec(ih) = default_H*pow(2.0, -ih);
	}
	return H_vec;
}

void setup_and_run(Problem* problem) {
	int problem_dimension = problem->problem_dimension;
	RandomTaskDriver driver;
	int total_Hs = 5;
	vec H_vec = generate_H_vec(total_Hs,problem->default_H);
	vec M_vec = vec("5 10");
	vec output_tspan = linspace(problem->t_0, problem->t_f, 11);
	mat Y_true = get_true_sol(&output_tspan, problem);

	printf("\n%s Problem.\n", problem->name);
	driver.run(problem, &H_vec, &M_vec, &output_tspan, &Y_true);
}

int main(int argc, char* argv[]) {
	if(argc != 2) {
		printf("Error: Requires 1 command-line argument. Ex: GenericFixedDriver.exe <ProblemName>\n");
		return 1;
	} else {
		const char* input_problem_name = argv[1];

		if(strcmp("Bicoupling",input_problem_name) == 0) {
			BicouplingProblem problem;
			setup_and_run(&problem);
		} else if(strcmp("BicouplingDL",input_problem_name) == 0) {
			BicouplingDLProblem problem;
			setup_and_run(&problem);
		} else if(strcmp("BicouplingLN",input_problem_name) == 0) {
			BicouplingLNProblem problem;
			setup_and_run(&problem);
		} else if(strcmp("Brusselator",input_problem_name) == 0) {
			BrusselatorProblem problem;
			setup_and_run(&problem);
		} else if(strcmp("BrusselatorDL",input_problem_name) == 0) {
			BrusselatorDLProblem problem;
			setup_and_run(&problem);
		} else if(strcmp("FourBody3d",input_problem_name) == 0) {
			FourBody3dProblem problem;
			setup_and_run(&problem);
		} else if(strcmp("Kaps",input_problem_name) == 0) {
			KapsProblem problem;
			setup_and_run(&problem);
		} else if(strcmp("KapsDL",input_problem_name) == 0) {
			KapsDLProblem problem;
			setup_and_run(&problem);
		} else if(strcmp("KapsLN",input_problem_name) == 0) {
			KapsLNProblem problem;
			setup_and_run(&problem);
		} else if(strcmp("KPR",input_problem_name) == 0) {
			KPRProblem problem;
			setup_and_run(&problem);
		} else if(strcmp("KPRDL",input_problem_name) == 0) {
			KPRProblem problem;
			setup_and_run(&problem);
		} else if(strcmp("Lienard",input_problem_name) == 0) {
			LienardProblem problem;
			setup_and_run(&problem);
		} else if(strcmp("LienardDL",input_problem_name) == 0) {
			LienardProblem problem;
			setup_and_run(&problem);
		} else if(strcmp("LienardLN",input_problem_name) == 0) {
			LienardProblem problem;
			setup_and_run(&problem);
		} else if(strcmp("Oregonator",input_problem_name) == 0) {
			OregonatorProblem problem;
			setup_and_run(&problem);
		} else if(strcmp("OregonatorDL",input_problem_name) == 0) {
			OregonatorProblem problem;
			setup_and_run(&problem);
		} else if(strcmp("Pleiades",input_problem_name) == 0) {
			PleiadesProblem problem;
			setup_and_run(&problem);
		} else if(strcmp("DecoupledLinear",input_problem_name) == 0) {
			DecoupledLinearProblem problem;
			setup_and_run(&problem);
		} else {
			printf("Error: Did not recognize problem name: %s\n", input_problem_name);
			return 1;
		}
		return 0;
	}
}