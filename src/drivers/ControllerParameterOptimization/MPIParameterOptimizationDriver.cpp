#include <armadillo>
#include <math.h>
#include <map>
#include <string>
#include "mpi.h"

#include "BicouplingProblem.hpp"
#include "BrusselatorProblem.hpp"
#include "FourBody3dProblem.hpp"
#include "KapsProblem.hpp"
#include "KPRProblem.hpp"
#include "ForcedVanderPolProblem.hpp"
#include "OregonatorProblem.hpp"
#include "PleiadesProblem.hpp"
#include "Problem.hpp"
#include "MRIGARKERK33Coefficients.hpp"
#include "MRIGARKIRK21aCoefficients.hpp"
#include "MRIGARKERK45aCoefficients.hpp"
#include "MRIGARKESDIRK34aCoefficients.hpp"
#include "HeunEulerERKCoefficients.hpp"
#include "BogackiShampineERKCoefficients.hpp"
#include "DormandPrinceERKCoefficients.hpp"
#include "WeightedErrorNorm.hpp"
#include "ConstantConstantController.hpp"
#include "LinearConstantController.hpp"
#include "LinearLinearController.hpp"
#include "PIMRController.hpp"
#include "QuadraticQuadraticController.hpp"
#include "PIDMRController.hpp"
#include "MRIGARKAdaptiveMethod.hpp"
#include "MRIGARKAdaptiveStepSlowMeasurement.hpp"
#include "MRIGARKAdaptiveStepFastMeasurement.hpp"

using namespace std;
using namespace arma;

struct optimal_stats {
	int slow_function_evals;
	int fast_function_evals;
};

void print_timepointval(double start, double objective_function_value, double* point, int n);
void print_pointval(double objective_function_value, double* point, int n);
void print_threadpoint(int thread_id, double* point, int n);

mat get_true_sol(vec* output_tspan, Problem& problem);
mat load_true_sol(const char* problem_name);

void copy_point(double* src, double* dest, int n);
void copy_point(double* src, double* dest, int n, int src_offset);

double** generate_parameter_space(double half_width, double* centers, double step_size, int n_param, int* n_points_total_);
ConstantConstantController get_CC_controller();
LinearConstantController get_LC_controller();
LinearLinearController get_LL_controller();
PIMRController get_PIMR_controller();
QuadraticQuadraticController get_QQ_controller();
PIDMRController get_PIDMR_controller();

std::map<std::string, optimal_stats> read_optimal_data(Problem** problem_array, int n_problem, 
	MRIGARKCoefficients** coeff_array, int n_mri, 
	std::string* tol_string_array, int n_tol);

void drive_optimization(int n_param, int n_procs);
void evaluate_parameter_space(double** param_space, int n_points_total, int n_param, int n_procs,
		double* min_objective_func_value, double* argmin_point);
void shut_down_workers(int n_param, int n_procs);

void setup_and_evaluate_parameter_points(const char* controller_name, int n_param);
void evaluate_parameter_points(Problem** problem_array, int n_problem, 
	MRIGARKCoefficients** coeff_array, SingleRateMethodCoefficients** inner_coeff_array, int n_mri, 
	double* tol_array, std::string* tol_string_array, int n_tol, 
	const char** measurement_type_array, int n_esf_mt,
	Controller* controller, int n_param, 
	std::map<std::string, optimal_stats>& optimal_data);
double evaluate_parameter_point(Problem** problem_array, int n_problem, MRIGARKCoefficients** coeff_array, SingleRateMethodCoefficients** inner_coeff_array, 
	int n_mri, double* tol_array, std::string* tol_string_array, int n_tol, const char** measurement_type_array, int n_esf_mt,
	Controller* controller, int n_param, std::map<std::string,optimal_stats>& optimal_data, double* point, int myid);

int main(int argc, char* argv[]) {
	int ierr = MPI_Init(&argc, &argv);
	if (ierr != MPI_SUCCESS) {
		std::cerr << "Error in calling MPI_Init\n";
		return 1;
	}

	if(argc != 2) {
		printf("Error: Requires 1 command-line argument.\n");
		printf("Ex: mpiexec -n <n_procs> ./exe/MPIParameterOptimizationDriver.exe <Controller>\n"); 
		return 1;
	} else {
		const char* controller_name = argv[1];

		int n_param, n_procs, my_id;

		if(strcmp("ConstantConstant",controller_name) == 0) {
			n_param = 2;
		} else if(strcmp("LinearConstantIncomplete",controller_name) == 0) {
			n_param = 3;
		} else if(strcmp("LinearConstant",controller_name) == 0) {
			n_param = 3;
		} else if(strcmp("LinearLinear",controller_name) == 0) {
			n_param = 4;
		} else if(strcmp("PIMR",controller_name) == 0) {
			n_param = 4;
		} else if(strcmp("QuadraticQuadratic",controller_name) == 0) {
			n_param = 6;
		} else if(strcmp("PIDMR",controller_name) == 0) {
			n_param = 6;
		} else {
			printf("Error: Did not recognize controller name: %s\n",controller_name);
			return 1;
		}

		ierr = MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
		ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

		if (my_id == 0) {
			drive_optimization(n_param, n_procs);
		} else {
			setup_and_evaluate_parameter_points(controller_name, n_param);
		}
	}
	MPI_Finalize();
}

// Manager main function
void drive_optimization(int n_param, int n_procs) {
	// Set coarse grid settings 
	double optimization_start = MPI_Wtime();

	double centers[n_param];
	for (int i=0; i<n_param; i++) {
		centers[i] = 0.5;
	}

	// Initialize mesh data
	int n_refinements = 3;
	double half_widths[] = { 0.4, 0.2, 0.04 };
	double step_sizes[] = { 0.2, 0.04, 0.02 };
	double half_width, step_size;

	// Initialize optimization values
	double min_objective_func_value = 1e20;
	double argmin_point[n_param];

	for (int i=0; i<n_refinements; i++) {
		half_width = half_widths[i];
		step_size = step_sizes[i];

		int n_points_total = 0;
		double** param_space = generate_parameter_space(half_width, centers, step_size, n_param, &n_points_total);

		// Evaluate the coarse grid, modifying the min_objective_func_value and argmin_point variables to the best options found
		evaluate_parameter_space(param_space, n_points_total, n_param, n_procs,
			&min_objective_func_value, argmin_point);

		// Clean out old param_space
		for(int i=0; i<n_points_total; i++) {
			delete[] param_space[i];
		}
		delete[] param_space;

		// Set next refinement centered on current best point.
		copy_point(argmin_point,centers,n_param);
	}

	printf("Optimization finished. Optimal settings: ");
	print_pointval(min_objective_func_value, argmin_point, n_param);
	printf("\n");

	double optimization_end = MPI_Wtime();
	printf("Total duration (seconds): %.2f\n",optimization_end-optimization_start);

	shut_down_workers(n_param, n_procs);
};

// Manager parameter space evaluation function
void evaluate_parameter_space(double** param_space, int n_points_total, int n_param, int n_procs,
		double* min_objective_func_value, double* argmin_point) {

	double parameter_space_start = MPI_Wtime();

	int points_remaining_to_send = n_points_total;
	double objective_function_value = 0.0;
	int param_idx = 0;
	int tag, ierr, recv_param_idx;
	MPI_Status status;

	// Send initial points to workers
	for (int proc_idx=1; proc_idx < n_procs && param_idx < n_points_total; proc_idx++) {
		tag = param_idx + 1;
		ierr = MPI_Send(param_space[param_idx], n_param, MPI_DOUBLE, proc_idx, tag, MPI_COMM_WORLD);
		if (ierr != MPI_SUCCESS) {
			fprintf(stderr," error in MPI_Send = %i\n",ierr);
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		print_threadpoint(proc_idx, param_space[param_idx], n_param);

		param_idx++;
	}

	// Read in point evalutaions and dole out more work
	while (param_idx < n_points_total) {
		MPI_Recv(&objective_function_value, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		if (ierr != MPI_SUCCESS) {
			fprintf(stderr," error in MPI_Recv = %i\n",ierr);
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		tag = status.MPI_TAG;
		recv_param_idx = tag-1;
		if (objective_function_value < *min_objective_func_value) {
			*min_objective_func_value = objective_function_value;
			copy_point(param_space[recv_param_idx],argmin_point,n_param);
		}
		print_timepointval(parameter_space_start, objective_function_value, param_space[recv_param_idx], n_param);

		tag = param_idx + 1;
		ierr = MPI_Send(param_space[param_idx], n_param, MPI_DOUBLE, status.MPI_SOURCE, tag, MPI_COMM_WORLD);
		if (ierr != MPI_SUCCESS) {
			fprintf(stderr," error in MPI_Send = %i\n",ierr);
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		print_threadpoint(status.MPI_SOURCE, param_space[param_idx], n_param);

		param_idx++;
	}

	// Receive last point evaluations from workers
	param_idx = 0;
	for (int proc_idx=1; proc_idx < n_procs && param_idx < n_points_total; proc_idx++) {
		MPI_Recv(&objective_function_value, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		if (ierr != MPI_SUCCESS) {
			fprintf(stderr," error in MPI_Recv = %i\n",ierr);
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		tag = status.MPI_TAG;
		recv_param_idx = tag-1;
		if (objective_function_value < *min_objective_func_value) {
			*min_objective_func_value = objective_function_value;
			copy_point(param_space[recv_param_idx],argmin_point,n_param);
		}
		print_timepointval(parameter_space_start, objective_function_value, param_space[recv_param_idx], n_param);

		param_idx++;
	}

	double parameter_space_end = MPI_Wtime();
	printf("Parameter space total duration (seconds): %.2f\n",parameter_space_end-parameter_space_start);

	printf("Most optimal point in space: \n");
	print_pointval(*min_objective_func_value, argmin_point, n_param);

}

// Send shut down signal to workers
void shut_down_workers(int n_param, int n_procs) {
	double dummy_point[n_param];
	int ierr;
	for (int proc_idx=1; proc_idx < n_procs; proc_idx++) {
		ierr = MPI_Send(dummy_point, n_param, MPI_DOUBLE, proc_idx, 0, MPI_COMM_WORLD);
		if (ierr != MPI_SUCCESS) {
			fprintf(stderr," error in MPI_Send = %i\n",ierr);
			MPI_Abort(MPI_COMM_WORLD,1);
		}
	}
}

// Worker setup function
void setup_and_evaluate_parameter_points(const char* controller_name, int n_param) {

	BicouplingProblem bicoupling_problem;
	BrusselatorProblem brusselator_problem;
	KapsProblem kaps_problem;
	KPRProblem kpr_problem;
	ForcedVanderPolProblem forcedvanderpol_problem;
	PleiadesProblem pleiades_problem;
	FourBody3dProblem fourbody3d_problem;

	Problem* problem_array[] = { 
		&bicoupling_problem, &brusselator_problem, &kaps_problem,
		&kpr_problem, &forcedvanderpol_problem, &pleiades_problem, 
		&fourbody3d_problem
	};
	int n_problem = 7;

	BogackiShampineERKCoefficients bogacki_shampine;
	HeunEulerERKCoefficients heun_euler;
	DormandPrinceERKCoefficients dormand_prince;

	SingleRateMethodCoefficients* inner_coeff_array[] = { 
		&bogacki_shampine,
		&heun_euler,
		&dormand_prince,
		&bogacki_shampine
	};

	MRIGARKERK33Coefficients mrigarkerk33;
	MRIGARKIRK21aCoefficients mrigarkirk21a;
	MRIGARKERK45aCoefficients mrigarkerk45a;
	MRIGARKESDIRK34aCoefficients mrigarkesdirk34a;

	MRIGARKCoefficients* coeff_array[] = {
		&mrigarkerk33,
		&mrigarkirk21a,
		&mrigarkerk45a,
		&mrigarkesdirk34a
	};
	int n_mri = 4;

	double tol_array[] = { 1e-3, 1e-5, 1e-7 };
	std::string tol_string_array[] = { "1e-3", "1e-5", "1e-7" };
	int n_tol = 3;

	const char** measurement_type_array = FastError::types;
	int n_esf_mt = 5;

	// Read optimal data
	std::map<std::string, optimal_stats> optimal_data = read_optimal_data(problem_array,n_problem,coeff_array,n_mri,tol_string_array,n_tol);

	if(strcmp("ConstantConstant",controller_name) == 0) {
		ConstantConstantController controller = get_CC_controller();
		evaluate_parameter_points(problem_array, n_problem, coeff_array, inner_coeff_array, n_mri, 
			tol_array, tol_string_array, n_tol, measurement_type_array, n_esf_mt,
			&controller, n_param, optimal_data);
	} else if(strcmp("LinearConstant",controller_name) == 0) {
		LinearConstantController controller = get_LC_controller();
		evaluate_parameter_points(problem_array, n_problem, coeff_array, inner_coeff_array, n_mri, 
			tol_array, tol_string_array, n_tol, measurement_type_array, n_esf_mt,
			&controller, n_param, optimal_data);
	} else if(strcmp("LinearLinear",controller_name) == 0) {
		LinearLinearController controller = get_LL_controller();
		evaluate_parameter_points(problem_array, n_problem, coeff_array, inner_coeff_array, n_mri, 
			tol_array, tol_string_array, n_tol, measurement_type_array, n_esf_mt,
			&controller, n_param, optimal_data);
	} else if(strcmp("PIMR",controller_name) == 0) {
		PIMRController controller = get_PIMR_controller();
		evaluate_parameter_points(problem_array, n_problem, coeff_array, inner_coeff_array, n_mri, 
			tol_array, tol_string_array, n_tol, measurement_type_array, n_esf_mt,
			&controller, n_param, optimal_data);
	} else if(strcmp("QuadraticQuadratic",controller_name) == 0) {
		QuadraticQuadraticController controller = get_QQ_controller();
		evaluate_parameter_points(problem_array, n_problem, coeff_array, inner_coeff_array, n_mri, 
			tol_array, tol_string_array, n_tol, measurement_type_array, n_esf_mt,
			&controller, n_param, optimal_data);
	} else if(strcmp("PIDMR",controller_name) == 0) {
		PIDMRController controller = get_PIDMR_controller();
		evaluate_parameter_points(problem_array, n_problem, coeff_array, inner_coeff_array, n_mri, 
			tol_array, tol_string_array, n_tol, measurement_type_array, n_esf_mt,
			&controller, n_param, optimal_data);
	}
}

// Worker main driver function
void evaluate_parameter_points(Problem** problem_array, int n_problem, 
	MRIGARKCoefficients** coeff_array, SingleRateMethodCoefficients** inner_coeff_array, int n_mri, 
	double* tol_array, std::string* tol_string_array, int n_tol, 
	const char** measurement_type_array, int n_esf_mt,
	Controller* controller, int n_param, 
	std::map<std::string, optimal_stats>& optimal_data) {

	int more_work = 1;
	int tag, ierr;
	double point[n_param];
	MPI_Status status;

	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	while (more_work) {
		ierr = MPI_Recv(point, n_param, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		if (ierr != MPI_SUCCESS) {
			fprintf(stderr," error in MPI_Recv = %i\n",ierr);
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		tag = status.MPI_TAG;

		if (tag == 0) {
			more_work = 0;
		} else {
			double objective_function_value = evaluate_parameter_point(problem_array, n_problem, coeff_array, inner_coeff_array, 
				n_mri,tol_array, tol_string_array, n_tol, measurement_type_array, n_esf_mt,
				controller, n_param, optimal_data, point, myid);

			ierr = MPI_Send(&objective_function_value, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
			if (ierr != MPI_SUCCESS) {
				fprintf(stderr," error in MPI_Send = %i\n",ierr);
				MPI_Abort(MPI_COMM_WORLD,1);
			}
		}
	}
}

// Worker evaluate function 
double evaluate_parameter_point(Problem** problem_array, int n_problem, MRIGARKCoefficients** coeff_array, SingleRateMethodCoefficients** inner_coeff_array, 
	int n_mri, double* tol_array, std::string* tol_string_array, int n_tol, const char** measurement_type_array, int n_esf_mt,
	Controller* controller, int n_param, std::map<std::string,optimal_stats>& optimal_data, double* point, int myid) {
	double objective_function_value = 0.0;

	// Extract controller parameter values from point
	double k1[3] = { 0.0, 0.0, 0.0 };
	double k2[3] = { 0.0, 0.0, 0.0 };
	if (n_param == 2) {
		copy_point(point,k1,1,0);
		copy_point(point,k2,1,1);
	} else if (n_param == 3) {
		copy_point(point,k1,2,0);
		copy_point(point,k2,1,2);
	} else if (n_param == 4) {
		if (strcmp("QuadraticConstant",controller->name) == 0) {
			copy_point(point,k1,3,0);
			copy_point(point,k2,1,3);
		} else { // LinearLinear or PIMR
			copy_point(point,k1,2,0);
			copy_point(point,k2,2,2);
		}
	} else if (n_param == 5) {
		copy_point(point,k1,3,0);
		copy_point(point,k2,2,3);
	} else if (n_param == 6) {
		copy_point(point,k1,3,0);
		copy_point(point,k2,3,3);
	}
	printf("Worker %d working on k1: %.2f %.2f %.2f, k2: %.2f %.2f %.2f\n",myid,k1[0],k1[1],k1[2],k2[0],k2[1],k2[2]);
	controller->set_parameters(k1, k2);

	// Iterate over each problem
	for(int i=0; i<n_problem; i++) {
		// Set up problem-dependent values
		std::string problem_name = problem_array[i]->name;
		vec output_tspan = linspace(problem_array[i]->t_0,problem_array[i]->t_f,11);
		mat Y_true = get_true_sol(&output_tspan, *(problem_array[i]));
		double H_0 = problem_array[i]->default_H*std::pow(2,-5.0);
		int M_0 = 10;

		// Iterate over each method
		for(int j=0; j<n_mri; j++) {
			// Skip method if problem is explicit only and method is implicit
			if (problem_array[i]->explicit_only && !coeff_array[j]->explicit_mrigark) {
				continue;
			}
			// Set up method-dependent values
			std::string method_name = coeff_array[j]->name;
			double P = std::min(coeff_array[j]->primary_order,coeff_array[j]->secondary_order);
			double p = std::min(inner_coeff_array[j]->primary_order,inner_coeff_array[j]->secondary_order);
			controller->set_orders(P,p);
			controller->reset();

			// Iterate over each tolerance
			for (int k=0; k<n_tol; k++) {
				// Iterate over each measurement_type
				for (int l=0; l<n_esf_mt; l++) {
					// This block must be in the inner loop.
					std::string tol_string = tol_string_array[k];
					vec atol = tol_array[k]*vec(problem_array[i]->problem_dimension,fill::ones);
					double rtol = tol_array[k];
					WeightedErrorNorm err_norm(&atol, rtol);
					MRIGARKAdaptiveMethod mrigark_method(
						problem_array[i],
						problem_array[i]->problem_dimension,
						&err_norm
					);
					std::string key = problem_name + "_" + method_name + "_" + tol_string;


					AdaptiveMultiRateMethodReturnValue ret;
					err_norm.reset_weights();
					controller->reset();
					if (FastError::is_slow_type(measurement_type_array[l])) {
						MRIGARKAdaptiveStepSlowMeasurement mrigark_step_slow_measurement(
							coeff_array[j], 
							inner_coeff_array[j], 
							&(problem_array[i]->fast_rhs), 
							&(problem_array[i]->slow_rhs), 
							&(problem_array[i]->fast_rhsjacobian), 
							&(problem_array[i]->slow_rhsjacobian), 
							problem_array[i]->problem_dimension,
							&err_norm
						);
						mrigark_method.solve(problem_array[i]->t_0, H_0, M_0, &(problem_array[i]->y_0), 
							&output_tspan, &mrigark_step_slow_measurement, controller, measurement_type_array[l], false, 
							&ret);
					} else {
						MRIGARKAdaptiveStepFastMeasurement mrigark_step_fast_measurement(
							coeff_array[j], 
							inner_coeff_array[j], 
							&(problem_array[i]->fast_rhs), 
							&(problem_array[i]->slow_rhs), 
							&(problem_array[i]->fast_rhsjacobian), 
							&(problem_array[i]->slow_rhsjacobian), 
							problem_array[i]->problem_dimension,
							&err_norm
						);
						mrigark_method.solve(problem_array[i]->t_0, H_0, M_0, &(problem_array[i]->y_0), 
							&output_tspan, &mrigark_step_fast_measurement, controller, measurement_type_array[l], false, 
							&ret);
					}

					if (ret.status == 0) {
						mat Y = ret.Y;
						double err = abs(Y_true-Y).max();
						double cost = 10*problem_array[i]->slow_function_evals + problem_array[i]->slow_function_evals;
						double optimal_cost = 10*optimal_data[key].slow_function_evals+optimal_data[key].fast_function_evals;

						objective_function_value += cost/optimal_cost + 100.0*std::pow(log10(err/tol_array[k]),2.0);
					} else {
						objective_function_value += 1e10;
					}
					problem_array[i]->reset_eval_counts();
				}
			}
		}
	}
	return objective_function_value;
}

void print_timepointval(double start, double objective_function_value, double* point, int n) {
	double end = MPI_Wtime();
	printf("(%.2fs), ",end-start);
	printf("[");
	for (int i=0; i<n; i++) {
		printf("%.3f",point[i]);
		if (i != n-1) {
			printf(",");
		}
	}
	printf("] = %.16f\n",objective_function_value);
	fflush(stdout);
}

void print_pointval(double objective_function_value, double* point, int n) {
	printf("[");
	for (int i=0; i<n; i++) {
		printf("%.3f",point[i]);
		if (i != n-1) {
			printf(",");
		}
	}
	printf("] = %.16f\n",objective_function_value);
	fflush(stdout);
}

void print_threadpoint(int thread_id, double* point, int n) {
	printf("Rank %d working on parameter point [",thread_id);
	for (int i=0; i<n; i++) {
		printf("%.3f",point[i]);
		if (i != n-1) {
			printf(",");
		}
	}
	printf("]\n");
	fflush(stdout);
}

mat get_true_sol(vec* output_tspan, Problem& problem) {
	if (problem.has_true_solution) {
		mat Y_true(problem.problem_dimension, output_tspan->n_elem, fill::zeros);
		vec y_true(problem.problem_dimension, fill::zeros);
		double t = 0.0;
		for(int it=0; it<output_tspan->n_elem; it++) {
			t = (*output_tspan)(it);
			(problem.
				true_solution).evaluate(t, &y_true);
			Y_true.col(it) = y_true;
		}
		return Y_true;
	} else {
		return load_true_sol(problem.name);
	}
}

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

double** generate_parameter_space(double half_width, double* centers, double step_size, int n_param, int* n_points_total_) {
	// Declare the parameter range data
	std::vector<std::vector<double>> param_ranges;
	int n_points_1d_max = 2*half_width/step_size+1;
	double lower_bound_cutoff = 0.0;
	double upper_bound_cutoff = 1.0;

	// Initialize the 1d vectors of parameter ranges for each parameter.
	// Memory: <= (n_param * n_points_1d_max) doubles 
	double center, lower_bound, coord;
	for(int i=0; i<n_param; i++) {
		center = centers[i];
		std::vector<double> param_range;
		lower_bound = centers[i]-half_width;
		for (int j=0; j<n_points_1d_max; j++) {
			coord = lower_bound + j*step_size;
			// Filter out param coordinates outside desired range
			if (coord > lower_bound_cutoff && coord <= upper_bound_cutoff) {
				param_range.push_back(lower_bound + j*step_size);
			}
		}
		param_ranges.push_back(param_range);
	}

	// DELETE THIS, JUST FOR VERIFICATION
	//for (int i=0; i<n_points_1d; i++) {
	//	for (int j=0; j<n_param; j++) {
	//		printf("%.3f,",param_ranges[j][i]);
	//	}
	//	printf("\n");
	//}

	// Set total points in space
	int n_points_total = 1;
	for(int i=0; i<n_param; i++) {
		n_points_total *= param_ranges[i].size();
	}
	*n_points_total_ = n_points_total;

	// Initialize the parameter space memory to all 0s
	// Memory: <= (n_param * n_points_1d^n_param) doubles
	//	       (n_param * prod(param_range[i].size(),i,1,n_param)) doubles
	// 1 double for each parameter (n_params), for every point in the space 
	double** param_space = new double*[n_points_total];
	for (int i=0; i<n_points_total; i++) {
		param_space[i] = new double[n_param];
		for (int j=0; j<n_param; j++) {
			param_space[i][j] = 0.0;		
		}
	}

	int space_point_idx = 0;
	int range_point_idx = 0;
	int param_idx = 0;
	int num_ranges = n_points_total;
	int num_values_in_a_row = 1;
	int n_points_1d = 0;
	for (int i=0; i<n_param; i++) {
		param_idx = i;
		num_ranges = num_ranges/param_ranges[i].size();
		n_points_1d = param_ranges[i].size();
		space_point_idx = 0;
		
		for (int j=0; j<num_ranges; j++) {
			for (int l=0; l<n_points_1d; l++) {
				range_point_idx = l;
				for (int k=0; k<num_values_in_a_row; k++) {
					param_space[space_point_idx][param_idx] = param_ranges[param_idx][range_point_idx];
					space_point_idx++;
				}
			}
		}
		num_values_in_a_row = num_values_in_a_row*param_ranges[i].size();
	}

	// DELETE THIS, JUST FOR VERIFICATION
	//for (int i=0; i<n_points_total; i++) {
	//	for (int j=0; j<n_param; j++) {
	//		printf("%.3f,",param_space[i][j]);
	//	}
	//	printf("\n");
	//}

	return param_space;
};

std::map<std::string, optimal_stats> read_optimal_data(Problem** problem_array, int n_problem, 
	MRIGARKCoefficients** coeff_array, int n_mri, 
	std::string* tol_string_array, int n_tol) {
	std::map<std::string, optimal_stats> optimal_data_map;
	std::string problem_name;
	std::string method_name;
	std::string tol_string;

	for(int i=0; i<n_problem; i++) {
		problem_name = problem_array[i]->name;
		for(int j=0; j<n_mri; j++) {
			if (problem_array[i]->explicit_only && !coeff_array[j]->explicit_mrigark) {
				continue;
			}
			method_name = coeff_array[j]->name;
			for (int k=0; k<n_tol; k++) {
				tol_string = tol_string_array[k];

				std::string filename = "./resources/OptimalitySearch/" + problem_name + "/" + problem_name
				+ "_OptimalitySearch_" + method_name + "_" +tol_string
				+ "_10_optimal.csv";
				std::string key = problem_name + "_" + method_name + "_" + tol_string;

				mat optimal_data;
				optimal_data.load(filename, csv_ascii);
				int slow_function_evals = accu(optimal_data.col(3));
				int fast_function_evals = accu(optimal_data.col(4));

				struct optimal_stats stat = {
					slow_function_evals,
					fast_function_evals
				};
				optimal_data_map[key] = stat;
			}
		}
	}

	// DELETE THIS, JUST FOR VERIFICATION
	//for (const auto& [key, value] : optimal_data_map) {
	//	std::cout << key << ". slow: " << value.slow_function_evals << ", fast: " << value.fast_function_evals << std::endl;
	//}

	return optimal_data_map;
};

void copy_point(double* src, double* dest, int n) {
	for (int i=0; i<n; i++) {
		dest[i] = src[i];
	}
}

void copy_point(double* src, double* dest, int n, int src_offset) {
	for (int i=0; i<n; i++) {
		dest[i] = src[i+src_offset];
	}
}

ConstantConstantController get_CC_controller() {
	double k1_CC[] = { 0.22 };
	double k2_CC[] = { 0.18 };
	ConstantConstantController controller(
		1.0,
		1.0,
		1.0,
		0.85,
		k1_CC,
		k2_CC
	);
	return controller;
}

LinearConstantController get_LC_controller() {
	double k1_CC[] = { 0.22 };
	double k2_CC[] = { 0.18 };
	ConstantConstantController* initial_controller = new ConstantConstantController(
		1.0,
		1.0,
		1.0,
		0.85,
		k1_CC,
		k2_CC
	);

	double k1_LC[] = { 1.0, 1.0 };
	double k2_LC[] = { 1.0 };
	LinearConstantController controller(
		1.0,
		1.0,
		1.0,
		0.85,
		k1_LC,
		k2_LC,
		initial_controller
	);
	return controller;
} 

LinearLinearController get_LL_controller() {
	double k1_CC[] = { 0.22 };
	double k2_CC[] = { 0.18 };
	ConstantConstantController* initial_controller = new ConstantConstantController(
		1.0,
		1.0,
		1.0,
		0.85,
		k1_CC,
		k2_CC
	);

	double k1_LL[] = { 1.0, 1.0 };
	double k2_LL[] = { 1.0, 1.0 };
	LinearLinearController controller(
		1.0,
		1.0,
		1.0,
		0.85,
		k1_LL,
		k2_LL,
		initial_controller
	);
	return controller;
} 

PIMRController get_PIMR_controller() {
	double k1_CC[] = { 0.22 };
	double k2_CC[] = { 0.18 };
	ConstantConstantController* initial_controller = new ConstantConstantController(
		1.0,
		1.0,
		1.0,
		0.85,
		k1_CC,
		k2_CC
	);

	double k1_PIMR[] = { 1.0, 1.0 };
	double k2_PIMR[] = { 1.0, 1.0 };
	PIMRController controller(
		1.0,
		1.0,
		1.0,
		0.85,
		k1_PIMR,
		k2_PIMR,
		initial_controller
	);
	return controller;
} 

QuadraticQuadraticController get_QQ_controller() {
	double k1_CC[] = { 0.22 };
	double k2_CC[] = { 0.18 };
	ConstantConstantController* initial_controller = new ConstantConstantController(
		1.0,
		1.0,
		1.0,
		0.85,
		k1_CC,
		k2_CC
	);

	double k1_QQ[] = { 1.0, 1.0, 1.0 };
	double k2_QQ[] = { 1.0, 1.0, 1.0 };
	QuadraticQuadraticController controller(
		1.0,
		1.0,
		1.0,
		0.85,
		k1_QQ,
		k2_QQ,
		initial_controller
	);
	return controller;
} 

PIDMRController get_PIDMR_controller() {
	double k1_CC[] = { 0.22 };
	double k2_CC[] = { 0.18 };
	ConstantConstantController* initial_controller = new ConstantConstantController(
		1.0,
		1.0,
		1.0,
		0.85,
		k1_CC,
		k2_CC
	);

	double k1_PIMR[] = { 1.0, 1.0, 1.0 };
	double k2_PIMR[] = { 1.0, 1.0, 1.0 };
	PIDMRController controller(
		1.0,
		1.0,
		1.0,
		0.85,
		k1_PIMR,
		k2_PIMR,
		initial_controller
	);
	return controller;
} 
