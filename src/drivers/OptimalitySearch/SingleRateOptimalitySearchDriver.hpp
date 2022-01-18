#ifndef SINGLERATEOPTIMALITYSEARCHDRIVER_DEFINED__
#define SINGLERATEOPTIMALITYSEARCHDRIVER_DEFINED__

#include <armadillo>
#include <math.h>
#include <iostream>
#include <fstream>

#include "Problem.hpp"
#include "FixedStepMultiRateMethod.hpp"
#include "FixedStepMultiRateStep.hpp"
#include "WeightedErrorNorm.hpp"
#include "FixedDIRKMethod.hpp"
#include "DormandPrinceERKCoefficients.hpp"

using namespace std;
using namespace arma;

struct stats {
	int total_timesteps; int total_successful_timesteps; int total_microtimesteps; 
	int total_successful_microtimesteps; double rel_err; double abs_err;
	int full_function_evals; int full_jacobian_evals; 
	int status;
};

struct FindOptimalReturnValue {
	int status;
};

struct FindHReturnValue {
	double H;
	double eff;
	vec y;
	int full_function_evals;
	int full_jacobian_evals;
	int status;
};

struct ComputeReferenceSolutionReturnValue {
	vec y;
	int status;
};

struct ComputeStepReturnValue {
	double err;
	double eff;
	vec y;
	int full_function_evals;
	int full_jacobian_evals;
	int status;
};

class SingleRateOptimalitySearchDriver {
public:
	FixedDIRKMethod* reference_solver;
	std::vector<double> optimal_H;
	std::vector<double> optimal_eff;
	std::vector<vec> optimal_y;
	std::vector<int> optimal_full_function_evals;
	std::vector<int> optimal_full_jacobian_evals;
	std::vector<double> temp_H;
	std::vector<double> temp_eff;
	std::vector<vec> temp_y;
	std::vector<int> temp_full_function_evals;
	std::vector<int> temp_full_jacobian_evals;
	mat Y_ref;
	vec t_ref;
	FindOptimalReturnValue find_optimal_ret;
	FindHReturnValue find_H_ret;
	ComputeReferenceSolutionReturnValue compute_reference_ret;
	ComputeStepReturnValue compute_step_ret;
	vec y;
	vec* y_ref;
	vec y_sol;
	double t;
	double min_eff;
	int min_eff_idx;
	int n;
	double H;
	int storage_idx;
	int temp_idx;
	int max_storage;

	void run(Problem* problem, FixedDIRKMethod* method, const char* method_name, const char* tol_string, FixedDIRKMethod* reference_method, vec* y_0, double t_0, double t_f, double tol, double H_fine, double H_tol, double H_interval, double eff_rtol) {
		initialize_storage(y_0, t_0, t_f, H_fine);

		int output_index = 0;

		while(t+H_fine < t_f) {
			prepare_find_optimal(t,&y);
			//printf("t: %.16f\n",t);
			find_optimal(problem, method, reference_method, &y, t, t_0, t_f, tol, H_fine, H_tol, H_interval, eff_rtol, &find_optimal_ret);
			if (find_optimal_ret.status == 0) {
				t += optimal_H[output_index];
				//printf("Optimal H: %.16f, Optimal eff: %.16f\n",optimal_H[output_index], optimal_eff[output_index]);
				output_index++;
			} else {
				break;
			}
		}

		//printf("Finished with status: %d\n",find_optimal_ret.status);

		//printf("Optimal H/eff\n");
		for(int i=0; i<optimal_H.size(); i++) {
			//printf("\tH: %.16f, eff: %.16f\n",optimal_H[i],optimal_eff[i]);
		}

		save_output(problem, method_name, tol_string, H_fine, H_tol, H_interval, eff_rtol);
	}

	void initialize_storage(vec* y_0, double t_0, double t_f, double H_fine) {
		y = *y_0;
		t = t_0;
		optimal_H.clear();
		optimal_eff.clear();
		optimal_y.clear();
		temp_H.clear();
		temp_eff.clear();
		temp_y.clear();

		max_storage = (int) std::max(20.0, 50.0*std::ceil(std::log2((t_f-t_0)/H_fine)));
		Y_ref = mat(y_0->n_elem, max_storage, fill::zeros);
		t_ref = vec(max_storage, fill::zeros);
		storage_idx = 0;
	}

	void prepare_find_optimal(double t_, vec* y_) {
		storage_idx = 1;
		t_ref.zeros();
		Y_ref.zeros();

		t_ref(0) = t_;
		Y_ref.col(0) = *y_;

		temp_H.clear();
		temp_eff.clear();
		temp_y.clear();
	}

	void find_optimal(Problem* problem, FixedDIRKMethod* method, FixedDIRKMethod* reference_method, vec* y, double t, double t_0, double t_f, double tol, double H_fine, double H_tol, double H_interval, double eff_rtol, FindOptimalReturnValue* ret) {
		if(temp_eff.empty()) {
			min_eff = 1e10;
		} else {
			min_eff = *std::min_element(temp_eff.begin(), temp_eff.end());
		}

		double H;
		double eff;
		vec y_sol;
		int status;
		find_H(problem, method, reference_method, y, t, t_0, t_f, tol, H_fine, H_tol, H_interval, &find_H_ret);
		H = find_H_ret.H;
		eff = find_H_ret.eff;
		y_sol = find_H_ret.y;
		status = find_H_ret.status;

		if (status == 0) {
			min_eff_idx = std::min_element(temp_eff.begin(),temp_eff.end()) - temp_eff.begin();
			optimal_H.push_back(H);
			optimal_eff.push_back(eff);
			optimal_full_function_evals.push_back(find_H_ret.full_function_evals);
			optimal_full_jacobian_evals.push_back(find_H_ret.full_jacobian_evals);

			t += H;
			*y = y_sol;

			ret->status = 0;
		} else {
			ret->status = status;
		}
	}

	void find_H(Problem* problem, FixedDIRKMethod* method, FixedDIRKMethod* reference_method, 	vec* y, double t, double t_0, double t_f, double tol, double H_fine, double H_tol, double H_interval, FindHReturnValue* ret) {
		double H_left = H_fine;
		compute_reference_solution(problem, reference_method, y, t, H_left, t_0, t_f, &compute_reference_ret);	
		vec* y_ref = &(compute_reference_ret.y);
		int status = compute_reference_ret.status;

		compute_step(problem, method, y, t, H_left, y_ref, &compute_step_ret);
		double err = compute_step_ret.err;
		double eff = compute_step_ret.eff;
		vec y_sol = compute_step_ret.y;

		if(err < tol && status == 0) {
			double H_right = 0.0;
			while(err < tol && t+H_right < t_f && status == 0) {
				H_left = H_right;
				H_right = std::min(H_right + H_interval, t_f-t);

				compute_reference_solution(problem, reference_method, y, t, H_right, t_0, t_f, &compute_reference_ret);	
				y_ref = &(compute_reference_ret.y);
				status = compute_reference_ret.status;

				compute_step(problem, method, y, t, H_right, y_ref, &compute_step_ret);
				err = compute_step_ret.err;
				eff = compute_step_ret.eff;
				y_sol = compute_step_ret.y;
			}

			if ((err > tol || !isfinite(err)) && status == 0) {
				double H_mid = 0.5*(H_left + H_right);
				while((H_right-H_left)/H_mid > H_tol*H_mid && status == 0) {
					H_mid = 0.5*(H_left + H_right);
					compute_reference_solution(problem, reference_method, y, t, H_mid, t_0, t_f, &compute_reference_ret);	
					y_ref = &(compute_reference_ret.y);
					status = compute_reference_ret.status;

					compute_step(problem, method, y, t, H_mid, y_ref, &compute_step_ret);
					err = compute_step_ret.err;
					eff = compute_step_ret.eff;
					y_sol = compute_step_ret.y;

					if(err <= tol) {
						H_left = H_mid;
					} else {
						H_right = H_mid;
					}
				}

				if (status == 0) {
					ret->H = H_left;
					ret->eff = eff;
					ret->y = y_sol;
					ret->full_function_evals = compute_step_ret.full_function_evals;
					ret->full_jacobian_evals = compute_step_ret.full_jacobian_evals;
					ret->status = 0;
				} else {
					ret->status = status;
				}
			} else if (err <= tol && status == 0) {
				ret->H = H_right;
				ret->eff = eff;
				ret->y = y_sol;
				ret->full_function_evals = compute_step_ret.full_function_evals;
				ret->full_jacobian_evals = compute_step_ret.full_jacobian_evals;
				ret->status = 0;
			} else {
				//printf("erroring H_left: %.16f, H_right: %.16f, eff: %.16f, err: %.16f\n",H_left,H_right,eff,err);
				ret->status = 2;
			}
		} else {
			//printf("Failure. Error too large with step of H_fine at t: %.16f. Reduce H_fine.\n",t);
			ret->status = 1;
		}
	}

	void compute_reference_solution(Problem* problem, FixedDIRKMethod* reference_method, vec* y_0, double t, double H, double t_0, double t_f, ComputeReferenceSolutionReturnValue* ret) {
		vec output_tspan = {t+H};
		double H_ref = std::min(H/10.0, (t_f-t_0)/(1000.0));
		mat Y = reference_method->solve(t, H_ref, y_0, &output_tspan);

		ret->y = Y.col(0);
		ret->status = 0;
	}

	void compute_step(Problem* problem, FixedDIRKMethod* method, vec* y_0, double t, double H, vec* y_ref, ComputeStepReturnValue* ret) {
		vec output_tspan = {t+H};
		problem->reset_eval_counts();
		mat Y = method->solve(t, H, y_0, &output_tspan);
		vec y_sol = Y.col(0);

		double err = norm(abs(*y_ref - y_sol),2);
		double cost = problem->full_function_evals;
		double eff = cost/H;

		ret->y = y_sol;
		ret->err = err;
		ret->eff = eff;
		ret->full_function_evals = problem->full_function_evals;
		ret->full_jacobian_evals = problem->full_jacobian_evals;
		ret->status = 0;
	}

	int index_of(vec* t_vec, double t, int storage_idx) {
		for(int i=0; i<storage_idx; i++) {
			if ((*t_vec)(i) == t) {
				return i;
			}
		}

		return -1;
	}

	void save_output(Problem* problem, const char* method_name, const char* tol_string, double H_fine, double H_tol, double H_interval, double eff_rtol) {
		save_input_parameters(problem, method_name, tol_string, H_fine, H_tol, H_interval, eff_rtol);
		save_optimal_output(problem, method_name, tol_string);
	}

	void save_input_parameters(Problem* problem, const char* method_name, const char* tol_string, double H_fine, double H_tol, double H_interval, double eff_rtol) {
		char data[100];
		sprintf(data, "%s,%f,%f,%f,%f",tol_string,H_fine,H_tol,H_interval,eff_rtol);

		char filename[120];
		sprintf(filename, "./resources/OptimalitySearch/%s/%s_OptimalitySearch_%s_%s_parameters.csv",problem->name,problem->name,method_name,tol_string);

		std::ofstream input_parameters_file;
		input_parameters_file.open(filename);
		input_parameters_file << data;
		input_parameters_file.close();
	}

	void save_optimal_output(Problem* problem, const char* method_name, const char* tol_string) {
		mat output(optimal_H.size(),4,fill::zeros);
		for(int i=0; i<output.n_rows; i++) {
			output(i,0) = optimal_H[i];
			output(i,1) = optimal_eff[i];
			output(i,2) = optimal_full_function_evals[i];
			output(i,3) = optimal_full_jacobian_evals[i];
		}

		char filename[120];
		sprintf(filename, "./resources/OptimalitySearch/%s/%s_OptimalitySearch_%s_%s_optimal.csv",problem->name,problem->name,method_name,tol_string);
		output.save(filename, csv_ascii);
	}
};

#endif