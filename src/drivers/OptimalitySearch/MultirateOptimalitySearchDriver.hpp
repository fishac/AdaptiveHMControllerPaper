#ifndef MULTIRATEOPTIMALITYSEARCHDRIVER_DEFINED__
#define MULTIRATEOPTIMALITYSEARCHDRIVER_DEFINED__

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
	int fast_function_evals; int slow_function_evals; int implicit_function_evals;
	int explicit_function_evals; int fast_jacobian_evals; int slow_jacobian_evals; int implicit_jacobian_evals;
	int status;
};

struct FindOptimalReturnValue {
	int status;
};

struct FindHReturnValue {
	double H;
	double eff;
	vec y;
	int slow_function_evals;
	int fast_function_evals;
	int implicit_function_evals;
	int explicit_function_evals;
	int slow_jacobian_evals;
	int fast_jacobian_evals;
	int implicit_jacobian_evals;
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
	int slow_function_evals;
	int fast_function_evals;
	int implicit_function_evals;
	int explicit_function_evals;
	int slow_jacobian_evals;
	int fast_jacobian_evals;
	int implicit_jacobian_evals;
	int status;
};

class MultirateOptimalitySearchDriver {
public:
	FixedDIRKMethod* reference_solver;
	std::vector<double> optimal_H;
	std::vector<int> optimal_M;
	std::vector<double> optimal_eff;
	std::vector<vec> optimal_y;
	std::vector<int> optimal_slow_function_evals;
	std::vector<int> optimal_fast_function_evals;
	std::vector<int> optimal_implicit_function_evals;
	std::vector<int> optimal_explicit_function_evals;
	std::vector<int> optimal_slow_jacobian_evals;
	std::vector<int> optimal_fast_jacobian_evals;
	std::vector<int> optimal_implicit_jacobian_evals;
	std::vector<double> temp_H;
	std::vector<int> temp_M;
	std::vector<double> temp_eff;
	std::vector<vec> temp_y;
	std::vector<int> temp_slow_function_evals;
	std::vector<int> temp_fast_function_evals;
	std::vector<int> temp_implicit_function_evals;
	std::vector<int> temp_explicit_function_evals;
	std::vector<int> temp_slow_jacobian_evals;
	std::vector<int> temp_fast_jacobian_evals;
	std::vector<int> temp_implicit_jacobian_evals;
	std::vector<std::vector<double>> near_optimal_H;
	std::vector<std::vector<int>> near_optimal_M;
	std::vector<std::vector<double>> near_optimal_eff; 
	std::vector<std::vector<vec>> near_optimal_y; 
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
	int M;
	double max_eff;
	int max_eff_idx;
	int n;
	double H;
	int storage_idx;
	int temp_idx;
	int max_storage;

	void run(Problem* problem, FixedStepMultiRateMethod* method, FixedStepMultiRateStep* step, const char* method_name, const char* tol_string, const char* spf_string, FixedDIRKMethod* reference_method, vec* y_0, double t_0, double t_f, double tol, double slow_penalty_factor, double H_fine, double H_tol, double H_interval, int M_max_iter, int M_min_iter, double eff_rtol) {
		initialize_storage(y_0, t_0, t_f, H_fine);

		int output_index = 0;

		while(t+H_fine < t_f) {
			prepare_find_optimal(t,&y);
			//printf("t: %.16f\n",t);
			find_optimal(problem, method, step, reference_method, &y, t, t_0, t_f, tol, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol, &find_optimal_ret);
			if (find_optimal_ret.status == 0) {
				t += optimal_H[output_index];
				printf("Optimal H: %.16f, Optimal M: %d, Optimal eff: %.16f\n",optimal_H[output_index], optimal_M[output_index], optimal_eff[output_index]);
				output_index++;
			} else {
				break;
			}
		}

		//printf("Finished with status: %d\n",find_optimal_ret.status);

		printf("Optimal H/M/eff\n");
		for(int i=0; i<optimal_H.size(); i++) {
			printf("\tH: %.16f, M: %d, eff: %.16f\n",optimal_H[i],optimal_M[i],optimal_eff[i]);
		}

		save_output(problem, method_name, tol_string, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
	}

	void initialize_storage(vec* y_0, double t_0, double t_f, double H_fine) {
		y = *y_0;
		t = t_0;
		optimal_H.clear();
		optimal_M.clear();
		optimal_eff.clear();
		optimal_y.clear();
		temp_H.clear();
		temp_M.clear();
		temp_eff.clear();
		temp_y.clear();
		near_optimal_H.clear();
		near_optimal_M.clear();
		near_optimal_eff.clear();
		near_optimal_y.clear();

		max_storage = (int) std::max(20.0, 50.0*std::ceil(std::log2((t_f-t_0)/H_fine)));
		Y_ref = mat(y_0->n_elem, max_storage, fill::zeros);
		t_ref = vec(max_storage, fill::zeros);
		storage_idx = 0;
	}

	void prepare_find_optimal(double t_, vec* y_) {
		M = 1;
		storage_idx = 1;
		t_ref.zeros();
		Y_ref.zeros();

		t_ref(0) = t_;
		Y_ref.col(0) = *y_;

		temp_H.clear();
		temp_M.clear();
		temp_eff.clear();
		temp_y.clear();
	}

	void find_optimal(Problem* problem, FixedStepMultiRateMethod* method, FixedStepMultiRateStep* step, FixedDIRKMethod* reference_method, vec* y, double t, double t_0, double t_f, double tol, double slow_penalty_factor, double H_fine, double H_tol, double H_interval, int M_max_iter, int M_min_iter, double eff_rtol, FindOptimalReturnValue* ret) {
		if(temp_eff.empty()) {
			max_eff = -1.0;
		} else {
			max_eff = *std::max_element(temp_eff.begin(), temp_eff.end());
		}

		double H;
		double eff;
		vec y_sol;
		int status;
		int has_set_max_eff = 0;
		while(M < M_max_iter) {
			find_H(problem, method, step, reference_method, y, t, t_0, t_f, M, tol, slow_penalty_factor, H_fine, H_tol, H_interval, &find_H_ret);
			H = find_H_ret.H;
			eff = find_H_ret.eff;
			y_sol = find_H_ret.y;
			status = find_H_ret.status;

			if (status == 0) {
				//printf("\tM: %d, H: %.16f, eff: %.16f\n",M,H,eff);
				if (has_set_max_eff && (max_eff - eff)/max_eff > eff_rtol && M >= M_min_iter) {
					break;
				} else {
					temp_H.push_back(H);
					temp_M.push_back(M);
					temp_eff.push_back(eff);
					temp_y.push_back(y_sol);
					temp_slow_function_evals.push_back(find_H_ret.slow_function_evals);
					temp_fast_function_evals.push_back(find_H_ret.fast_function_evals);
					temp_implicit_function_evals.push_back(find_H_ret.implicit_function_evals);
					temp_explicit_function_evals.push_back(find_H_ret.explicit_function_evals);
					temp_slow_jacobian_evals.push_back(find_H_ret.slow_jacobian_evals);
					temp_fast_jacobian_evals.push_back(find_H_ret.fast_jacobian_evals);
					temp_implicit_jacobian_evals.push_back(find_H_ret.implicit_jacobian_evals);

					if(eff > max_eff) {
						max_eff = eff;
					}
					has_set_max_eff = 1;
					M++;
				}
			} else {
				break;
			}
		}

		if (status == 0) {
			max_eff_idx = std::max_element(temp_eff.begin(),temp_eff.end()) - temp_eff.begin();
			optimal_H.push_back(temp_H[max_eff_idx]);
			optimal_M.push_back(temp_M[max_eff_idx]);
			optimal_eff.push_back(temp_eff[max_eff_idx]);
			optimal_slow_function_evals.push_back(temp_slow_function_evals[max_eff_idx]);
			optimal_fast_function_evals.push_back(temp_fast_function_evals[max_eff_idx]);
			optimal_implicit_function_evals.push_back(temp_implicit_function_evals[max_eff_idx]);
			optimal_explicit_function_evals.push_back(temp_explicit_function_evals[max_eff_idx]);
			optimal_slow_jacobian_evals.push_back(temp_slow_jacobian_evals[max_eff_idx]);
			optimal_fast_jacobian_evals.push_back(temp_fast_jacobian_evals[max_eff_idx]);
			optimal_implicit_jacobian_evals.push_back(temp_implicit_jacobian_evals[max_eff_idx]);

			t += temp_H[max_eff_idx];
			*y = temp_y[max_eff_idx];

			find_near_optimal_parameters(&temp_H, &temp_M, &temp_eff, max_eff, eff_rtol);

			ret->status = 0;
		} else {
			ret->status = status;
		}
	}

	void find_H(Problem* problem, FixedStepMultiRateMethod* method, FixedStepMultiRateStep* step, FixedDIRKMethod* reference_method, 	vec* y, double t, double t_0, double t_f, int M, double tol, double slow_penalty_factor, double H_fine, double H_tol, double H_interval, FindHReturnValue* ret) {
		double H_left = H_fine;
		compute_reference_solution(problem, reference_method, y, t, H_left, M, t_0, t_f, &compute_reference_ret);	
		vec* y_ref = &(compute_reference_ret.y);
		int status = compute_reference_ret.status;

		compute_step(problem, method, step, y, t, H_left, M, y_ref, slow_penalty_factor, &compute_step_ret);
		double err = compute_step_ret.err;
		double eff = compute_step_ret.eff;
		vec y_sol = compute_step_ret.y;

		if(err < tol && status == 0) {
			double H_right = 0.0;
			while(err < tol && t+H_right < t_f && status == 0) {
				H_left = H_right;
				H_right = std::min(H_right + H_interval, t_f-t);

				compute_reference_solution(problem, reference_method, y, t, H_right, M, t_0, t_f, &compute_reference_ret);	
				y_ref = &(compute_reference_ret.y);
				status = compute_reference_ret.status;

				compute_step(problem, method, step, y, t, H_right, M, y_ref, slow_penalty_factor, &compute_step_ret);
				err = compute_step_ret.err;
				eff = compute_step_ret.eff;
				y_sol = compute_step_ret.y;
			}

			if ((err > tol || !isfinite(err)) && status == 0) {
				double H_mid = 0.5*(H_left + H_right);
				while((H_right-H_left)/H_mid > H_tol/**H_mid*/ && status == 0) {
					H_mid = 0.5*(H_left + H_right);
					compute_reference_solution(problem, reference_method, y, t, H_mid, M, t_0, t_f, &compute_reference_ret);	
					y_ref = &(compute_reference_ret.y);
					status = compute_reference_ret.status;

					compute_step(problem, method, step, y, t, H_mid, M, y_ref, slow_penalty_factor, &compute_step_ret);
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
					ret->slow_function_evals = compute_step_ret.slow_function_evals;
					ret->fast_function_evals = compute_step_ret.fast_function_evals;
					ret->implicit_function_evals = compute_step_ret.implicit_function_evals;
					ret->explicit_function_evals = compute_step_ret.explicit_function_evals;
					ret->slow_jacobian_evals = compute_step_ret.slow_jacobian_evals;
					ret->fast_jacobian_evals = compute_step_ret.fast_jacobian_evals;
					ret->implicit_jacobian_evals = compute_step_ret.implicit_jacobian_evals;
					ret->status = 0;
				} else {
					ret->status = status;
				}
			} else if (err <= tol && status == 0) {
				ret->H = H_right;
				ret->eff = eff;
				ret->y = y_sol;
				ret->slow_function_evals = compute_step_ret.slow_function_evals;
				ret->fast_function_evals = compute_step_ret.fast_function_evals;
				ret->implicit_function_evals = compute_step_ret.implicit_function_evals;
				ret->explicit_function_evals = compute_step_ret.explicit_function_evals;
				ret->slow_jacobian_evals = compute_step_ret.slow_jacobian_evals;
				ret->fast_jacobian_evals = compute_step_ret.fast_jacobian_evals;
				ret->implicit_jacobian_evals = compute_step_ret.implicit_jacobian_evals;
				ret->status = 0;
			} else {
				//printf("erroring H_left: %.16f, H_right: %.16f, M: %d, eff: %.16f, err: %.16f\n",H_left,H_right,M,eff,err);
				ret->status = 2;
			}
		} else {
			//printf("Failure. Error too large with step of H_fine at t: %.16f. Reduce H_fine.\n",t);
			ret->status = 1;
		}
	}

	void find_near_optimal_parameters(std::vector<double>* H, std::vector<int>* M, std::vector<double>* eff, double max_eff, double eff_rtol) {
		for(int i=eff->size()-1; i>=0; i--) {
			if(!((max_eff - (*eff)[i])/max_eff > eff_rtol)) { 
				H->erase(H->begin() + i);
				M->erase(M->begin() + i);
				eff->erase(eff->begin() + i);
			}
		}

		near_optimal_H.push_back(*H);
		near_optimal_M.push_back(*M);
		near_optimal_eff.push_back(*eff);
	}

	void compute_reference_solution(Problem* problem, FixedDIRKMethod* reference_method, vec* y_0, double t, double H, int M, double t_0, double t_f, ComputeReferenceSolutionReturnValue* ret) {
		vec output_tspan = {t+H};
		double H_ref = std::min(H/10.0, (t_f-t_0)/(1000.0*M));
		mat Y = reference_method->solve(t, H_ref, y_0, &output_tspan);

		ret->y = Y.col(0);
		ret->status = 0;
	}

	void compute_step(Problem* problem, FixedStepMultiRateMethod* method, FixedStepMultiRateStep* step, vec* y_0, double t, double H, int M, vec* y_ref, double slow_penalty_factor, ComputeStepReturnValue* ret) {
		vec output_tspan = {t+H};
		problem->reset_eval_counts();
		mat Y = method->solve(t, H, M, y_0, &output_tspan, step);
		vec y_sol = Y.col(0);

		double err = norm(abs(*y_ref - y_sol),2);
		double cost = slow_penalty_factor*problem->slow_function_evals + problem->fast_function_evals;
		double eff = H/cost;

		ret->y = y_sol;
		ret->err = err;
		ret->eff = eff;
		ret->slow_function_evals = problem->slow_function_evals;
		ret->fast_function_evals = problem->fast_function_evals;
		ret->implicit_function_evals = problem->implicit_function_evals;
		ret->explicit_function_evals = problem->explicit_function_evals;
		ret->slow_jacobian_evals = problem->slow_jacobian_evals;
		ret->fast_jacobian_evals = problem->fast_jacobian_evals;
		ret->implicit_jacobian_evals = problem->implicit_jacobian_evals;
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

	void save_output(Problem* problem, const char* method_name, const char* tol_string, const char* spf_string, double slow_penalty_factor, double H_fine, double H_tol, double H_interval, int M_max_iter, int M_min_iter, double eff_rtol) {
		//save_input_parameters(problem, method_name, tol_string, spf_string, slow_penalty_factor, H_fine, H_tol, H_interval, M_max_iter, M_min_iter, eff_rtol);
		save_optimal_output(problem, method_name, tol_string, spf_string);
		//save_near_optimal_output(problem, method_name, tol_string, spf_string);
	}

	void save_input_parameters(Problem* problem, const char* method_name, const char* tol_string, const char* spf_string, double slow_penalty_factor, double H_fine, double H_tol, double H_interval, int M_max_iter, int M_min_iter, double eff_rtol) {
		char data[100];
		sprintf(data, "%s,%f,%f,%f,%f,%d,%d,%f",tol_string,slow_penalty_factor,H_fine,H_tol,H_interval,M_max_iter,M_min_iter,eff_rtol);

		char filename[120];
		sprintf(filename, "./resources/OptimalitySearch/%s/%s_OptimalitySearch_%s_%s_%s_parameters.csv",problem->name,problem->name,method_name,tol_string,spf_string);

		std::ofstream input_parameters_file;
		input_parameters_file.open(filename);
		input_parameters_file << data;
		input_parameters_file.close();
	}

	void save_optimal_output(Problem* problem, const char* method_name, const char* tol_string, const char* spf_string) {
		mat output(optimal_H.size(),10,fill::zeros);
		for(int i=0; i<output.n_rows; i++) {
			output(i,0) = optimal_H[i];
			output(i,1) = optimal_M[i];
			output(i,2) = optimal_eff[i];
			output(i,3) = optimal_slow_function_evals[i];
			output(i,4) = optimal_fast_function_evals[i];
			output(i,5) = optimal_implicit_function_evals[i];
			output(i,6) = optimal_explicit_function_evals[i];
			output(i,7) = optimal_slow_jacobian_evals[i];
			output(i,8) = optimal_fast_jacobian_evals[i];
			output(i,9) = optimal_implicit_jacobian_evals[i];
		}

		char filename[120];
		sprintf(filename, "./resources/OptimalitySearch/%s/%s_OptimalitySearch_%s_%s_%s_optimal.csv",problem->name,problem->name,method_name,tol_string,spf_string);
		output.save(filename, csv_ascii);
	}

	void save_near_optimal_output(Problem* problem, const char* method_name, const char* tol_string, const char* spf_string) {
		char filename[120];

		// Save near optimal H's
		sprintf(filename, "./resources/OptimalitySearch/%s/%s_OptimalitySearch_%s_%s_%s_nearoptimal_H.csv",problem->name,problem->name,method_name,tol_string,spf_string);
	
		std::ofstream near_optimal_H_file;
		near_optimal_H_file.open(filename);

		for(std::vector<double> near_optimal_H_vec : near_optimal_H) {
			for(int i=0; i<near_optimal_H_vec.size(); i++) {
				near_optimal_H_file << near_optimal_H_vec[i];
				if (i != near_optimal_H_vec.size()-1) {
					near_optimal_H_file << ",";
				}
			}
			near_optimal_H_file << "\n";
		}

		near_optimal_H_file.close();

		// Save near optimal M's
		sprintf(filename, "./resources/OptimalitySearch/%s/%s_OptimalitySearch_%s_%s_%s_nearoptimal_M.csv",problem->name,problem->name,method_name,tol_string,spf_string);
	
		std::ofstream near_optimal_M_file;
		near_optimal_M_file.open(filename);

		for(std::vector<int> near_optimal_M_vec : near_optimal_M) {
			for(int i=0; i<near_optimal_M_vec.size(); i++) {
				near_optimal_M_file << near_optimal_M_vec[i];
				if (i != near_optimal_M_vec.size()-1) {
					near_optimal_M_file << ",";
				}
			}
			near_optimal_M_file << "\n";
		}

		near_optimal_M_file.close();

		// Save near optimal eff's
		sprintf(filename, "./resources/OptimalitySearch/%s/%s_OptimalitySearch_%s_%s_%s_nearoptimal_eff.csv",problem->name,problem->name,method_name,tol_string,spf_string);
	
		std::ofstream near_optimal_eff_file;
		near_optimal_eff_file.open(filename);

		for(std::vector<double> near_optimal_eff_vec : near_optimal_eff) {
			for(int i=0; i<near_optimal_eff_vec.size(); i++) {
				near_optimal_eff_file << near_optimal_eff_vec[i];
				if (i != near_optimal_eff_vec.size()-1) {
					near_optimal_eff_file << ",";
				}
			}
			near_optimal_eff_file << "\n";
		}

		near_optimal_eff_file.close();
	}
};

#endif