#ifndef MRIGARKINNERRHSFUNCTIONS_DEFINED__
#define MRIGARKINNERRHSFUNCTIONS_DEFINED__

#include "RHS.hpp"
#include "RHSJacobian.hpp"

using namespace arma;

class MRIGARKInnerRHSFunctions {
public:
	RHS* fast_func;
	RHS* slow_func;
	RHSJacobian* fast_func_jac;
	RHSJacobian* slow_func_jac;
	
	vec* c;
	std::vector<mat>* gammas;

	int problem_dimension;
	int num_gammas;
	mat* y_stages;
	double H;
	double t;
	int stage_index;
	int embedding_shift;

	vec f_temp;
	vec f_temp2;
	vec implicit_previous_terms;
	mat explicit_previous_terms;
	vec y_temp;
	mat jac_temp;
	mat jac_temp2;
	mat I;

	MRIGARKInnerRHSFunctions(MRIGARKCoefficients* coeffs, RHS* fast_func_, RHS* slow_func_, RHSJacobian* fast_func_jac_, RHSJacobian* slow_func_jac_, int problem_dimension_) {
		c = &(coeffs->c);
		gammas = &(coeffs->gammas);
		fast_func = fast_func_;
		slow_func = slow_func_;
		fast_func_jac = fast_func_jac_;
		slow_func_jac = slow_func_jac_;
		problem_dimension = problem_dimension_;
		num_gammas = coeffs->num_gammas;

		f_temp = vec(problem_dimension, fill::zeros);
		f_temp2 = vec(problem_dimension, fill::zeros);
		explicit_previous_terms = mat(problem_dimension, coeffs->num_stages, fill::zeros);
		implicit_previous_terms = vec(problem_dimension, fill::zeros);
		y_temp = vec(problem_dimension, fill::zeros);
		jac_temp = mat(problem_dimension, problem_dimension, fill::zeros);
		jac_temp2 = mat(problem_dimension, problem_dimension, fill::zeros);
		I = eye(problem_dimension, problem_dimension);
	}

	void explicit_solve_rhs(double theta, vec* v, vec* f) {		
		y_temp.zeros();
		f_temp.zeros();
		f->zeros();

		double delta_c = (*c)(stage_index) - (*c)(stage_index-1); 
		double T_prev = t + (*c)(stage_index-1)*H;
		fast_func->evaluate(T_prev + delta_c*theta, v, &f_temp);
		(*f) = delta_c*f_temp;

		for(int sub_stage_index=0; sub_stage_index<stage_index; sub_stage_index++) {
			f_temp = explicit_previous_terms.col(sub_stage_index);
			f_temp *= gamma(stage_index + embedding_shift, sub_stage_index, theta/H);
			(*f) += f_temp;
		}
	}

	void explicit_solve_rhsjacobian(double theta, vec* v, mat* jac) {		
		y_temp.zeros();
		jac_temp.zeros();
		jac->zeros();

		double delta_c = (*c)(stage_index) - (*c)(stage_index-1); 
		double T_prev = t + (*c)(stage_index-1)*H;
		fast_func_jac->evaluate(T_prev + delta_c*theta, v, &jac_temp);
		*jac = delta_c*jac_temp;
	}

	void explicit_set_previous_terms() {
		y_temp.zeros();
		explicit_previous_terms.zeros();

		double tj;

		for(int sub_stage_index=0; sub_stage_index<stage_index; sub_stage_index++) {
			// Set y_temp = Y_j = Y_{sub_stage_index}
			y_temp = y_stages->col(sub_stage_index);
			// Set tj = t_n = t+c_j*H
			tj = t + (*c)(sub_stage_index)*H;

			// Set f_temp = gamma_bar_{i,j}*f^I(t+c_j*H,Y_j)
			slow_func->evaluate(tj, &y_temp, &f_temp);
			// Add gamma_bar_{i,j}*f^S(t+c_j*H,Y_j) to the result vector
			explicit_previous_terms.col(sub_stage_index) = f_temp;
		}
	}

	void implicit_solve_residual(vec* explicit_data, vec* y_prev, vec* y, vec* f) {
		f_temp.zeros();
		f->zeros();

		implicit_solve_current_term(y, &f_temp);

		// Compute result vector
		*f = *y - *y_prev - H*(*explicit_data + f_temp);
	}

	void implicit_solve_current_term(vec* y, vec* f) {
		// Clear existing data in f, f_temp, y_temp
		f_temp2.zeros();
		f->zeros();

		double ti = t + (*c)(stage_index)*H;

		// Set f_temp = gamma_bar_{i,i}*f^I(t_n+c_i*H, y)
		slow_func->evaluate(ti, y, &f_temp2);
		f_temp2 *= gamma_bar(stage_index + embedding_shift, stage_index);

		(*f) = f_temp2;
	}

	void implicit_set_previous_terms() {
		f_temp.zeros();
		y_temp.zeros();
		implicit_previous_terms.zeros();

		double tj;

		for(int sub_stage_index=0; sub_stage_index<stage_index; sub_stage_index++) {
			// Set y_temp = Y_j = Y_{sub_stage_index}
			y_temp = y_stages->col(sub_stage_index);
			// Set tj = t_n = t+c_j*H
			tj = t + (*c)(sub_stage_index)*H;

			// Set f_temp = gamma_bar_{i,j}*f^I(t+c_j*H,Y_j)
			slow_func->evaluate(tj, &y_temp, &f_temp);
			f_temp *= gamma_bar(stage_index + embedding_shift, sub_stage_index);
			// Add gamma_bar_{i,j}*f^I(t+c_j*H,Y_j) to the result vector
			implicit_previous_terms += f_temp;
		}
	}

	vec implicit_get_previous_terms() {
		return implicit_previous_terms;
	}

	void implicit_solve_residual_jacobian(vec* y, mat* jac) {
		jac->zeros();
		jac_temp.zeros();

		// Set jac_temp = gamma_bar_{i,i}*f^I_y(t+c_i*H,y)
		double gbar = gamma_bar(stage_index + embedding_shift, stage_index);
		if (gbar <= 2.0*1e-8) {
			(*jac) = I;
		} else {
			implicit_solve_jacobian(y, &jac_temp);
			(*jac) = I - H*gbar*jac_temp;
		}
	}

	void implicit_solve_jacobian(vec* y, mat* jac) {
		jac->zeros();
		jac_temp2.zeros();

		// Set ti = t + c_i*H
		double ti = t + (*c)(stage_index)*H;

		// Set jac_temp = gamma_bar_{i,i}*f^I_y(t+c_i*H,y)
		slow_func_jac->evaluate(ti, y, &jac_temp2);
		(*jac) = jac_temp2;
	}
	
	double gamma(int i, int j, double tau) {
		double gamma_val = 0.0;
		for(int k=0; k<num_gammas; k++) {
			gamma_val += (gammas->at(k))(i,j)*pow(tau,k);
		}
		return gamma_val;
	}

	double gamma_bar(int i, int j) {
		double gamma_bar_val = 0.0;
		for(int k=0; k<num_gammas; k++) {
			gamma_bar_val += (gammas->at(k))(i,j)/(k+1);
		}
		return gamma_bar_val;
	}

	void set_problem_dependent_data(mat* y_stages_) {
		y_stages = y_stages_;
	}

	void set_function_dependent_data(double H_, double t_, int stage_index_, bool embedding_) {
		H = H_;
		t = t_;
		stage_index = stage_index_;
		if (embedding_) {
			embedding_shift = 1;
		} else {
			embedding_shift = 0;
		}
	}
};

#endif