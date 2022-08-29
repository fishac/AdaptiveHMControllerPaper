#ifndef TYPE1MRGARKADAPTIVESTEP_DEFINED__
#define TYPE1MRGARKADAPTIVESTEP_DEFINED__

#include "Problem.hpp"
#include "MRGARKCoefficients.hpp"
#include "WeightedErrorNorm.hpp"

using namespace arma;

struct MRGARKAdaptiveStepReturnValue {
	vec y;
	double err;
	double err_s;
	double err_f;
	int status = 0;
};

class Type1MRGARKAdaptiveStep {

public:
	Problem* problem;
	MRGARKCoefficients* coeffs; 
	WeightedErrorNorm* err_norm;
	vec y_temp;
	vec y_temp2;
	vec y_temp3;
	vec y;
	vec y_hat;
	vec y_hat_s;
	vec y_hat_f;
	vec f_temp;
	mat y_stages_s;
	std::vector<mat> y_stages_f;
	std::vector<mat> A_sfl_mats;
	std::vector<mat> A_fsl_mats;
	mat A_ss;
	mat A_ff;
	mat A_coupling_temp;
	vec b_f;
	vec b_s;
	vec d_f;
	vec d_s;
	vec c_ss;
	vec c_ff;
	std::vector<vec> c_fs;
	std::vector<vec> c_sf;
	int problem_dimension;
	int num_stages = 0;

	Type1MRGARKAdaptiveStep(MRGARKCoefficients* coeffs_, Problem* problem_, int problem_dimension_, WeightedErrorNorm* err_norm_) {
		coeffs = coeffs_;
		problem = problem_;
		problem_dimension = problem_dimension_;
		num_stages = coeffs->num_stages;
		err_norm = err_norm_;
		declare_vectors();
	}

	void step_solution(double t, double H, int M, vec* y_prev, MRGARKAdaptiveStepReturnValue* ret) {
		prepare_step(M);

		// For lambda=1, alternate computing fast and slow stages.
		for(int stage_idx=0; stage_idx<num_stages; stage_idx++) {
			// Compute fast stage.
			y_stages_f[0].col(stage_idx) = *y_prev;

			// Compute slow stage contribution.
			y_temp.zeros();
			for(int slow_stage_idx=0; slow_stage_idx<stage_idx; slow_stage_idx++) {
				y_temp2 = y_stages_s.col(slow_stage_idx);
				problem->slow_rhs(t+c_fs[0](slow_stage_idx)*H, &y_temp2, &f_temp);
				y_temp += A_fsl_mats[0](stage_idx,slow_stage_idx)*f_temp;

			}
			y_stages_f[0].col(stage_idx) += H*y_temp;

			// Compute fast stage contribution.
			y_temp.zeros();
			for(int fast_stage_idx=0; fast_stage_idx<stage_idx; fast_stage_idx++) {
				y_temp2 = y_stages_f[0].col(fast_stage_idx);
				problem->fast_rhs(t+c_ff(fast_stage_idx)*H/M, &y_temp2, &f_temp);
				y_temp += A_ff(stage_idx,fast_stage_idx)*f_temp;
			}
			y_stages_f[0].col(stage_idx) += H/M*y_temp;

			// Compute slow stage.
			y_stages_s.col(stage_idx) = *y_prev;

			// Compute slow stage contribution.
			y_temp.zeros();
			for(int slow_stage_idx=0; slow_stage_idx<stage_idx; slow_stage_idx++) {
				y_temp2 = y_stages_s.col(slow_stage_idx);
				problem->slow_rhs(t+c_ss(slow_stage_idx)*H, &y_temp2, &f_temp);
				y_temp += A_ss(stage_idx,slow_stage_idx)*f_temp;
			}
			y_stages_s.col(stage_idx) += H*y_temp;

			// Compute fast stage contribution.
			y_temp.zeros();
			for(int fast_stage_idx=0; fast_stage_idx<=stage_idx; fast_stage_idx++) {
				y_temp2 = y_stages_f[0].col(fast_stage_idx);
				problem->fast_rhs(t+c_sf[0](fast_stage_idx)*H/M, &y_temp2, &f_temp);
				y_temp += A_sfl_mats[0](stage_idx,fast_stage_idx)*f_temp;
			}
			y_stages_s.col(stage_idx) += H/M*y_temp;
		}

		// For lambda=2,...,M, compute the rest of the fast stages.
		for(int lambda=1; lambda<M; lambda++) {
			for(int stage_idx=0; stage_idx<num_stages; stage_idx++) {
				// Compute fast stage.
				y_stages_f[lambda].col(stage_idx) = *y_prev;

				// Compute slow stage contribution.
				y_temp.zeros();
				for(int slow_stage_idx=0; slow_stage_idx<num_stages; slow_stage_idx++) {
					y_temp2 = y_stages_s.col(slow_stage_idx);
					problem->slow_rhs(t+c_fs[lambda](slow_stage_idx)*H, &y_temp2, &f_temp);
					y_temp += A_fsl_mats[lambda](stage_idx,slow_stage_idx)*f_temp;
				}
				y_stages_f[lambda].col(stage_idx) += H*y_temp;

				// Compute fast stage contribution.
				y_temp.zeros();
				for(int fast_stage_idx=0; fast_stage_idx<stage_idx; fast_stage_idx++) {
					y_temp2 = y_stages_f[lambda].col(fast_stage_idx);
					problem->fast_rhs(t+c_ff(fast_stage_idx)*H/M, &y_temp2, &f_temp);
					y_temp += A_ff(stage_idx,fast_stage_idx)*f_temp;
				}
				y_stages_f[lambda].col(stage_idx) += H/M*y_temp;

				// Compute previous micro time step contribution.
				y_temp.zeros();
				for(int lambda_idx=0; lambda_idx<lambda; lambda_idx++) {
					for(int lambda_stage_idx=0; lambda_stage_idx<num_stages; lambda_stage_idx++) {
						y_temp2 = y_stages_f[lambda_idx].col(lambda_stage_idx);
						problem->fast_rhs(t+c_ff(lambda_stage_idx)*H/M+lambda_idx*H/M, &y_temp2, &f_temp);
						y_temp += b_f(lambda_stage_idx)*f_temp;
					}
				}
				y_stages_f[lambda].col(stage_idx) += H/M*y_temp;
			}
		}

		y = *y_prev;
		y_hat = *y_prev;
		y_hat_s = *y_prev;
		y_hat_f = *y_prev;

		// Compute slow stage contribution.
		y_temp.zeros();
		y_temp2.zeros();
		y_temp3.zeros();
		f_temp.zeros();
		for(int slow_stage_idx=0; slow_stage_idx<num_stages; slow_stage_idx++) {
			y_temp = y_stages_s.col(slow_stage_idx);
			problem->slow_rhs(t+c_ss(slow_stage_idx)*H, &y_temp, &f_temp);
			y_temp2 += b_s(slow_stage_idx)*f_temp;
			y_temp3 += d_s(slow_stage_idx)*f_temp;
		}
		y += H*y_temp2;
		y_hat += H*y_temp3;
		y_hat_s += H*y_temp3;
		y_hat_f += H*y_temp2;

		// Compute fast stage contribution.
		y_temp.zeros();
		y_temp2.zeros();
		y_temp3.zeros();
		f_temp.zeros();
		for(int lambda=0; lambda<M; lambda++) {
			for(int fast_stage_idx=0; fast_stage_idx<num_stages; fast_stage_idx++) {
				y_temp = y_stages_f[lambda].col(fast_stage_idx);
				problem->fast_rhs(t+c_ff(fast_stage_idx)*H/M+lambda*H/M, &y_temp, &f_temp);
				y_temp2 += b_f(fast_stage_idx)*f_temp;
				y_temp3 += d_f(fast_stage_idx)*f_temp;
			}
		}
		y += H/M*y_temp2;
		y_hat += H/M*y_temp3;
		y_hat_s += H/M*y_temp2;
		y_hat_f += H/M*y_temp3;

		ret->y = y;
		ret->err = err_norm->compute_norm(y-y_hat);
		ret->err_s = err_norm->compute_norm(y-y_hat_s);
		ret->err_f = err_norm->compute_norm(y-y_hat_f);
	}

	void prepare_step(int M) {
		y_temp.zeros();
		y_temp2.zeros();
		y_temp3.zeros();
		f_temp.zeros();
		y.zeros();
		y_hat.zeros();
		y_hat_s.zeros();
		y_hat_f.zeros();
		y_stages_s.zeros();
		
		if (y_stages_f.size() == M) {
			for(mat y_stage_f : y_stages_f) {
				y_stage_f.zeros();
			}
		} else {
			y_stages_f.clear();
			for(int i=0; i<M; i++) {
				y_stages_f.push_back(mat(problem_dimension,coeffs->num_stages,fill::zeros));
			}
			refresh_coupling_coeffs(M);
		}
	}

	void refresh_coupling_coeffs(int M) {
		A_sfl_mats.clear();
		A_fsl_mats.clear();
		c_fs.clear();
		c_sf.clear();
		for(int lambda=1; lambda<=M; lambda++) {
			A_coupling_temp = coeffs->get_A_fsl(lambda,M);
			A_fsl_mats.push_back(mat(A_coupling_temp));
			c_fs.push_back(coeffs->get_c_fs(&A_coupling_temp));

			A_coupling_temp = coeffs->get_A_sfl(lambda,M);
			c_sf.push_back(coeffs->get_c_sf(&A_coupling_temp));
			A_sfl_mats.push_back(mat(A_coupling_temp));
		}
	}

	void set_coeffs() {
		A_ss = coeffs->get_A_ss();
		A_ff = coeffs->get_A_ff();
		b_f = coeffs->get_b_f();
		b_s = coeffs->get_b_s();
		d_f = coeffs->get_d_f();
		d_s = coeffs->get_d_s();
		c_ss = coeffs->get_c_ss();
		c_ff = coeffs->get_c_ff();
	}

	void declare_vectors() {
		y_temp = vec(problem_dimension, fill::zeros);
		y_temp2 = vec(problem_dimension, fill::zeros);
		y_temp3 = vec(problem_dimension, fill::zeros);
		f_temp = vec(problem_dimension, fill::zeros);
		y = vec(problem_dimension, fill::zeros);
		y_hat = vec(problem_dimension, fill::zeros);
		y_hat_s = vec(problem_dimension, fill::zeros);
		y_hat_f = vec(problem_dimension, fill::zeros);
		y_stages_s = mat(problem_dimension,coeffs->num_stages,fill::zeros);
	}
};

#endif