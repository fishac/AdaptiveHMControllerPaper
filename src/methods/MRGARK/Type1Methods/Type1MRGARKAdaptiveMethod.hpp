#ifndef TYPE1MRGARKADAPTIVEMETHOD_DEFINED__
#define TYPE1MRGARKADAPTIVEMETHOD_DEFINED__

#include "AdaptiveStepMultiRateMethod.hpp"
#include "Type1MRGARKAdaptiveStep.hpp"
#include "MRGARKCoefficients.hpp"
#include "WeightedErrorNorm.hpp"
#include <set>

using namespace arma;

class Type1MRGARKAdaptiveMethod : public AdaptiveStepMultiRateMethod {

public:
	int problem_dimension;
	mat Y;
	vec y_accepted;
	vec y;
	vec y_hat;
	vec y_hat_s;
	vec y_hat_f;
	double H;
	int M;
	double H_new;
	int M_new;
	double primary_order;
	double secondary_order;
	double tol = 1.0;
	double err;
	double err_s;
	double err_f;
	double safety_fac = 0.85;
	int output_index;
	int total_microtimesteps;
	int total_successful_microtimesteps;
	WeightedErrorNorm* err_norm;
	int status;

	Type1MRGARKAdaptiveMethod(int problem_dimension_, WeightedErrorNorm* err_norm_) {
		problem_dimension = problem_dimension_;
		err_norm = err_norm_;
	}

	void solve(double t_0, double H_0, int M_0, vec* y_0, vec* output_tspan, Type1MRGARKAdaptiveStep* type1_mrgark_step, Controller* controller, AdaptiveMultiRateMethodReturnValue* ret) {
		prepare_solve(t_0, H_0, M_0, output_tspan, type1_mrgark_step);
		output_index = 0;
		if (t_0 == (*output_tspan)(0)) {
			Y.col(0) = *y_0;
			output_index++;
		}
		err_norm->set_weights(y_0);
		y_accepted = *y_0;
		double t = t_0;

		status = 0;
		total_microtimesteps = 0;
		total_successful_microtimesteps = 0;
		std::vector<double> ts;
		std::vector<double> Hs;
		std::vector<int> Ms;
		ts.push_back(t_0);
		Hs.push_back(H);
		Ms.push_back(M);
		controller->initialize(H, M);
		total_microtimesteps += M;

		struct MRGARKAdaptiveStepReturnValue step_ret;
		bool continue_computation = true;
		while(continue_computation) {
			if (status == 0) {
				y = y_accepted;
				type1_mrgark_step->step_solution(t, H, M, &y_accepted, &step_ret);
				y = step_ret.y;

				err = std::max(step_ret.err,1e-6);
				err_s = std::max(step_ret.err_s,1e-6);
				err_f = std::max(step_ret.err_f,1e-6);

				controller->update_errors(err,err_f/err_s);
				H_new = controller->get_new_H();
				M_new = controller->get_new_M();

				if(M_new > M+2) {
					//M_new = M+2;
				} else if(M_new < std::max(1,M-1)) {
					//M_new = std::max(1,M-1);
				} 
				M_new =  std::min(std::max(M_new,1),300);

				//Accept step.
				if (err < tol) {
					if (abs(t+H - (*output_tspan)(output_index)) < 1e-15) {
						if (output_index < output_tspan->n_elem-1 && t+H+H_new > (*output_tspan)(output_index+1)) {
							H_new = (*output_tspan)(output_index+1)-t-H;
						}
					} else if (t+H+H_new > (*output_tspan)(output_index)) {
						H_new = (*output_tspan)(output_index)-t-H;
					}

					if(abs(t+H - (*output_tspan)(output_index)) < 1e-15 ) {
						//printf("Storing solution at t: %.16f\n",t+H);
						Y.col(output_index) = y;
						output_index++;

						if (output_index == output_tspan->n_elem) {
							continue_computation = false;
						}
					}

					err_norm->set_weights(&y);
					y_accepted = y;
					t += H;
					total_successful_microtimesteps += M;
				}	

				if (continue_computation && (abs(H_new) < 1e-15 || t+H_new == t)) {
					if (abs(H_new) < 1e-15) {
						//printf("H_new too small at t: %.16f\n", t);
					} else if (t+H_new == t) {
						//printf("t+H_new == t at t: %.16f\n",t);
					}
					continue_computation = false;
					status = 1;
				}
				if (!isfinite(H_new)) {
					printf("Nonfinite H_new. t: %.16f, H: %.16f, H_new: %.16f, M: %d, M_new: %d\n",t,H,H_new,M,M_new);
					printf("err: %.16f, err_s: %.16f, err_f: %.16f\n",err,err_s,err_f);
					printf("y.has_nan: %d, y.has_inf: %d\n",y.has_nan(),y.has_inf());
					printf("y_hat.has_nan: %d, y_hat.has_inf: %d\n",y_hat.has_nan(),y_hat.has_inf());
					printf("y_hat_s.has_nan: %d, y_hat_s.has_inf: %d\n",y_hat_s.has_nan(),y_hat_s.has_inf());
					printf("y_hat_f.has_nan: %d, y_hat_f.has_inf: %d\n",y_hat_f.has_nan(),y_hat_f.has_inf());
					y_accepted.print("y_accepted");
					//H_new = H/2.0;
					//printf("Nonfinite H_new. Stopping solve at t: %.16f.\n",t);
					continue_computation = false;
					status = 2;
				}
				if (!isfinite(M_new)) {
					printf("Nonfinite M_new. t: %.16f, H: %.16f, H_new: %.16f, M: %d, M_new: %d\n",t,H,H_new,M,M_new);
					printf("err: %.16f, err_s: %.16f, err_f: %.16f\n",err,err_s,err_f);
					printf("y.has_nan: %d, y.has_inf: %d\n",y.has_nan(),y.has_inf());
					printf("y_hat.has_nan: %d, y.y_hat: %d\n",y_hat.has_nan(),y_hat.has_inf());
					printf("y_hat_s.has_nan: %d, y_hat_s.has_inf: %d\n",y_hat_s.has_nan(),y_hat_s.has_inf());
					printf("y_hat_f.has_nan: %d, y_hat_f.has_inf: %d\n",y_hat_f.has_nan(),y_hat_f.has_inf());
					continue_computation = false;
				}
			}

			ts.push_back(t);
			Hs.push_back(H);
			Ms.push_back(M);
			total_microtimesteps += M;

			H = H_new;
			M = M_new;

			controller->update_H(H_new);
			controller->update_M(M_new);
		}

		ret->Y = Y;
		ret->ts = ts;
		ret->Hs = Hs;
		ret->Ms = Ms;
		ret->total_timesteps = ts.size();
		ret->total_successful_timesteps = std::set<double>(ts.begin(), ts.end()).size();
		ret->total_microtimesteps = total_microtimesteps;
		ret->total_successful_microtimesteps = total_successful_microtimesteps;
		ret->status = status;
	}

	void prepare_solve(double t_0, double H_, int M_, vec* output_tspan, Type1MRGARKAdaptiveStep* type1_mrgark_step) {
		type1_mrgark_step->set_coeffs();
		type1_mrgark_step->refresh_coupling_coeffs(M_);

		H = H_;
		M = M_;

		Y = mat(problem_dimension, output_tspan->n_elem, fill::zeros);
		y_accepted = vec(problem_dimension, fill::zeros);
		y = vec(problem_dimension, fill::zeros);
		y_hat = vec(problem_dimension, fill::zeros);
		y_hat_s = vec(problem_dimension, fill::zeros);
		y_hat_f = vec(problem_dimension, fill::zeros);
	}
};

#endif