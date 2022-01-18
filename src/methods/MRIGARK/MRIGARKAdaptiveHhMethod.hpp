#ifndef MRIGARKADAPTIVEHHMETHOD_DEFINED__
#define MRIGARKADAPTIVEHHMETHOD_DEFINED__

#include "MRIGARKAdaptiveStep.hpp"
#include "AdaptiveStepMultiRateMethod.hpp"
#include "MRIGARKAdaptiveStep.hpp"
#include "MRIGARKCoefficients.hpp"
#include "SingleRateMethodCoefficients.hpp"
#include "MRIGARKFixedStep.hpp"
#include "RHS.hpp"
#include "RHSJacobian.hpp"
#include "WeightedErrorNorm.hpp"
#include "HhController.hpp"
#include <set>

using namespace arma;

class MRIGARKAdaptiveHhMethod : AdaptiveStepMultiRateMethod {
public:
	mat Y;
	vec y_accepted;
	vec y;
	vec y_hat;
	double H;
	double H_new;
	int M_new;
	double h;
	double ess;
	double esf;
	double tol = 1.0;
	double H_recovery_factor = 0.1;
	int M;
	int output_index;
	int problem_dimension;
	WeightedErrorNorm* err_norm;
	int status;

	MRIGARKAdaptiveHhMethod(int problem_dimension_, WeightedErrorNorm* err_norm_) {
		problem_dimension = problem_dimension_;
		err_norm = err_norm_;
	}

	void solve(double t_0, double H_0, int M_0, vec* y_0, vec* output_tspan, MRIGARKAdaptiveStep* mrigark_step, HhController* controller, int esf_measurement_type, AdaptiveMultiRateMethodReturnValue* ret) {
		prepare_solve(t_0, H_0, M_0, output_tspan);
		output_index = 0;
		if (t_0 == (*output_tspan)(0)) {
			Y.col(0) = *y_0;
			output_index++;
		}
		y_accepted = *y_0;
		err_norm->set_weights(y_0);

		status = 0;
		int total_timesteps = 1;
		int total_microtimesteps = 0;
		int total_successful_microtimesteps = 0;
		int total_true_positives = 0;
		int total_true_negatives = 0;
		int total_false_positives = 0;
		int total_false_negatives = 0;
		std::vector<double> ts;
		std::vector<double> Hs;
		std::vector<int> effective_Ms;
		ts.push_back(t_0);
		Hs.push_back(H);
		controller->initialize(H, H/M);

		double t = t_0;
		bool continue_computation = true;
		struct MRIGARKAdaptiveStepReturnValue step_ret;
		while(continue_computation) {
			y = y_accepted;
			mrigark_step->step_solution(t, H, M, &y, esf_measurement_type, &step_ret);
				
			if (step_ret.status == 0 && status == 0) {
				y = step_ret.y;
				ess = std::max(step_ret.ess,1e-6);
				esf = std::max(step_ret.esf,1e-6);
				total_microtimesteps += step_ret.total_microtimesteps;
				total_successful_microtimesteps += step_ret.total_successful_microtimesteps;

				//printf("\ness: %.16f, esf: %.16f\n",ess,esf);
				//printf("solver H: %.16f\n",H);

				controller->update_errors(ess,esf);
				H_new = controller->get_new_H();
				h = controller->get_new_h();
				h = std::min(h,H);
				M_new = std::ceil(H/h);
				M_new = std::max(1,std::min(M_new,500));


				//Accept step.
				if (ess+esf < tol) {
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
				} else if (!isfinite(ess+esf)) {
					H_new = H_recovery_factor*H;
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
					//printf("Nonfinite H_new. Stopping solve at t: %.16f.\n",t);
					continue_computation = false;
					status = 2;
				}
			} else if (step_ret.status == 1) {
				H_new = H/2.0;
				//continue_computation = false;
				//status = 3;
			} else if (step_ret.status == 2) {
				continue_computation = false;
				status = 4;
			} else if (step_ret.status == 3) {
				H_new = H_recovery_factor*H;
			} else if (step_ret.status == 4) {
				continue_computation = false;
				status = 5;
			}

			if (continue_computation) {
				//printf("t: %.16f, H: %.16f, t+H: %.16f, H_new: %.16f, h: %.16f, M_new: %d\n\n",t,H,t+H,H_new,h,M_new);

				total_timesteps++;
				ts.push_back(t);
				Hs.push_back(H);
				//effective_Ms.push_back(step_ret.total_microtimesteps);
				effective_Ms.push_back(M);
				H = H_new;
				M = M_new;

				controller->update_H(H_new);
				controller->update_h(H_new/M_new);
			}
		}

		ret->Y = Y;
		ret->ts = ts;
		ret->Hs = Hs;
		ret->Ms = effective_Ms;
		ret->total_timesteps = total_timesteps;
		ret->total_successful_timesteps = std::set<double>(ts.begin(), ts.end()).size();
		ret->total_microtimesteps = total_microtimesteps;
		ret->total_successful_microtimesteps = total_successful_microtimesteps;
		ret->status = status;
	}

	void declare_vectors() {
		y_accepted = vec(problem_dimension, fill::zeros);
		y = vec(problem_dimension, fill::zeros);
		y_hat = vec(problem_dimension, fill::zeros);
	}

	void prepare_solve(double t_0, double H_0, int M_0, vec* output_tspan) {
		H = H_0;
		M = M_0;
		y_accepted.zeros();
		y.zeros();
		y_hat.zeros();
		Y = mat(problem_dimension, output_tspan->n_elem, fill::zeros);
	}
};

#endif