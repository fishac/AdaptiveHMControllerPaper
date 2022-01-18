#ifndef MRIGARKADAPTIVESTEPFASTMEASUREMENT_DEFINED__
#define MRIGARKADAPTIVESTEPFASTMEASUREMENT_DEFINED__

#include "MRIGARKAdaptiveStep.hpp"
#include "MRIGARKCoefficients.hpp"
#include "MRIGARKInnerRHSFunctions.hpp"
#include "MRIGARKExplicitRHS.hpp"
#include "MRIGARKExplicitRHSJacobian.hpp"
#include "MRIGARKImplicitResidual.hpp"
#include "MRIGARKImplicitResidualJacobian.hpp"
#include "AdaptiveDIRKMethod.hpp"
#include "SingleRateMethodCoefficients.hpp"
#include "FixedDIRKMethodFastMeasurement.hpp"
#include "NewtonSolver.hpp"
#include "RHS.hpp"
#include "RHSJacobian.hpp"
#include "WeightedErrorNorm.hpp"
#include "FastErrorMeasurementTypes.hpp"
#include <set>

using namespace arma;

class MRIGARKAdaptiveStepFastMeasurement : public MRIGARKAdaptiveStep {
public:
	MRIGARKCoefficients* coeffs;
	MRIGARKInnerRHSFunctions inner_rhs_funcs;
	MRIGARKExplicitRHS explicit_rhs;
	MRIGARKExplicitRHSJacobian explicit_rhs_jacobian;
	MRIGARKImplicitResidual implicit_residual;
	MRIGARKImplicitResidualJacobian implicit_residual_jacobian;
	NewtonSolver newton_solver;
	struct NewtonSolverReturnValue newton_ret;
	FixedDIRKMethodFastMeasurement dirk;
	struct AdaptiveSingleRateMethodReturnValue dirk_ret;
	WeightedErrorNorm* err_norm;
	vec output_tspan;
	mat y_stages;
	vec y;
	vec y_hat;
	vec v_0;
	vec v_H;
	vec errs;
	vec err_vec;
	double H;
	int M;
	int total_microtimesteps;
	int total_successful_microtimesteps;
	int problem_dimension;
	int num_stages;
	int status;
	int inner_status;

	MRIGARKAdaptiveStepFastMeasurement(MRIGARKCoefficients* coeffs_, SingleRateMethodCoefficients* inner_coeffs_, RHS* fast_func_, RHS* slow_func_, RHSJacobian* fast_func_jac_, RHSJacobian* slow_func_jac_, int problem_dimension_, WeightedErrorNorm* err_norm_) :
	inner_rhs_funcs(coeffs_, fast_func_, slow_func_, fast_func_jac_, slow_func_jac_, problem_dimension_),
	implicit_residual(&(MRIGARKAdaptiveStepFastMeasurement::inner_rhs_funcs)),
	implicit_residual_jacobian(&(MRIGARKAdaptiveStepFastMeasurement::inner_rhs_funcs)),
	explicit_rhs(&(MRIGARKAdaptiveStepFastMeasurement::inner_rhs_funcs)),
	explicit_rhs_jacobian(&(MRIGARKAdaptiveStepFastMeasurement::inner_rhs_funcs)),
	dirk(inner_coeffs_, &(MRIGARKAdaptiveStepFastMeasurement::explicit_rhs), &(MRIGARKAdaptiveStepFastMeasurement::explicit_rhs_jacobian), problem_dimension_, err_norm_),
	newton_solver(&(MRIGARKAdaptiveStepFastMeasurement::implicit_residual), &(MRIGARKAdaptiveStepFastMeasurement::implicit_residual_jacobian), 20, 0.1, problem_dimension_, err_norm_)
	{
		coeffs = coeffs_;
		num_stages = coeffs->num_stages;
		problem_dimension = problem_dimension_;
		err_norm = err_norm_;
		declare_vectors();
	}

	// Adaptive step solution
	void step_solution(double t, double H, int M, vec* y_0, const char* measurement_type, MRIGARKAdaptiveStepReturnValue* ret) {
		prepare_solve(H);
		//printf("step_solution. t: %.16f, H: %.16f, M: %d\n",t,H,M);
		total_microtimesteps = 0;
		total_successful_microtimesteps = 0;
		status = 0;

		y_stages.col(0) = *y_0;
		for(int stage_index=1; stage_index<num_stages; stage_index++) {
			if (status == 0) {
				inner_rhs_funcs.set_function_dependent_data(H, t, stage_index, false);
				double gbar = inner_rhs_funcs.gamma_bar(stage_index,stage_index);
				if(gbar != 0.0) {
					v_0 = y_stages.col(stage_index-1);
					implicit_step(t);
					y_stages.col(stage_index) = v_H;
				} else {
					v_0 = y_stages.col(stage_index-1);
					explicit_step(H, M, stage_index);
					y_stages.col(stage_index) = v_H;
				}
				
			}
		}
		y = y_stages.col(num_stages-1);

		int stage_index = num_stages-1;
		if (status == 0) {
			inner_rhs_funcs.set_function_dependent_data(H, t, stage_index, true);
			v_0 = y_stages.col(stage_index-1);
			double gbar = inner_rhs_funcs.gamma_bar(stage_index+1,stage_index);
			if(gbar != 0.0) {
				implicit_step(t);
			} else {
				explicit_step(H, M, stage_index);
			}
			y_hat = v_H;
		}

		ret->y = y;
		ret->ess = err_norm->compute_norm(y-y_hat);
		ret->esf = compute_esf(measurement_type);
		ret->total_microtimesteps = total_microtimesteps;
		ret->total_successful_microtimesteps = total_successful_microtimesteps;
		ret->status = status;

		//printf("step slow measurement ess: %.16f, esf: %.16f, t: %.16f, H: %.16f, M: %d\n",ret->ess,ret->esf,t,H,M);
	}

	void implicit_step(double t) {
		newton_solver.solve(t, &v_0, &newton_ret);
		v_H = newton_ret.y;
		inner_status = newton_ret.status;
		total_microtimesteps += 1;
		if (inner_status == 0) {
			total_successful_microtimesteps += 1;
		} else if (inner_status == 1) {
			// Newton nonconvergence.
			status = 3;
		} else if (inner_status == 2) {
			// Newton linear solver failure.
			status = 4;
		}
	}

	void explicit_step(double H, int M, int stage_index) {
		inner_rhs_funcs.explicit_set_previous_terms();
		struct FixedDIRKMethodFastMeasurementReturnValue ret;
		dirk.solve(0, (double) H/M, &v_0, &output_tspan, &ret);
		v_H = ret.y;
		errs(stage_index) = ret.err_estimate;
		inner_status = dirk_ret.status;
		total_microtimesteps += M;
		total_successful_microtimesteps += M;
	}

	double compute_esf(const char* measurement_type) {
		if (FastError::is_LASAsum(measurement_type)) {
			// Aggregator: sum
			return accu(errs);
		} else if (FastError::is_LASAmean(measurement_type)) {
			// Aggregator: average
			return 1.0/num_stages*accu(errs);
		} else if (FastError::is_LASAmax(measurement_type)) {
			// Aggregator: max
			return max(errs);
		} else {
			return 0.0;
		}
	}

	void declare_vectors() {
		output_tspan = vec(1,fill::zeros);
		y = vec(problem_dimension, fill::zeros);
		y_hat = vec(problem_dimension, fill::zeros);
		v_0 = vec(problem_dimension, fill::zeros);
		v_H = vec(problem_dimension, fill::zeros);
		errs = vec(num_stages, fill::zeros);
		y_stages = mat(problem_dimension, num_stages, fill::zeros);
		inner_rhs_funcs.set_problem_dependent_data(&y_stages);
	}

	void prepare_solve(double H) {
		output_tspan(0) = H;
		v_0.zeros();
		v_H.zeros();
		y_stages.zeros();
	}
};

#endif