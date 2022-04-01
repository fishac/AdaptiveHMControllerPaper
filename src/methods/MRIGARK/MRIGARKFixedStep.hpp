#ifndef MRIGARKFIXEDSTEP_DEFINED__
#define MRIGARKFIXEDSTEP_DEFINED__

#include "MRIGARKCoefficients.hpp"
#include "MRIGARKInnerRHSFunctions.hpp"
#include "MRIGARKExplicitRHS.hpp"
#include "MRIGARKExplicitRHSJacobian.hpp"
#include "MRIGARKImplicitResidual.hpp"
#include "MRIGARKImplicitResidualJacobian.hpp"
#include "SingleRateMethodCoefficients.hpp"
#include "FixedDIRKMethod.hpp"
#include "NewtonSolver.hpp"
#include "WeightedErrorNorm.hpp"
#include "RHS.hpp"
#include "RHSJacobian.hpp"
#include "FixedStepMultiRateStep.hpp"

using namespace arma;

class MRIGARKFixedStep : public FixedStepMultiRateStep {
public:
	MRIGARKCoefficients* coeffs;
	MRIGARKInnerRHSFunctions inner_rhs_funcs;
	MRIGARKExplicitRHS explicit_rhs;
	MRIGARKExplicitRHSJacobian explicit_rhs_jacobian;
	MRIGARKImplicitResidual implicit_residual;
	MRIGARKImplicitResidualJacobian implicit_residual_jacobian;
	NewtonSolver newton_solver;
	struct NewtonSolverReturnValue newton_ret;
	FixedDIRKMethod dirk;
	mat y_stages;
	vec v_0;
	vec v_H;
	double H;
	int M;
	int problem_dimension;
	int num_stages;
	bool useEmbedding;

	MRIGARKFixedStep(MRIGARKCoefficients* coeffs_, SingleRateMethodCoefficients* inner_coeffs_, RHS* fast_func_, RHS* slow_func_, RHSJacobian* fast_func_jac_, RHSJacobian* slow_func_jac_, int problem_dimension_, WeightedErrorNorm* err_norm) :
	inner_rhs_funcs(coeffs_, fast_func_, slow_func_, fast_func_jac_, slow_func_jac_, problem_dimension_),
	implicit_residual(&(MRIGARKFixedStep::inner_rhs_funcs)),
	implicit_residual_jacobian(&(MRIGARKFixedStep::inner_rhs_funcs)),
	explicit_rhs(&(MRIGARKFixedStep::inner_rhs_funcs)),
	explicit_rhs_jacobian(&(MRIGARKFixedStep::inner_rhs_funcs)),
	dirk(inner_coeffs_, &(MRIGARKFixedStep::explicit_rhs), &(MRIGARKFixedStep::explicit_rhs_jacobian), problem_dimension_, err_norm),
	newton_solver(&(MRIGARKFixedStep::implicit_residual), &(MRIGARKFixedStep::implicit_residual_jacobian), 20, 1.0, problem_dimension_, err_norm)
	{
		coeffs = coeffs_;
		problem_dimension = problem_dimension_;
		num_stages = coeffs_->num_stages;
		useEmbedding = false;
		
		declare_vectors();
	}
	
	MRIGARKFixedStep(MRIGARKCoefficients* coeffs_, SingleRateMethodCoefficients* inner_coeffs_, RHS* fast_func_, RHS* slow_func_, RHSJacobian* fast_func_jac_, RHSJacobian* slow_func_jac_, int problem_dimension_, WeightedErrorNorm* err_norm, bool useEmbedding_) :
	inner_rhs_funcs(coeffs_, fast_func_, slow_func_, fast_func_jac_, slow_func_jac_, problem_dimension_),
	implicit_residual(&(MRIGARKFixedStep::inner_rhs_funcs)),
	implicit_residual_jacobian(&(MRIGARKFixedStep::inner_rhs_funcs)),
	explicit_rhs(&(MRIGARKFixedStep::inner_rhs_funcs)),
	explicit_rhs_jacobian(&(MRIGARKFixedStep::inner_rhs_funcs)),
	dirk(inner_coeffs_, &(MRIGARKFixedStep::explicit_rhs), &(MRIGARKFixedStep::explicit_rhs_jacobian), problem_dimension_, err_norm),
	newton_solver(&(MRIGARKFixedStep::implicit_residual), &(MRIGARKFixedStep::implicit_residual_jacobian), 20, 1.0, problem_dimension_, err_norm)
	{
		coeffs = coeffs_;
		problem_dimension = problem_dimension_;
		num_stages = coeffs_->num_stages;
		useEmbedding = useEmbedding_;
		
		declare_vectors();
	}

	void step_solution(double t, double H, int M, vec* y_prev, vec* solution_vec) {
		y_stages.col(0) = *y_prev;
		for(int stage_index=1; stage_index<num_stages; stage_index++) {
			inner_rhs_funcs.set_function_dependent_data(H, t, stage_index, false);
			if (stage_index == num_stages-1 && useEmbedding) {
				inner_rhs_funcs.set_function_dependent_data(H, t, stage_index, true);
			}
			v_0 = y_stages.col(stage_index-1);

			double gbar = inner_rhs_funcs.gamma_bar(stage_index,stage_index);
			if(gbar != 0.0) {
				implicit_step(t);
			} else {
				explicit_step(H,M);
			}
			y_stages.col(stage_index) = v_H;
		}
		*solution_vec = y_stages.col(num_stages-1);
	}

	void implicit_step(double t) {
		newton_solver.solve(t, &v_0, &newton_ret);
		v_H = newton_ret.y;
	}

	void explicit_step(double H, int M) {
		inner_rhs_funcs.explicit_set_previous_terms();
		vec output_tspan = {H};
		mat Y = dirk.solve(0.0, H/M, &v_0, &output_tspan);
		v_H = Y.col(0);
	}

	void declare_vectors() {
		v_0 = vec(problem_dimension, fill::zeros);
		v_H = vec(problem_dimension, fill::zeros);
		y_stages = mat(problem_dimension, num_stages, fill::zeros);
		inner_rhs_funcs.set_problem_dependent_data(&y_stages);
	}
};

#endif