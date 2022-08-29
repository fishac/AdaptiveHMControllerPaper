#ifndef MRIGARKFIXEDSTEP_DEFINED__
#define MRIGARKFIXEDSTEP_DEFINED__

#include "MRICoefficients.hpp"
#include "MRIGARKInnerRHSFunctions.hpp"
#include "MRIGARKExplicitSolveProblem.hpp"
#include "MRIGARKImplicitSolveResidual.hpp"
#include "SingleRateMethodCoefficients.hpp"
#include "FixedDIRKMethod.hpp"
#include "NewtonSolver.hpp"
#include "WeightedErrorNorm.hpp"
#include "FixedStepMultiRateStep.hpp"
#include "Problem.hpp"

using namespace arma;

class MRIGARKFixedStep : public FixedStepMultiRateStep {
public:
	MRICoefficients* coeffs;
	MRIGARKInnerRHSFunctions inner_rhs_funcs;
	MRIGARKExplicitSolveProblem explicit_solve_problem;
	MRIGARKImplicitSolveResidual implicit_solve_residual;
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
	int num_groups;
	int stage_index;
	bool use_embedding;

	MRIGARKFixedStep(MRICoefficients* coeffs_, SingleRateMethodCoefficients* inner_coeffs_, Problem* problem, int problem_dimension_, WeightedErrorNorm* err_norm) :
	inner_rhs_funcs(coeffs_, problem, problem_dimension_),
	implicit_solve_residual(&(MRIGARKFixedStep::inner_rhs_funcs)),
	explicit_solve_problem(&(MRIGARKFixedStep::inner_rhs_funcs)),
	dirk(inner_coeffs_, &(MRIGARKFixedStep::explicit_solve_problem), problem_dimension_, err_norm),
	newton_solver(&(MRIGARKFixedStep::implicit_solve_residual), 20, 1.0, problem_dimension_, err_norm)
	{
		coeffs = coeffs_;
		problem_dimension = problem_dimension_;
		num_stages = coeffs_->num_stages;
		num_groups = coeffs_->num_groups;
		use_embedding = false;
		
		declare_vectors();
	}
	
	MRIGARKFixedStep(MRICoefficients* coeffs_, SingleRateMethodCoefficients* inner_coeffs_, Problem* problem, int problem_dimension_, WeightedErrorNorm* err_norm, bool use_embedding_) :
	inner_rhs_funcs(coeffs_, problem, problem_dimension_),
	implicit_solve_residual(&(MRIGARKFixedStep::inner_rhs_funcs)),
	explicit_solve_problem(&(MRIGARKFixedStep::inner_rhs_funcs)),
	dirk(inner_coeffs_, &(MRIGARKFixedStep::explicit_solve_problem), problem_dimension_, err_norm),
	newton_solver(&(MRIGARKFixedStep::implicit_solve_residual), 20, 1.0, problem_dimension_, err_norm)
	{
		coeffs = coeffs_;
		problem_dimension = problem_dimension_;
		num_stages = coeffs_->num_stages;
		num_groups = coeffs_->num_groups;
		use_embedding = use_embedding_;
		
		declare_vectors();
	}
	
	
	void step_solution(double t, double H, int M, vec* y_prev, vec* solution_vec) {
		y_stages.zeros();
		inner_rhs_funcs.reset_stage_func_eval_storage();
		inner_rhs_funcs.set_step_dependent_data(H, t);
		y_stages.col(0) = *y_prev;
		inner_rhs_funcs.store_stage_func_eval(y_prev,0);
		
		double interval_start;
		double interval_end;
		int stage_initial_condition_index;
		int embedding_shift;
		
		for(std::vector<int> stage_group : coeffs->stage_groups) {
			interval_start = 0.0;
			interval_end = 0.0;
			stage_initial_condition_index = 0;
			embedding_shift = 0;
			for(int stage_group_index=0; stage_group_index<stage_group.size(); stage_group_index++) {
				stage_index = stage_group[stage_group_index];
				inner_rhs_funcs.set_stage_dependent_data(stage_index, false);
				if (stage_index == coeffs->num_stages-1 && use_embedding) {
					// If testing embeddings.
					inner_rhs_funcs.set_stage_dependent_data(stage_index, true);
					embedding_shift = 1;
				}
				

				if (coeffs->method_type == 0) {
					interval_start = t+coeffs->c(stage_index-1)*H;
					interval_end = t+coeffs->c(stage_index)*H;
					stage_initial_condition_index = stage_index-1;
				} else if (coeffs->method_type == 1) {
					interval_end = coeffs->c(stage_index)*H;
				}
				v_0 = y_stages.col(stage_initial_condition_index);
				
				if (coeffs->method_type == 0) {
					double gbar = inner_rhs_funcs.gamma_bar(stage_index + embedding_shift,stage_index);
					double delta_c = coeffs->c(stage_index) - coeffs->c(stage_index-1);
					//printf("stage_index: %d, delta_c: %.16f, c(%d): %.16f, c(%d): %.16f\n",stage_index,delta_c,stage_index,coeffs->c(stage_index),stage_index-1,coeffs->c(stage_index-1));
					if (delta_c == 0) {
						if(gbar != 0.0) {
							inner_rhs_funcs.implicit_set_previous_terms();
							implicit_step(t);
						} else  {
							erk_step();
						}
					} else {
						// No support for methods where delta_C != 0 and gbar != 0 simultaneously.
						// Therefore assume all cases where delta_C != 0, gbar = 0, corresponding to ODE stage solve.
						explicit_solve(H,M,interval_start,interval_end);
					}
				} else if (coeffs->method_type == 1) {
					explicit_solve(H,M,interval_start,interval_end);
				}
				y_stages.col(stage_index) = v_H;
				
				interval_start = interval_end;
				stage_initial_condition_index = stage_index;
				
				// If not last stage, store function evals of the stage
				if (stage_index < num_stages-1) {
					inner_rhs_funcs.store_stage_func_eval(&v_H,stage_index);
				}
			}
		}
		*solution_vec = y_stages.col(num_stages-1);
	}

	void implicit_step(double t) {
		newton_solver.solve(t, &v_0, &newton_ret);
		v_H = newton_ret.y;
	}
	
	void erk_step() {
		inner_rhs_funcs.erk_step(&v_0, &v_H);
	}

	void explicit_solve(double H, int M, double interval_start, double interval_end) {
		//printf("H: %.16f, interval: ( %.16f, %.16f)\n",H,interval_start,interval_end);
		vec output_tspan = {interval_end};
		mat Y = dirk.solve(interval_start, H/M, &v_0, &output_tspan);
		v_H = Y.col(0);
	}

	void declare_vectors() {
		v_0 = vec(problem_dimension, fill::zeros);
		v_H = vec(problem_dimension, fill::zeros);
		y_stages = mat(problem_dimension, num_stages, fill::zeros);
	}
};

#endif