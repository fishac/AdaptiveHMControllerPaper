#ifndef PROBLEM_DEFINED__
#define PROBLEM_DEFINED__

#include "RHS.hpp"
#include "RHSJacobian.hpp"
#include "TrueSolution.hpp"

using namespace std;
using namespace arma;

//typedef void (Problem::*EvaluateRHSFunc)(double, vec*, vec*);
//typedef void (Problem::*EvaluateRHSJacobianFunc)(double, vec*, mat*);
//typedef void (Problem::*EvaluateTrueSolutionFunc)(double, vec*);

class ProblemRHS : public RHS {
public:
	RHS* rhs;
	int* function_evals;

	ProblemRHS(RHS* rhs_, int* function_evals_) {
		rhs = rhs_;
		function_evals = function_evals_;
	}

	void evaluate(double t, vec* y, vec* f) {
		(*function_evals)++;
		rhs->evaluate(t,y,f);
	}

	void set_yhat_that(vec yhat, double that) {
		rhs->set_yhat_that(yhat,that);
	}
};

class ProblemRHSJacobian : public RHSJacobian {
public:
	RHSJacobian* rhsjacobian;
	int* function_evals;

	ProblemRHSJacobian(RHSJacobian* rhsjacobian_, int* function_evals_) {
		rhsjacobian = rhsjacobian_;
		function_evals = function_evals_;
	}

	void evaluate(double t, vec* y, mat* j) {
		(*function_evals)++;
		rhsjacobian->evaluate(t,y,j);
	}

	void set_yhat_that(vec yhat, double that) {
		rhsjacobian->set_yhat_that(yhat,that);
	}
};

class ProblemTrueSolution : public TrueSolution {
public:
	TrueSolution* true_solution;
	ProblemTrueSolution(TrueSolution* true_solution_) {
		true_solution = true_solution_;
	}

	void evaluate(double t, vec* y) {
		true_solution->evaluate(t,y);
	}
};

class Problem {
public:
	const char* name;
	int problem_dimension = 0;
	double default_H;
	double t_0;
	double t_f;
	bool has_true_solution;
	bool explicit_only;
	vec y_0;
	int full_function_evals = 0;
	int fast_function_evals = 0;
	int slow_function_evals = 0;
	int implicit_function_evals = 0;
	int explicit_function_evals = 0;
	int full_jacobian_evals = 0;
	int fast_jacobian_evals = 0;
	int slow_jacobian_evals = 0;
	int implicit_jacobian_evals = 0;
	
	ProblemRHS full_rhs;
	ProblemRHS fast_rhs;
	ProblemRHS slow_rhs;
	ProblemRHS implicit_rhs;
	ProblemRHS explicit_rhs;
	ProblemRHSJacobian full_rhsjacobian;
	ProblemRHSJacobian fast_rhsjacobian;
	ProblemRHSJacobian slow_rhsjacobian;
	ProblemRHSJacobian implicit_rhsjacobian;
	ProblemTrueSolution true_solution;

	Problem(const char* name_, int problem_dimension_, double default_H_, 
		double t_0_, double t_f_, bool has_true_solution_,
		bool explicit_only_,
		RHS* instance_full_rhs,
		RHS* instance_fast_rhs,
		RHS* instance_slow_rhs,
		RHS* instance_implicit_rhs,
		RHS* instance_explicit_rhs,
		RHSJacobian* instance_full_rhsjacobian,
		RHSJacobian* instance_fast_rhsjacobian,
		RHSJacobian* instance_slow_rhsjacobian,
		RHSJacobian* instance_implicit_rhsjacobian,
		TrueSolution* instance_true_solution):
	name(name_),
	problem_dimension(problem_dimension_),
	default_H(default_H_),
	t_0(t_0_),
	t_f(t_f_),
	has_true_solution(has_true_solution_),
	explicit_only(explicit_only_),
	full_rhs(instance_full_rhs, &full_function_evals),
	fast_rhs(instance_fast_rhs, &fast_function_evals),
	slow_rhs(instance_slow_rhs, &slow_function_evals),
	implicit_rhs(instance_implicit_rhs, &implicit_function_evals),
	explicit_rhs(instance_explicit_rhs, &explicit_function_evals),
	full_rhsjacobian(instance_full_rhsjacobian, &full_jacobian_evals),
	fast_rhsjacobian(instance_fast_rhsjacobian, &fast_jacobian_evals),
	slow_rhsjacobian(instance_slow_rhsjacobian, &slow_jacobian_evals),
	implicit_rhsjacobian(instance_implicit_rhsjacobian, &implicit_jacobian_evals),
	true_solution(instance_true_solution)
	{}

	void reset_eval_counts() {
		full_function_evals = 0;
		fast_function_evals = 0;
		slow_function_evals = 0;
		implicit_function_evals = 0;
		explicit_function_evals = 0;
		full_jacobian_evals = 0;
		fast_jacobian_evals = 0;
		slow_jacobian_evals = 0;
		implicit_jacobian_evals = 0;
	}

	virtual void set_yhat_that(vec yhat, double that) {}
};

#endif