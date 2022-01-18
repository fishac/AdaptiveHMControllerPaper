#ifndef KAPSPROBLEM_DEFINED__
#define KAPSPROBLEM_DEFINED__

#include "Problem.hpp"
#include "RHS.hpp"
#include "RHSJacobian.hpp"

using namespace std;
using namespace arma;

class KapsFullRHS : public RHS {
public:
	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = -102.0*u+100.0*v*v;
		(*f)(1) = -v*v + u - v;
	}
};

class KapsFastRHS : public RHS {
public:
	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = -102.0*u+100.0*v*v;
		(*f)(1) = 0.0;
	}
};

class KapsSlowRHS : public RHS {
public:
	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = 0.0;
		(*f)(1) = -v*v + u - v;
	}
};

class KapsImplicitRHS : public RHS {
public:
	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = 0.0;
		(*f)(1) = -v*v;
	}
};

class KapsExplicitRHS : public RHS {
public:
	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = 0.0;
		(*f)(1) = u - v;
	}
};

class KapsFullRHSJacobian : public RHSJacobian {
public:
	void evaluate(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);
		
		(*j)(0,0) = -102.0;
		(*j)(0,1) = 100.0*2.0*v;

		(*j)(1,0) = 1.0;
		(*j)(1,1) = -2.0*v - 1.0;
	}
};

class KapsFastRHSJacobian : public RHSJacobian {
public:
	void evaluate(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*j)(0,0) = -102.0;
		(*j)(0,1) = 100.0*2.0*v;
		
		(*j)(1,0) = 0.0;
		(*j)(1,1) = 0.0;
	}
};

class KapsSlowRHSJacobian : public RHSJacobian {
public:
	void evaluate(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*j)(0,0) = 0.0;
		(*j)(0,1) = 0.0;
		
		(*j)(1,0) = 1.0;
		(*j)(1,1) = -2.0*v - 1.0;
	}
};

class KapsImplicitRHSJacobian : public RHSJacobian {
public:
	void evaluate(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*j)(0,0) = 0.0;
		(*j)(0,1) = 0.0;

		(*j)(1,0) = 0.0;
		(*j)(1,1) = -2.0*v;
	}
};

class KapsTrueSolution : public TrueSolution {
public:
	void evaluate(double t, vec* y) {
		(*y)(0) = exp(-2.0*t);
		(*y)(1) = exp(-t);
	}
};

class KapsProblem : public Problem {
public:
	static const int problem_dimension_kaps = 2;
	static constexpr double default_H = std::pow(2.0,-6.0);
	static constexpr double t_0 = 0.0;
	static constexpr double t_f = 2.0;

	KapsFullRHS kaps_full_rhs;
	KapsFastRHS kaps_fast_rhs;
	KapsSlowRHS kaps_slow_rhs;
	KapsImplicitRHS kaps_implicit_rhs;
	KapsExplicitRHS kaps_explicit_rhs;
	KapsFullRHSJacobian kaps_full_rhsjacobian;
	KapsFastRHSJacobian kaps_fast_rhsjacobian;
	KapsSlowRHSJacobian kaps_slow_rhsjacobian;
	KapsImplicitRHSJacobian kaps_implicit_rhsjacobian;
	KapsTrueSolution kaps_true_solution;
	
	KapsProblem() :
	kaps_full_rhs(),
	kaps_fast_rhs(),
	kaps_slow_rhs(),
	kaps_implicit_rhs(),
	kaps_explicit_rhs(),
	kaps_full_rhsjacobian(),
	kaps_fast_rhsjacobian(),
	kaps_slow_rhsjacobian(),
	kaps_implicit_rhsjacobian(),
	kaps_true_solution(),
	Problem("Kaps", problem_dimension_kaps, default_H, t_0, t_f, true, false,
		&kaps_full_rhs,
		&kaps_fast_rhs,
		&kaps_slow_rhs,
		&kaps_implicit_rhs,
		&kaps_explicit_rhs,
		&kaps_full_rhsjacobian,
		&kaps_fast_rhsjacobian,
		&kaps_slow_rhsjacobian,
		&kaps_implicit_rhsjacobian,
		&kaps_true_solution)
	{
		y_0 = { 1.0, 1.0 };
	}
};

#endif