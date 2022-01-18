#ifndef LIENARDPROBLEM_DEFINED__
#define LIENARDPROBLEM_DEFINED__

#include "Problem.hpp"
#include "RHS.hpp"
#include "RHSJacobian.hpp"

using namespace std;
using namespace arma;

class LienardFullRHS : public RHS {
public:
	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = v;
		(*f)(1) = -u - 8.53*(u*u-1)*v + 1.2*sin(M_PI/5.0 * t);
	}
};

class LienardFastRHS : public RHS {
public:
	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = 0.0;
		(*f)(1) = -8.53*(u*u-1)*v + 1.2*sin(M_PI/5.0 * t);
	}
};

class LienardSlowRHS : public RHS {
public:
	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = v;
		(*f)(1) = -u;
	}
};

class LienardImplicitRHS : public RHS {
public:
	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = 0.0;
		(*f)(1) = -u;
	}
};

class LienardExplicitRHS : public RHS {
public:
	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = v;
		(*f)(1) = 0.0;
	}
};

class LienardFullRHSJacobian : public RHSJacobian {
public:
	void evaluate(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*j)(0,0) = 0.0;
		(*j)(0,1) = 1.0;

		(*j)(1,0) = -1.0 - 17.06*u*v;
		(*j)(1,1) = -8.53*(u*u-1.0);
	}
};

class LienardFastRHSJacobian : public RHSJacobian {
public:
	void evaluate(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*j)(0,0) = 0.0;
		(*j)(0,1) = 0.0;

		(*j)(1,0) = -17.06*u*v;
		(*j)(1,1) = -8.53*(u*u-1.0);
	}
};

class LienardSlowRHSJacobian : public RHSJacobian {
public:
	void evaluate(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*j)(0,0) = 0.0;
		(*j)(0,1) = 1.0;

		(*j)(1,0) = -1.0;
		(*j)(1,1) = 0.0;
	}
};

class LienardImplicitRHSJacobian : public RHSJacobian {
public:
	void evaluate(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*j)(0,0) = 0.0;
		(*j)(0,1) = 0.0;

		(*j)(1,0) = -1.0;
		(*j)(1,1) = 0.0;
	}
};

class LienardTrueSolution : public TrueSolution {
public:
	void evaluate(double t, vec* y) {}
};

class LienardProblem : public Problem {
public:
	static const int problem_dimension_lienard = 2;
	static constexpr double default_H = std::pow(2.0,-6.0);
	static constexpr double t_0 = 0.0;
	static constexpr double t_f = 25.0;

	LienardFullRHS lienard_full_rhs;
	LienardFastRHS lienard_fast_rhs;
	LienardSlowRHS lienard_slow_rhs;
	LienardImplicitRHS lienard_implicit_rhs;
	LienardExplicitRHS lienard_explicit_rhs;
	LienardFullRHSJacobian lienard_full_rhsjacobian;
	LienardFastRHSJacobian lienard_fast_rhsjacobian;
	LienardSlowRHSJacobian lienard_slow_rhsjacobian;
	LienardImplicitRHSJacobian lienard_implicit_rhsjacobian;
	LienardTrueSolution lienard_true_solution;
	
	LienardProblem() :
	lienard_full_rhs(),
	lienard_fast_rhs(),
	lienard_slow_rhs(),
	lienard_implicit_rhs(),
	lienard_explicit_rhs(),
	lienard_full_rhsjacobian(),
	lienard_fast_rhsjacobian(),
	lienard_slow_rhsjacobian(),
	lienard_implicit_rhsjacobian(),
	lienard_true_solution(),
	Problem("Lienard", problem_dimension_lienard, default_H, t_0, t_f, false, false,
		&lienard_full_rhs,
		&lienard_fast_rhs,
		&lienard_slow_rhs,
		&lienard_implicit_rhs,
		&lienard_explicit_rhs,
		&lienard_full_rhsjacobian,
		&lienard_fast_rhsjacobian,
		&lienard_slow_rhsjacobian,
		&lienard_implicit_rhsjacobian,
		&lienard_true_solution)
	{
		y_0 = { 1.45, 0.0 };
	}
};

#endif