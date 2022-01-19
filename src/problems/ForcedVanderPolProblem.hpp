#ifndef FORCEDVANDERPOLPROBLEM_DEFINED__
#define FORCEDVANDERPOLPROBLEM_DEFINED__

#include "Problem.hpp"
#include "RHS.hpp"
#include "RHSJacobian.hpp"

using namespace std;
using namespace arma;

class ForcedVanderPolFullRHS : public RHS {
public:
	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = v;
		(*f)(1) = -u - 8.53*(u*u-1)*v + 1.2*sin(M_PI/5.0 * t);
	}
};

class ForcedVanderPolFastRHS : public RHS {
public:
	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = 0.0;
		(*f)(1) = -8.53*(u*u-1)*v + 1.2*sin(M_PI/5.0 * t);
	}
};

class ForcedVanderPolSlowRHS : public RHS {
public:
	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = v;
		(*f)(1) = -u;
	}
};

class ForcedVanderPolImplicitRHS : public RHS {
public:
	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = 0.0;
		(*f)(1) = -u;
	}
};

class ForcedVanderPolExplicitRHS : public RHS {
public:
	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = v;
		(*f)(1) = 0.0;
	}
};

class ForcedVanderPolFullRHSJacobian : public RHSJacobian {
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

class ForcedVanderPolFastRHSJacobian : public RHSJacobian {
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

class ForcedVanderPolSlowRHSJacobian : public RHSJacobian {
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

class ForcedVanderPolImplicitRHSJacobian : public RHSJacobian {
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

class ForcedVanderPolTrueSolution : public TrueSolution {
public:
	void evaluate(double t, vec* y) {}
};

class ForcedVanderPolProblem : public Problem {
public:
	static const int problem_dimension_ForcedVanderPol = 2;
	static constexpr double default_H = std::pow(2.0,-6.0);
	static constexpr double t_0 = 0.0;
	static constexpr double t_f = 25.0;

	ForcedVanderPolFullRHS ForcedVanderPol_full_rhs;
	ForcedVanderPolFastRHS ForcedVanderPol_fast_rhs;
	ForcedVanderPolSlowRHS ForcedVanderPol_slow_rhs;
	ForcedVanderPolImplicitRHS ForcedVanderPol_implicit_rhs;
	ForcedVanderPolExplicitRHS ForcedVanderPol_explicit_rhs;
	ForcedVanderPolFullRHSJacobian ForcedVanderPol_full_rhsjacobian;
	ForcedVanderPolFastRHSJacobian ForcedVanderPol_fast_rhsjacobian;
	ForcedVanderPolSlowRHSJacobian ForcedVanderPol_slow_rhsjacobian;
	ForcedVanderPolImplicitRHSJacobian ForcedVanderPol_implicit_rhsjacobian;
	ForcedVanderPolTrueSolution ForcedVanderPol_true_solution;
	
	ForcedVanderPolProblem() :
	ForcedVanderPol_full_rhs(),
	ForcedVanderPol_fast_rhs(),
	ForcedVanderPol_slow_rhs(),
	ForcedVanderPol_implicit_rhs(),
	ForcedVanderPol_explicit_rhs(),
	ForcedVanderPol_full_rhsjacobian(),
	ForcedVanderPol_fast_rhsjacobian(),
	ForcedVanderPol_slow_rhsjacobian(),
	ForcedVanderPol_implicit_rhsjacobian(),
	ForcedVanderPol_true_solution(),
	Problem("ForcedVanderPol", problem_dimension_ForcedVanderPol, default_H, t_0, t_f, false, false,
		&ForcedVanderPol_full_rhs,
		&ForcedVanderPol_fast_rhs,
		&ForcedVanderPol_slow_rhs,
		&ForcedVanderPol_implicit_rhs,
		&ForcedVanderPol_explicit_rhs,
		&ForcedVanderPol_full_rhsjacobian,
		&ForcedVanderPol_fast_rhsjacobian,
		&ForcedVanderPol_slow_rhsjacobian,
		&ForcedVanderPol_implicit_rhsjacobian,
		&ForcedVanderPol_true_solution)
	{
		y_0 = { 1.45, 0.0 };
	}
};

#endif