#ifndef BRUSSELATORPROBLEM_DEFINED__
#define BRUSSELATORPROBLEM_DEFINED__

#include "Problem.hpp"
#include "RHS.hpp"
#include "RHSJacobian.hpp"

using namespace std;
using namespace arma;

class BrusselatorFullRHS : public RHS {
public:
	double a;
	double b;
	double epsilon;

	BrusselatorFullRHS(double a_, double b_, double epsilon_) {
		a = a_;
		b = b_;
		epsilon = epsilon_;
	}

	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);
		double w = (*y)(2);

		(*f)(0) = a - (w + 1.0)*u + u*u*v;
		(*f)(1) = w*u - u*u*v;
		(*f)(2) = (b-w)/epsilon - u*w;
	}
};

class BrusselatorFastRHS : public RHS {
public:
	double a;
	double b;
	double epsilon;

	BrusselatorFastRHS(double a_, double b_, double epsilon_) {
		a = a_;
		b = b_;
		epsilon = epsilon_;
	}

	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);
		double w = (*y)(2);

		(*f)(0) = 0.0;
		(*f)(1) = 0.0;
		(*f)(2) = -w/epsilon;
	}
};

class BrusselatorSlowRHS : public RHS {
public:
	double a;
	double b;
	double epsilon;

	BrusselatorSlowRHS(double a_, double b_, double epsilon_) {
		a = a_;
		b = b_;
		epsilon = epsilon_;
	}

	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);
		double w = (*y)(2);

		(*f)(0) = a - (w + 1.0)*u + u*u*v;
		(*f)(1) = w*u - u*u*v;
		(*f)(2) = b/epsilon - u*w;
	}
};

class BrusselatorImplicitRHS : public RHS {
public:
	double a;
	double b;
	double epsilon;

	BrusselatorImplicitRHS(double a_, double b_, double epsilon_) {
		a = a_;
		b = b_;
		epsilon = epsilon_;
	}

	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);
		double w = (*y)(2);

		(*f)(0) = u*u*v;
		(*f)(1) = -u*u*v;
		(*f)(2) = 0.0;
	}
};

class BrusselatorExplicitRHS : public RHS {
public:
	double a;
	double b;
	double epsilon;

	BrusselatorExplicitRHS(double a_, double b_, double epsilon_) {
		a = a_;
		b = b_;
		epsilon = epsilon_;
	}

	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);
		double w = (*y)(2);

		(*f)(0) = a - (w + 1.0)*u;
		(*f)(1) = w*u;
		(*f)(2) = b/epsilon - u*w;
	}
};

class BrusselatorFullRHSJacobian : public RHSJacobian {
public:
	double a;
	double b;
	double epsilon;

	BrusselatorFullRHSJacobian(double a_, double b_, double epsilon_) {
		a = a_;
		b = b_;
		epsilon = epsilon_;
	}

	void evaluate(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);
		double w = (*y)(2);

		(*j)(0,0) = -(w + 1.0) + 2.0*u*v;
		(*j)(0,1) = u*u;
		(*j)(0,2) = -u;

		(*j)(1,0) = w - 2.0*u*v;
		(*j)(1,1) = -u*u;
		(*j)(1,2) = u;

		(*j)(2,0) = -w;
		(*j)(2,1) = 0.0;
		(*j)(2,2) = -1.0/epsilon -u;
	}
};

class BrusselatorFastRHSJacobian : public RHSJacobian {
public:
	double a;
	double b;
	double epsilon;

	BrusselatorFastRHSJacobian(double a_, double b_, double epsilon_) {
		a = a_;
		b = b_;
		epsilon = epsilon_;
	}

	void evaluate(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);
		double w = (*y)(2);

		(*j)(0,0) = 0.0;
		(*j)(0,1) = 0.0;
		(*j)(0,2) = 0.0;
		
		(*j)(1,0) = 0.0;
		(*j)(1,1) = 0.0;
		(*j)(1,2) = 0.0;
		
		(*j)(2,0) = 0.0;
		(*j)(2,1) = 0.0;
		(*j)(2,2) = -1.0/epsilon;
	}
};

class BrusselatorSlowRHSJacobian : public RHSJacobian {
public:
	double a;
	double b;
	double epsilon;

	BrusselatorSlowRHSJacobian(double a_, double b_, double epsilon_) {
		a = a_;
		b = b_;
		epsilon = epsilon_;
	}

	void evaluate(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);
		double w = (*y)(2);

		(*j)(0,0) = -(w + 1.0) + 2.0*u*v;
		(*j)(0,1) = u*u;
		(*j)(0,2) = -u;
		
		(*j)(1,0) = w - 2.0*u*v;
		(*j)(1,1) = -u*u;
		(*j)(1,2) = u;
		
		(*j)(2,0) = -w;
		(*j)(2,1) = 0.0;
		(*j)(2,2) = -u;
	}
};

class BrusselatorImplicitRHSJacobian : public RHSJacobian {
public:
	double a;
	double b;
	double epsilon;

	BrusselatorImplicitRHSJacobian(double a_, double b_, double epsilon_) {
		a = a_;
		b = b_;
		epsilon = epsilon_;
	}

	void evaluate(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);
		double w = (*y)(2);

		(*j)(0,0) = 2.0*u*v;
		(*j)(0,1) = u*u;
		(*j)(0,2) = 0.0;

		(*j)(1,0) = -2.0*u*v;
		(*j)(1,1) = -u*u;
		(*j)(1,2) = 0.0;

		(*j)(2,0) = 0.0;
		(*j)(2,1) = 0.0;
		(*j)(2,2) = 0.0;
	}
};

class BrusselatorTrueSolution : public TrueSolution {
public:
	void evaluate(double t, vec* y) {}
};

class BrusselatorProblem : public Problem {
public:
	static const int problem_dimension_brusselator = 3;
	static constexpr double default_H = std::pow(2.0,-6.0);
	static constexpr double t_0 = 0.0;
	static constexpr double t_f = 2.0;
	double a = 1.0;
	double b = 3.5;
	double epsilon = 0.01;

	BrusselatorFullRHS bruss_full_rhs;
	BrusselatorFastRHS bruss_fast_rhs;
	BrusselatorSlowRHS bruss_slow_rhs;
	BrusselatorImplicitRHS bruss_implicit_rhs;
	BrusselatorExplicitRHS bruss_explicit_rhs;
	BrusselatorFullRHSJacobian bruss_full_rhsjacobian;
	BrusselatorFastRHSJacobian bruss_fast_rhsjacobian;
	BrusselatorSlowRHSJacobian bruss_slow_rhsjacobian;
	BrusselatorImplicitRHSJacobian bruss_implicit_rhsjacobian;
	BrusselatorTrueSolution bruss_true_solution;
	
	BrusselatorProblem() :
	bruss_full_rhs(a,b,epsilon),
	bruss_fast_rhs(a,b,epsilon),
	bruss_slow_rhs(a,b,epsilon),
	bruss_implicit_rhs(a,b,epsilon),
	bruss_explicit_rhs(a,b,epsilon),
	bruss_full_rhsjacobian(a,b,epsilon),
	bruss_fast_rhsjacobian(a,b,epsilon),
	bruss_slow_rhsjacobian(a,b,epsilon),
	bruss_implicit_rhsjacobian(a,b,epsilon),
	bruss_true_solution(),
	Problem("Brusselator", problem_dimension_brusselator, default_H, t_0, t_f, false, false,
		&bruss_full_rhs,
		&bruss_fast_rhs,
		&bruss_slow_rhs,
		&bruss_implicit_rhs,
		&bruss_explicit_rhs,
		&bruss_full_rhsjacobian,
		&bruss_fast_rhsjacobian,
		&bruss_slow_rhsjacobian,
		&bruss_implicit_rhsjacobian,
		&bruss_true_solution)
	{
		y_0 = { 1.2, 3.1, 3.0 };
	}
};

#endif