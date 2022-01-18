#ifndef BICOUPLINGPROBLEM_DEFINED__
#define BICOUPLINGPROBLEM_DEFINED__

#include "Problem.hpp"
#include "RHS.hpp"
#include "RHSJacobian.hpp"

using namespace std;
using namespace arma;

class BicouplingFullRHS : public RHS {
public:
	double a;
	double b;
	double w;
	double l;
	double p;

	BicouplingFullRHS(double a_, double b_, double w_, double l_, double p_) {
		a = a_;
		b = b_;
		w = w_;
		l = l_;
		p = p_;
	}

	void evaluate(double t, vec* y, vec* f) {
		double y0 = (*y)(0);
		double y1 = (*y)(1);
		double y2 = (*y)(2);

		(*f)(0) = w*y1 - y2 - p*t;
		(*f)(1) = -w*y0;
		(*f)(2) = -l*y2 - l*p*t - p*std::pow(y0 - a*y2/(a*l + b*w) - a*p*t/(a*l + b*w), 2.0) - p*std::pow(y1 - b*y2/(a*l + b*w) - b*p*t/(a*l + b*w), 2.0);
	}
};

class BicouplingFastRHS : public RHS {
public:
	double a;
	double b;
	double w;
	double l;
	double p;

	BicouplingFastRHS(double a_, double b_, double w_, double l_, double p_) {
		a = a_;
		b = b_;
		w = w_;
		l = l_;
		p = p_;
	}

	void evaluate(double t, vec* y, vec* f) {
		double y0 = (*y)(0);
		double y1 = (*y)(1);
		double y2 = (*y)(2);

		(*f)(0) = w*y1;
		(*f)(1) = -w*y0;
		(*f)(2) = 0.0;
	}
};

class BicouplingSlowRHS : public RHS {
public:
	double a;
	double b;
	double w;
	double l;
	double p;

	BicouplingSlowRHS(double a_, double b_, double w_, double l_, double p_) {
		a = a_;
		b = b_;
		w = w_;
		l = l_;
		p = p_;
	}

	void evaluate(double t, vec* y, vec* f) {
		double y0 = (*y)(0);
		double y1 = (*y)(1);
		double y2 = (*y)(2);

		(*f)(0) = -y2 - p*t;
		(*f)(1) = 0.0;
		(*f)(2) = -l*y2 - l*p*t - p*std::pow(y0 - a*y2/(a*l + b*w) - a*p*t/(a*l + b*w), 2.0) - p*std::pow(y1 - b*y2/(a*l + b*w) - b*p*t/(a*l + b*w), 2.0);
	}
};

class BicouplingImplicitRHS : public RHS {
public:
	double a;
	double b;
	double w;
	double l;
	double p;

	BicouplingImplicitRHS(double a_, double b_, double w_, double l_, double p_) {
		a = a_;
		b = b_;
		w = w_;
		l = l_;
		p = p_;
	}

	void evaluate(double t, vec* y, vec* f) {
		double y0 = (*y)(0);
		double y1 = (*y)(1);
		double y2 = (*y)(2);

		(*f)(0) = 0.0;
		(*f)(1) = 0.0;
		(*f)(2) = 0.0;
	}
};

class BicouplingExplicitRHS : public RHS {
public:
	double a;
	double b;
	double w;
	double l;
	double p;

	BicouplingExplicitRHS(double a_, double b_, double w_, double l_, double p_) {
		a = a_;
		b = b_;
		w = w_;
		l = l_;
		p = p_;
	}

	void evaluate(double t, vec* y, vec* f) {
		double y0 = (*y)(0);
		double y1 = (*y)(1);
		double y2 = (*y)(2);

		(*f)(0) = -y2 - p*t;
		(*f)(1) = 0.0;
		(*f)(2) = -l*y2 - l*p*t - p*std::pow(y0 - a*y2/(a*l + b*w) - a*p*t/(a*l + b*w), 2.0) - p*std::pow(y1 - b*y2/(a*l + b*w) - b*p*t/(a*l + b*w), 2.0);
	}
};

class BicouplingFullRHSJacobian : public RHSJacobian {
public:	
	double a;
	double b;
	double w;
	double l;
	double p;

	BicouplingFullRHSJacobian(double a_, double b_, double w_, double l_, double p_) {
		a = a_;
		b = b_;
		w = w_;
		l = l_;
		p = p_;
	}

	void evaluate(double t, vec* y, mat* j) {
		double y0 = (*y)(0);
		double y1 = (*y)(1);
		double y2 = (*y)(2);
		
		(*j)(0,0) = 0.0;
		(*j)(0,1) = w;
		(*j)(0,2) = -1.0;

		(*j)(1,0) = -w;
		(*j)(1,1) = 0.0;
		(*j)(1,2) = 0.0;

		(*j)(2,0) = -2.0*p*(y0 - a*y2/(a*l + b*w) - a*p*t/(a*l + b*w));
		(*j)(2,1) = -2.0*p*(y1 - b*y2/(a*l + b*w) - b*p*t/(a*l + b*w));
		(*j)(2,2) = -l + (2.0*a*p)/(a*l + b*w)*(y0 - a*y2/(a*l + b*w) - a*p*t/(a*l + b*w)) + (2.0*b*p)/(a*l + b*w)*(y1 - b*y2/(a*l + b*w) - b*p*t/(a*l + b*w));
	}
};

class BicouplingFastRHSJacobian : public RHSJacobian {
public:
	double a;
	double b;
	double w;
	double l;
	double p;

	BicouplingFastRHSJacobian(double a_, double b_, double w_, double l_, double p_) {
		a = a_;
		b = b_;
		w = w_;
		l = l_;
		p = p_;
	}

	void evaluate(double t, vec* y, mat* j) {
		double y0 = (*y)(0);
		double y1 = (*y)(1);
		double y2 = (*y)(2);
		
		(*j)(0,0) = 0.0;
		(*j)(0,1) = w;
		(*j)(0,2) = 0.0;

		(*j)(1,0) = -w;
		(*j)(1,1) = 0.0;
		(*j)(1,2) = 0.0;

		(*j)(0,0) = 0.0;
		(*j)(0,1) = 0.0;
		(*j)(0,2) = 0.0;
	}
};

class BicouplingSlowRHSJacobian : public RHSJacobian {
public:
	double a;
	double b;
	double w;
	double l;
	double p;

	BicouplingSlowRHSJacobian(double a_, double b_, double w_, double l_, double p_) {
		a = a_;
		b = b_;
		w = w_;
		l = l_;
		p = p_;
	}

	void evaluate(double t, vec* y, mat* j) {
		double y0 = (*y)(0);
		double y1 = (*y)(1);
		double y2 = (*y)(2);
		
		(*j)(0,0) = 0.0;
		(*j)(0,1) = 0.0;
		(*j)(0,2) = -1.0;

		(*j)(1,0) = 0.0;
		(*j)(1,1) = 0.0;
		(*j)(1,2) = 0.0;

		(*j)(2,0) = -2.0*p*(y0 - a*y2/(a*l + b*w) - a*p*t/(a*l + b*w));
		(*j)(2,1) = -2.0*p*(y1 - b*y2/(a*l + b*w) - b*p*t/(a*l + b*w));
		(*j)(2,2) = -l + (2.0*a*p)/(a*l + b*w)*(y0 - a*y2/(a*l + b*w) - a*p*t/(a*l + b*w)) + (2.0*b*p)/(a*l + b*w)*(y1 - b*y2/(a*l + b*w) - b*p*t/(a*l + b*w));
	}
};

class BicouplingImplicitRHSJacobian : public RHSJacobian {
public:
	double a;
	double b;
	double w;
	double l;
	double p;

	BicouplingImplicitRHSJacobian(double a_, double b_, double w_, double l_, double p_) {
		a = a_;
		b = b_;
		w = w_;
		l = l_;
		p = p_;
	}

	void evaluate(double t, vec* y, mat* j) {
		double y0 = (*y)(0);
		double y1 = (*y)(1);
		double y2 = (*y)(2);
		
		(*j)(0,0) = 0.0;
		(*j)(0,1) = 0.0;
		(*j)(0,2) = 0.0;

		(*j)(1,0) = 0.0;
		(*j)(1,1) = 0.0;
		(*j)(1,2) = 0.0;

		(*j)(0,0) = 0.0;
		(*j)(0,1) = 0.0;
		(*j)(0,2) = 0.0;
	}
};

class BicouplingTrueSolution : public TrueSolution {
public:
	double a;
	double b;
	double w;
	double l;
	double p;

	BicouplingTrueSolution(double a_, double b_, double w_, double l_, double p_) {
		a = a_;
		b = b_;
		w = w_;
		l = l_;
		p = p_;
	}

	void evaluate(double t, vec* y) {
		(*y)(0) = cos(w*t) + a*exp(-l*t);
		(*y)(1) = -sin(w*t) + b*exp(-l*t);
		(*y)(2) = (a*l+b*w)*exp(-l*t) - p*t;
	}
};

class BicouplingProblem : public Problem {
public:
	static const int problem_dimension_bicoupling = 3;
	static constexpr double default_H = std::pow(2.0,-12.0);
	static constexpr double t_0 = 0.0;
	static constexpr double t_f = 1.0;
	double a = 1.0;
	double b = 20.0;
	double w = 100.0;
	double l = 5.0;
	double p = 0.01;

	BicouplingFullRHS bicoupling_full_rhs;
	BicouplingFastRHS bicoupling_fast_rhs;
	BicouplingSlowRHS bicoupling_slow_rhs;
	BicouplingImplicitRHS bicoupling_implicit_rhs;
	BicouplingExplicitRHS bicoupling_explicit_rhs;
	BicouplingFullRHSJacobian bicoupling_full_rhsjacobian;
	BicouplingFastRHSJacobian bicoupling_fast_rhsjacobian;
	BicouplingSlowRHSJacobian bicoupling_slow_rhsjacobian;
	BicouplingImplicitRHSJacobian bicoupling_implicit_rhsjacobian;
	BicouplingTrueSolution bicoupling_true_solution;
	
	BicouplingProblem() :
	bicoupling_full_rhs(a,b,w,l,p),
	bicoupling_fast_rhs(a,b,w,l,p),
	bicoupling_slow_rhs(a,b,w,l,p),
	bicoupling_implicit_rhs(a,b,w,l,p),
	bicoupling_explicit_rhs(a,b,w,l,p),
	bicoupling_full_rhsjacobian(a,b,w,l,p),
	bicoupling_fast_rhsjacobian(a,b,w,l,p),
	bicoupling_slow_rhsjacobian(a,b,w,l,p),
	bicoupling_implicit_rhsjacobian(a,b,w,l,p),
	bicoupling_true_solution(a,b,w,l,p),
	Problem("Bicoupling", problem_dimension_bicoupling, default_H, t_0, t_f, true, false,
		&bicoupling_full_rhs,
		&bicoupling_fast_rhs,
		&bicoupling_slow_rhs,
		&bicoupling_implicit_rhs,
		&bicoupling_explicit_rhs,
		&bicoupling_full_rhsjacobian,
		&bicoupling_fast_rhsjacobian,
		&bicoupling_slow_rhsjacobian,
		&bicoupling_implicit_rhsjacobian,
		&bicoupling_true_solution)
	{
		y_0 = { 1.0 + a, b, a*l+b*w };
	}
};

#endif