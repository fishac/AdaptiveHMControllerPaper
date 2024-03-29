#ifndef LIENARDPROBLEM_DEFINED__
#define LIENARDPROBLEM_DEFINED__

#include "Problem.hpp"

using namespace std;
using namespace arma;

class LienardProblem : public Problem {
public:
	LienardProblem() {
		name = "Lienard";
		problem_dimension = 2;
		default_H = std::pow(2.0,-6.0);
		t_0 = 0.0;
		t_f = 25.0;
		has_true_solution = false;
		explicit_only = false;
		y_0 = { 1.45, 0.0 };
	}
	
	void full_rhs_custom(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = v;
		(*f)(1) = -u - 8.53*(u*u-1)*v + 1.2*sin(M_PI/5.0 * t);
	}
	
	void fast_rhs_custom(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = 0.0;
		(*f)(1) = -8.53*(u*u-1)*v + 1.2*sin(M_PI/5.0 * t);
	}
	
	void slow_rhs_custom(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = v;
		(*f)(1) = -u;
	}
	
	void implicit_rhs_custom(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = 0.0;
		(*f)(1) = -u;
	}
	
	void explicit_rhs_custom(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = v;
		(*f)(1) = 0.0;
	}
	
	void linear_rhs_custom(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = v;
		(*f)(1) = -u;
	}
	
	void nonlinear_rhs_custom(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = 0.0;
		(*f)(1) = -8.53*(u*u-1)*v + 1.2*sin(M_PI/5.0 * t);
	}
	
	void full_rhsjacobian_custom(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*j)(0,0) = 0.0;
		(*j)(0,1) = 1.0;

		(*j)(1,0) = -1.0 - 17.06*u*v;
		(*j)(1,1) = -8.53*(u*u-1.0);
	}
	
	void fast_rhsjacobian_custom(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*j)(0,0) = 0.0;
		(*j)(0,1) = 0.0;

		(*j)(1,0) = -17.06*u*v;
		(*j)(1,1) = -8.53*(u*u-1.0);
	}
	
	void slow_rhsjacobian_custom(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*j)(0,0) = 0.0;
		(*j)(0,1) = 1.0;

		(*j)(1,0) = -1.0;
		(*j)(1,1) = 0.0;
	}
	
	void implicit_rhsjacobian_custom(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*j)(0,0) = 0.0;
		(*j)(0,1) = 0.0;

		(*j)(1,0) = -1.0;
		(*j)(1,1) = 0.0;
	}
	
	void linear_rhsjacobian_custom(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*j)(0,0) = 0.0;
		(*j)(0,1) = 1.0;

		(*j)(1,0) = -1.0;
		(*j)(1,1) = 0.0;
	}
};

#endif