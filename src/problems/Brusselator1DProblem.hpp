#ifndef BRUSSELATOR1DPROBLEM_DEFINED__
#define BRUSSELATOR1DPROBLEM_DEFINED__

#include "Problem.hpp"

using namespace std;
using namespace arma;

class Brusselator1DProblem : public Problem {
public:
	int n = 0;
	double dx = 0.0;
	int u_start = 0;
	int v_start = 0;
	int w_start = 0;
	int u_end = 0;
	int v_end = 0;
	int w_end = 0;
	vec x;
	double a = 1.0;
	double b = 3.5;
	double epsilon = 0.001;
	
	Brusselator1DProblem(int n_) {
		name = "Brusselator1D";
		n = n_;
		problem_dimension = 3*n;
		u_start = 0;
		v_start = n;
		w_start = 2*n;
		u_end = n-1;
		v_end = 2*n-1;
		w_end = 3*n-1;
		default_H = std::pow(2.0,-9.0);
		t_0 = 0.0;
		t_f = 2.0;
		has_true_solution = false;
		explicit_only = false;
		x = linspace(0.0, 1.0, n_);
		dx = x(1)-x(0);
		y_0 = get_y_0();
	}
	
	vec get_y_0() {
		vec y_0_(problem_dimension,fill::zeros);
		for(int i=0; i<n; i++) {
			y_0_(u_start + i) = 1.2 + 0.1*sin(M_PI*x(i));
			y_0_(v_start + i) = 3.1 + 0.1*sin(M_PI*x(i));
			y_0_(w_start + i) = 3.0 + 0.1*sin(M_PI*x(i));
		}
		return y_0_;
	}
	
	double d(double t) {
		return 0.006 + 0.005*cos(M_PI*t);
	}
	
	double r(double t) {
		return 0.6 + 0.5*cos(4.0*M_PI*t);
	}
	
	void full_rhs_custom(double t, vec* y, vec* f) {
		f->zeros();
		
		for(int i=1; i<n-1; i++) {
			double un = (*y)(u_start+i);
			double unm1 = (*y)(u_start+i-1);
			double unp1 = (*y)(u_start+i+1);

			double vn = (*y)(v_start+i);
			double vnm1 = (*y)(v_start+i-1);
			double vnp1 = (*y)(v_start+i+1);
			
			double wn = (*y)(w_start+i);
			double wnm1 = (*y)(w_start+i-1);
			double wnp1 = (*y)(w_start+i+1);
			
			(*f)(u_start+i) = d(t)*(1.0*unm1 -2.0*un + 1.0*unp1)/(dx*dx) + r(t)*(a - (wn+1.0)*un + un*un*vn);
			(*f)(v_start+i) = d(t)*(1.0*vnm1 -2.0*vn + 1.0*vnp1)/(dx*dx) + r(t)*(un*wn - un*un*vn);
			(*f)(w_start+i) = d(t)*(1.0*wnm1 -2.0*wn + 1.0*wnp1)/(dx*dx) + r(t)*((b - wn)/epsilon - un*wn);
		}
	}
	
	void slow_rhs_custom(double t, vec* y, vec* f) {
		f->zeros();
		
		for(int i=1; i<n-1; i++) {
			double un = (*y)(u_start+i);
			double unm1 = (*y)(u_start+i-1);
			double unp1 = (*y)(u_start+i+1);

			double vn = (*y)(v_start+i);
			double vnm1 = (*y)(v_start+i-1);
			double vnp1 = (*y)(v_start+i+1);
			
			double wn = (*y)(w_start+i);
			double wnm1 = (*y)(w_start+i-1);
			double wnp1 = (*y)(w_start+i+1);
			
			(*f)(u_start+i) = d(t)*(1.0*unm1 -2.0*un + 1.0*unp1)/(dx*dx);
			(*f)(v_start+i) = d(t)*(1.0*vnm1 -2.0*vn + 1.0*vnp1)/(dx*dx);
			(*f)(w_start+i) = d(t)*(1.0*wnm1 -2.0*wn + 1.0*wnp1)/(dx*dx);
		}
	}

	void fast_rhs_custom(double t, vec* y, vec* f) {
		f->zeros();
		
		for(int i=1; i<n-1; i++) {
			double un = (*y)(u_start+i);
			double unm1 = (*y)(u_start+i-1);
			double unp1 = (*y)(u_start+i+1);

			double vn = (*y)(v_start+i);
			double vnm1 = (*y)(v_start+i-1);
			double vnp1 = (*y)(v_start+i+1);
			
			double wn = (*y)(w_start+i);
			double wnm1 = (*y)(w_start+i-1);
			double wnp1 = (*y)(w_start+i+1);
			
			(*f)(u_start+i) = r(t)*(a - (wn+1.0)*un + un*un*vn);
			(*f)(v_start+i) = r(t)*(un*wn - un*un*vn);
			(*f)(w_start+i) = r(t)*((b - wn)/epsilon - un*wn);
		}
	}
	
	void implicit_rhs_custom(double t, vec* y, vec* f) {
		f->zeros();
	}
	
	void explicit_rhs_custom(double t, vec* y, vec* f) {
		f->zeros();
	}
	
	void linear_rhs_custom(double t, vec* y, vec* f) {
		f->zeros();
	}
	
	void nonlinear_rhs_custom(double t, vec* y, vec* f) {
		f->zeros();
	}
	
	void full_rhsjacobian_custom(double t, vec* y, mat* j) {
		j->zeros();
		
		for(int i=1; i<n-1; i++) {
			(*j)(u_start+i,u_start+i-1) = d(t);
			(*j)(u_start+i,u_start+i) = d(t)*-2.0 + r(t)*(((*y)(w_start+i)+1) + 2.0*(*y)(u_start+i)*(*y)(v_start+i));
			(*j)(u_start+i,u_start+i+1) = d(t);
			(*j)(u_start+i,v_start+i) = r(t)*(*y)(u_start+i)*(*y)(u_start+i);
			(*j)(u_start+i,w_start+i) = r(t)*(*y)(u_start+i);
			
			(*j)(v_start+i,v_start+i-1) = d(t);
			(*j)(v_start+i,v_start+i) = d(t)*-2.0 + r(t)*(-(*y)(u_start+i)*(*y)(u_start+i));
			(*j)(v_start+i,v_start+i+1) = d(t);
			(*j)(v_start+i,u_start+i) = r(t)*((*y)(w_start+i) - 2.0*(*y)(u_start+i)*(*y)(v_start+i));
			(*j)(v_start+i,w_start+i) = r(t)*(*y)(u_start+i);
			
			(*j)(w_start+i,w_start+i-1) = d(t);
			(*j)(w_start+i,w_start+i) = d(t)*-2.0 + r(t)*(-1.0/epsilon - (*y)(u_start+i));
			(*j)(w_start+i,w_start+i+1) = d(t);
			(*j)(w_start+i,u_start+i) = r(t)*(-(*y)(w_start+i));
			(*j)(w_start+i,v_start+i) = 0.0;
		}
	}
	
	void fast_rhsjacobian_custom(double t, vec* y, mat* j) {
		j->zeros();
		
		for(int i=1; i<n-1; i++) {
			(*j)(u_start+i,u_start+i) = r(t)*(((*y)(w_start+i)+1) + 2.0*(*y)(u_start+i)*(*y)(v_start+i));
			(*j)(u_start+i,v_start+i) = r(t)*(*y)(u_start+i)*(*y)(u_start+i);
			(*j)(u_start+i,w_start+i) = r(t)*(*y)(u_start+i);
			
			(*j)(v_start+i,v_start+i) = r(t)*(-(*y)(u_start+i)*(*y)(u_start+i));
			(*j)(v_start+i,u_start+i) = r(t)*((*y)(w_start+i) - 2.0*(*y)(u_start+i)*(*y)(v_start+i));
			(*j)(v_start+i,w_start+i) = r(t)*(*y)(u_start+i);
			
			(*j)(w_start+i,w_start+i) = r(t)*(-1.0/epsilon - (*y)(u_start+i));
			(*j)(w_start+i,u_start+i) = r(t)*(-(*y)(w_start+i));
			(*j)(w_start+i,v_start+i) = 0.0;
		}
	}
	
	void slow_rhsjacobian_custom(double t, vec* y, mat* j) {
		j->zeros();
		
		for(int i=1; i<n-1; i++) {
			(*j)(u_start+i,u_start+i-1) = d(t);
			(*j)(u_start+i,u_start+i) = d(t)*-2.0;
			(*j)(u_start+i,u_start+i+1) = d(t);
			
			(*j)(v_start+i,v_start+i-1) = d(t);
			(*j)(v_start+i,v_start+i) = d(t)*-2.0;
			(*j)(v_start+i,v_start+i+1) = d(t);
			
			(*j)(w_start+i,w_start+i-1) = d(t);
			(*j)(w_start+i,w_start+i) = d(t)*-2.0;
			(*j)(w_start+i,w_start+i+1) = d(t);
		}
	}
	
	void implicit_rhsjacobian_custom(double t, vec* y, mat* j) {
		j->zeros();
	}
	
	void linear_rhsjacobian_custom(double t, vec* y, mat* j) {
		j->zeros();
	}
};

#endif