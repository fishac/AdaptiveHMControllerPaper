#ifndef KPRPROBLEM_DEFINED__
#define KPRPROBLEM_DEFINED__

#include "Problem.hpp"
#include "RHS.hpp"
#include "RHSJacobian.hpp"

using namespace std;
using namespace arma;

class KPRFullRHS : public RHS {
public:
	double lambda_f;
	double lambda_s;
	double alpha;
	double beta;
	double epsilon;

	KPRFullRHS(double lambda_f_, double lambda_s_, double alpha_, double beta_, double epsilon_) {
		lambda_f = lambda_f_;
		lambda_s = lambda_s_;
		alpha = alpha_;
		beta = beta_;
		epsilon = epsilon_;
	}

	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = (1.0 - epsilon)*(lambda_f - lambda_s)*(-2.0 + v*v - cos(t))/(2.0*alpha*v) + lambda_f*(-3.0 + u*u - cos(beta*t))/(2.0*u) - beta*sin(beta*t)/(2.0*u);
		(*f)(1) = lambda_s*(-2.0 + v*v - cos(t))/(2.0*v) - alpha*epsilon*(lambda_f - lambda_s)*(-3.0 + u*u - cos(beta*t))/(2.0*u) - sin(t)/(2.0*v);
	}
};

class KPRFastRHS : public RHS {
public:
	double lambda_f;
	double lambda_s;
	double alpha;
	double beta;
	double epsilon;

	KPRFastRHS(double lambda_f_, double lambda_s_, double alpha_, double beta_, double epsilon_) {
		lambda_f = lambda_f_;
		lambda_s = lambda_s_;
		alpha = alpha_;
		beta = beta_;
		epsilon = epsilon_;
	}

	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = (1.0 - epsilon)*(lambda_f - lambda_s)*(-2.0 + v*v - cos(t))/(2.0*alpha*v) + lambda_f*(-3.0 + u*u - cos(beta*t))/(2.0*u) - beta*sin(beta*t)/(2.0*u);
		(*f)(1) = 0.0;
	}
};

class KPRSlowRHS : public RHS {
public:
	double lambda_f;
	double lambda_s;
	double alpha;
	double beta;
	double epsilon;

	KPRSlowRHS(double lambda_f_, double lambda_s_, double alpha_, double beta_, double epsilon_) {
		lambda_f = lambda_f_;
		lambda_s = lambda_s_;
		alpha = alpha_;
		beta = beta_;
		epsilon = epsilon_;
	}

	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = 0.0;
		(*f)(1) = lambda_s*(-2.0 + v*v - cos(t))/(2.0*v) - alpha*epsilon*(lambda_f - lambda_s)*(-3.0 + u*u - cos(beta*t))/(2.0*u) - sin(t)/(2.0*v);
	}
};

class KPRImplicitRHS : public RHS {
public:
	double lambda_f;
	double lambda_s;
	double alpha;
	double beta;
	double epsilon;

	KPRImplicitRHS(double lambda_f_, double lambda_s_, double alpha_, double beta_, double epsilon_) {
		lambda_f = lambda_f_;
		lambda_s = lambda_s_;
		alpha = alpha_;
		beta = beta_;
		epsilon = epsilon_;
	}

	void evaluate(double t, vec* y, vec* f) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*f)(0) = 0.0;
		(*f)(1) = lambda_s*(-2.0 + v*v - cos(t))/(2.0*v) - alpha*epsilon*(lambda_f - lambda_s)*(-3.0 + u*u - cos(beta*t))/(2.0*u);
	}
};

class KPRExplicitRHS : public RHS {
public:
	double lambda_f;
	double lambda_s;
	double alpha;
	double beta;
	double epsilon;

	KPRExplicitRHS(double lambda_f_, double lambda_s_, double alpha_, double beta_, double epsilon_) {
		lambda_f = lambda_f_;
		lambda_s = lambda_s_;
		alpha = alpha_;
		beta = beta_;
		epsilon = epsilon_;
	}

	void evaluate(double t, vec* y, vec* f) {
		double v = (*y)(1);

		(*f)(0) = 0.0;
		(*f)(1) = -sin(t)/(2.0*v);
	}
};

class KPRFullRHSJacobian : public RHSJacobian {
public:
	double lambda_f;
	double lambda_s;
	double alpha;
	double beta;
	double epsilon;

	KPRFullRHSJacobian(double lambda_f_, double lambda_s_, double alpha_, double beta_, double epsilon_) {
		lambda_f = lambda_f_;
		lambda_s = lambda_s_;
		alpha = alpha_;
		beta = beta_;
		epsilon = epsilon_;
	}

	void evaluate(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*j)(0,0) = lambda_f - lambda_f*(u*u-cos(beta*t)-3.0)/(2.0*u*u) + beta*sin(beta*t)/(2.0*u*u);
		(*j)(0,1) = (1.0-epsilon)*(lambda_f - lambda_s)/alpha - (1.0-epsilon)*(lambda_f - lambda_s)*(v*v-cos(t)-2.0)/(2.0*alpha*v*v);

		(*j)(1,0) = -alpha*epsilon*(lambda_f - lambda_s) + alpha*epsilon*(lambda_f - lambda_s)*(-3.0 + u*u - cos(beta*t))/(2.0*u*u);
		(*j)(1,1) = lambda_s - lambda_s*(-2.0 + v*v - cos(t))/(2.0*v*v) + sin(t)/(2.0*v*v);
	}
};

class KPRFastRHSJacobian : public RHSJacobian {
public:
	double lambda_f;
	double lambda_s;
	double alpha;
	double beta;
	double epsilon;

	KPRFastRHSJacobian(double lambda_f_, double lambda_s_, double alpha_, double beta_, double epsilon_) {
		lambda_f = lambda_f_;
		lambda_s = lambda_s_;
		alpha = alpha_;
		beta = beta_;
		epsilon = epsilon_;
	}

	void evaluate(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*j)(0,0) = lambda_f - lambda_f*(u*u-cos(beta*t)-3.0)/(2.0*u*u) + beta*sin(beta*t)/(2.0*u*u);
		(*j)(0,1) = (1.0-epsilon)*(lambda_f - lambda_s)/alpha - (1.0-epsilon)*(lambda_f - lambda_s)*(v*v-cos(t)-2.0)/(2.0*alpha*v*v);
		
		(*j)(1,0) = 0.0;
		(*j)(1,1) = 0.0;
	}
};

class KPRSlowRHSJacobian : public RHSJacobian {
public:
	double lambda_f;
	double lambda_s;
	double alpha;
	double beta;
	double epsilon;

	KPRSlowRHSJacobian(double lambda_f_, double lambda_s_, double alpha_, double beta_, double epsilon_) {
		lambda_f = lambda_f_;
		lambda_s = lambda_s_;
		alpha = alpha_;
		beta = beta_;
		epsilon = epsilon_;
	}

	void evaluate(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*j)(0,0) = 0.0;
		(*j)(0,1) = 0.0;
		
		(*j)(1,0) = -alpha*epsilon*(lambda_f - lambda_s) + alpha*epsilon*(lambda_f - lambda_s)*(-3.0 + u*u - cos(beta*t))/(2.0*u*u);
		(*j)(1,1) = lambda_s - lambda_s*(-2.0 + v*v - cos(t))/(2.0*v*v) + sin(t)/(2.0*v*v);
	}
};

class KPRImplicitRHSJacobian : public RHSJacobian {
public:
	double lambda_f;
	double lambda_s;
	double alpha;
	double beta;
	double epsilon;

	KPRImplicitRHSJacobian(double lambda_f_, double lambda_s_, double alpha_, double beta_, double epsilon_) {
		lambda_f = lambda_f_;
		lambda_s = lambda_s_;
		alpha = alpha_;
		beta = beta_;
		epsilon = epsilon_;
	}

	void evaluate(double t, vec* y, mat* j) {
		double u = (*y)(0);
		double v = (*y)(1);

		(*j)(0,0) = 0.0;
		(*j)(0,1) = 0.0;

		(*j)(1,0) = -alpha*epsilon*(lambda_f - lambda_s) + alpha*epsilon*(lambda_f - lambda_s)*(-3.0 + u*u - cos(beta*t))/(2.0*u*u);
		(*j)(1,1) = lambda_s - lambda_s*(-2.0 + v*v - cos(t))/(2.0*v*v);
	}
};

class KPRTrueSolution : public TrueSolution {
public:
	double beta;
	KPRTrueSolution(double beta_) {
		beta = beta_;
	}

	void evaluate(double t, vec* y) {
		(*y)(0) = sqrt(3.0 + cos(beta*t));
		(*y)(1) = sqrt(2.0 + cos(t));
	}
};

class KPRProblem : public Problem {
public:
	static const int problem_dimension_kpr = 2;
	static constexpr double default_H = M_PI * std::pow(2.0,-6.0);
	static constexpr double t_0 = 0.0;
	static constexpr double t_f = 5.0*M_PI/2.0;

	double lambda_f = -10.0;
	double lambda_s = -1.0;
	double alpha = 1.0;
	double beta = 20.0;
	double epsilon = 0.1;

	KPRFullRHS kpr_full_rhs;
	KPRFastRHS kpr_fast_rhs;
	KPRSlowRHS kpr_slow_rhs;
	KPRImplicitRHS kpr_implicit_rhs;
	KPRExplicitRHS kpr_explicit_rhs;
	KPRFullRHSJacobian kpr_full_rhsjacobian;
	KPRFastRHSJacobian kpr_fast_rhsjacobian;
	KPRSlowRHSJacobian kpr_slow_rhsjacobian;
	KPRImplicitRHSJacobian kpr_implicit_rhsjacobian;
	KPRTrueSolution kpr_true_solution;
	
	KPRProblem() :
	kpr_full_rhs(lambda_f,lambda_s,alpha,beta,epsilon),
	kpr_fast_rhs(lambda_f,lambda_s,alpha,beta,epsilon),
	kpr_slow_rhs(lambda_f,lambda_s,alpha,beta,epsilon),
	kpr_implicit_rhs(lambda_f,lambda_s,alpha,beta,epsilon),
	kpr_explicit_rhs(lambda_f,lambda_s,alpha,beta,epsilon),
	kpr_full_rhsjacobian(lambda_f,lambda_s,alpha,beta,epsilon),
	kpr_fast_rhsjacobian(lambda_f,lambda_s,alpha,beta,epsilon),
	kpr_slow_rhsjacobian(lambda_f,lambda_s,alpha,beta,epsilon),
	kpr_implicit_rhsjacobian(lambda_f,lambda_s,alpha,beta,epsilon),
	kpr_true_solution(beta),
	Problem("KPR", problem_dimension_kpr, default_H, t_0, t_f, true, false,
		&kpr_full_rhs,
		&kpr_fast_rhs,
		&kpr_slow_rhs,
		&kpr_implicit_rhs,
		&kpr_explicit_rhs,
		&kpr_full_rhsjacobian,
		&kpr_fast_rhsjacobian,
		&kpr_slow_rhsjacobian,
		&kpr_implicit_rhsjacobian,
		&kpr_true_solution)
	{
		y_0 = { 2.0, sqrt(3.0) };
	}
};

#endif