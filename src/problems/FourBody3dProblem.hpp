#ifndef FOURBODY3DPROBLEM_DEFINED__
#define FOURBODY3DPROBLEM_DEFINED__

#include "Problem.hpp"
#include "RHS.hpp"
#include "RHSJacobian.hpp"

using namespace std;
using namespace arma;

class FourBody3dFullRHS : public RHS {
public:
	int n = 4;
	int d = 3;
	double g;
	double softening_length_squared;
	double masses[4];
	vec delta_pos;
	vec delta_accel;
	int i_start;
	int j_start;
	double square_sum;

	FourBody3dFullRHS(double g_, double softening_length_squared_, double masses_[4]) {
		g = g_;
		softening_length_squared = softening_length_squared_;
		memcpy(masses, masses_, sizeof(masses));

		delta_pos = vec(d, fill::zeros);
		delta_accel = vec(d, fill::zeros);
	}

	void evaluate(double t, vec* y, vec* f) {
		f->zeros();

		for(int i=0; i<n*d; i++) {
			(*f)(i) = (*y)(n*d + i);
		}

		for(int i=0; i<n; i++) {
			i_start = i*d;

			for(int j=i+1; j<n; j++) {
				j_start = j*d;

				for(int k=0; k<d; k++) {
					delta_pos(k) = (*y)(j_start + k) - (*y)(i_start + k);
				}

				square_sum = sum(square(delta_pos));
				delta_accel = g*delta_pos/(std::pow(square_sum+softening_length_squared, 1.5));
				for(int k=0; k<d; k++) {
					(*f)(n*d + i_start + k) += masses[j]*delta_accel(k);
				}
				for(int k=0; k<d; k++) {
					(*f)(n*d + j_start + k) -= masses[i]*delta_accel(k);
				}
			}
		}
	}
};

class FourBody3dFastRHS : public RHS {
public:
	int n = 4;
	int d = 3;
	double g;
	double softening_length_squared;
	double masses[4];
	vec delta_pos;
	vec delta_accel;
	int i_start;
	int j_start;
	double square_sum;

	FourBody3dFastRHS(double g_, double softening_length_squared_, double masses_[4]) {
		g = g_;
		softening_length_squared = softening_length_squared_;
		memcpy(masses, masses_, sizeof(masses));

		delta_pos = vec(d, fill::zeros);
		delta_accel = vec(d, fill::zeros);
	}

	void evaluate(double t, vec* y, vec* f) {
		f->zeros();

		for(int i=0; i<n; i++) {
			i_start = i*d;

			for(int j=i+1; j<n; j++) {
				j_start = j*d;

				for(int k=0; k<d; k++) {
					delta_pos(k) = (*y)(j_start + k) - (*y)(i_start + k);
				}

				square_sum = sum(square(delta_pos));
				delta_accel = g*delta_pos/(std::pow(square_sum+softening_length_squared, 1.5));
				for(int k=0; k<d; k++) {
					(*f)(n*d + i_start + k) += masses[j]*delta_accel(k);
				}
				for(int k=0; k<d; k++) {
					(*f)(n*d + j_start + k) -= masses[i]*delta_accel(k);
				}
			}
		}
	}
};

class FourBody3dSlowRHS : public RHS {
public:
	int n = 4;
	int d = 3;
	double g;
	double softening_length_squared;
	double masses[4];
	vec delta_pos;
	vec delta_accel;
	int i_start;
	int j_start;
	double square_sum;

	FourBody3dSlowRHS(double g_, double softening_length_squared_, double masses_[4]) {
		g = g_;
		softening_length_squared = softening_length_squared_;
		memcpy(masses, masses_, sizeof(masses));

		delta_pos = vec(d, fill::zeros);
		delta_accel = vec(d, fill::zeros);
	}

	void evaluate(double t, vec* y, vec* f) {
		f->zeros();
		for(int i=0; i<n*d; i++) {
			(*f)(i) = (*y)(n*d + i);
		}
	}
};

class FourBody3dImplicitRHS : public RHS {
public:
	int n = 4;
	int d = 3;
	double g;
	double softening_length_squared;
	double masses[4];
	vec delta_pos;
	vec delta_accel;
	int i_start;
	int j_start;
	double square_sum;

	FourBody3dImplicitRHS(double g_, double softening_length_squared_, double masses_[4]) {
		g = g_;
		softening_length_squared = softening_length_squared_;
		memcpy(masses, masses_, sizeof(masses));

		delta_pos = vec(d, fill::zeros);
		delta_accel = vec(d, fill::zeros);
	}

	void evaluate(double t, vec* y, vec* f) {}
};

class FourBody3dExplicitRHS : public RHS {
public:
	int n = 4;
	int d = 3;
	double g;
	double softening_length_squared;
	double masses[4];
	vec delta_pos;
	vec delta_accel;
	int i_start;
	int j_start;
	double square_sum;

	FourBody3dExplicitRHS(double g_, double softening_length_squared_, double masses_[4]) {
		g = g_;
		softening_length_squared = softening_length_squared_;
		memcpy(masses, masses_, sizeof(masses));

		delta_pos = vec(d, fill::zeros);
		delta_accel = vec(d, fill::zeros);
	}

	void evaluate(double t, vec* y, vec* f) {}
};

class FourBody3dFullRHSJacobian : public RHSJacobian {
public:
	int n = 4;
	int d = 3;
	double g;
	double softening_length_squared;
	double masses[4];
	vec delta_pos;
	vec delta_accel;
	int i_start;
	int j_start;
	double square_sum;

	FourBody3dFullRHSJacobian(double g_, double softening_length_squared_, double masses_[4]) {
		g = g_;
		softening_length_squared = softening_length_squared_;
		memcpy(masses, masses_, sizeof(masses));

		delta_pos = vec(d, fill::zeros);
		delta_accel = vec(d, fill::zeros);
	}

	void evaluate(double t, vec* y, mat* j) {}
};

class FourBody3dFastRHSJacobian : public RHSJacobian {
public:
	int n = 4;
	int d = 3;
	double g;
	double softening_length_squared;
	double masses[4];
	vec delta_pos;
	vec delta_accel;
	int i_start;
	int j_start;
	double square_sum;

	FourBody3dFastRHSJacobian(double g_, double softening_length_squared_, double masses_[4]) {
		g = g_;
		softening_length_squared = softening_length_squared_;
		memcpy(masses, masses_, sizeof(masses));

		delta_pos = vec(d, fill::zeros);
		delta_accel = vec(d, fill::zeros);
	}

	void evaluate(double t, vec* y, mat* j) {}
};

class FourBody3dSlowRHSJacobian : public RHSJacobian {
public:
	int n = 4;
	int d = 3;
	double g;
	double softening_length_squared;
	double masses[4];
	vec delta_pos;
	vec delta_accel;
	int i_start;
	int j_start;
	double square_sum;

	FourBody3dSlowRHSJacobian(double g_, double softening_length_squared_, double masses_[4]) {
		g = g_;
		softening_length_squared = softening_length_squared_;
		memcpy(masses, masses_, sizeof(masses));

		delta_pos = vec(d, fill::zeros);
		delta_accel = vec(d, fill::zeros);
	}

	void evaluate(double t, vec* y, mat* j) {}
};

class FourBody3dImplicitRHSJacobian : public RHSJacobian {
public:
	int n = 4;
	int d = 3;
	double g;
	double softening_length_squared;
	double masses[4];
	vec delta_pos;
	vec delta_accel;
	int i_start;
	int j_start;
	double square_sum;

	FourBody3dImplicitRHSJacobian(double g_, double softening_length_squared_, double masses_[4]) {
		g = g_;
		softening_length_squared = softening_length_squared_;
		memcpy(masses, masses_, sizeof(masses));

		delta_pos = vec(d, fill::zeros);
		delta_accel = vec(d, fill::zeros);
	}

	void evaluate(double t, vec* y, mat* j) {}
};

class FourBody3dTrueSolution : public TrueSolution {
public:
	void evaluate(double t, vec* y) {}
};

class FourBody3dProblem : public Problem {
public:
	double g = 1.0;
	double softening_length_squared = 0.0;
	double masses[4] = {8.0, 10.0, 12.0, 14.0};
	static const int problem_dimension_fourbody3d = 24;
	static constexpr double default_H = std::pow(2.0,-9.0);
	static constexpr double t_0 = 0.0;
	static constexpr double t_f = 15.0;

	FourBody3dFullRHS fourbody3d_full_rhs;
	FourBody3dFastRHS fourbody3d_fast_rhs;
	FourBody3dSlowRHS fourbody3d_slow_rhs;
	FourBody3dImplicitRHS fourbody3d_implicit_rhs;
	FourBody3dExplicitRHS fourbody3d_explicit_rhs;
	FourBody3dFullRHSJacobian fourbody3d_full_rhsjacobian;
	FourBody3dFastRHSJacobian fourbody3d_fast_rhsjacobian;
	FourBody3dSlowRHSJacobian fourbody3d_slow_rhsjacobian;
	FourBody3dImplicitRHSJacobian fourbody3d_implicit_rhsjacobian;
	FourBody3dTrueSolution fourbody3d_true_solution;
	
	FourBody3dProblem() :
	fourbody3d_full_rhs(g, softening_length_squared, masses),
	fourbody3d_fast_rhs(g, softening_length_squared, masses),
	fourbody3d_slow_rhs(g, softening_length_squared, masses),
	fourbody3d_implicit_rhs(g, softening_length_squared, masses),
	fourbody3d_explicit_rhs(g, softening_length_squared, masses),
	fourbody3d_full_rhsjacobian(g, softening_length_squared, masses),
	fourbody3d_fast_rhsjacobian(g, softening_length_squared, masses),
	fourbody3d_slow_rhsjacobian(g, softening_length_squared, masses),
	fourbody3d_implicit_rhsjacobian(g, softening_length_squared, masses),
	fourbody3d_true_solution(),
	Problem("FourBody3d", problem_dimension_fourbody3d, default_H, t_0, t_f, false, true,
		&fourbody3d_full_rhs,
		&fourbody3d_fast_rhs,
		&fourbody3d_slow_rhs,
		&fourbody3d_implicit_rhs,
		&fourbody3d_explicit_rhs,
		&fourbody3d_full_rhsjacobian,
		&fourbody3d_fast_rhsjacobian,
		&fourbody3d_slow_rhsjacobian,
		&fourbody3d_implicit_rhsjacobian,
		&fourbody3d_true_solution)
	{
		y_0 = { 0.0, 0.0, 0.0, 4.0, 3.0, 1.0, 3.0, -4.0, -2.0, -3.0, 4.0, 5.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	}
};

#endif