#ifndef PLEIADESPROBLEM_DEFINED__
#define PLEIADESPROBLEM_DEFINED__

#include "Problem.hpp"
#include "RHS.hpp"
#include "RHSJacobian.hpp"

using namespace std;
using namespace arma;

class PleiadesFullRHS : public RHS {
public:
	int n = 7;
	int d = 2;
	double g;
	double softening_length_squared;
	double masses[7];
	vec delta_pos;
	vec delta_accel;
	int i_start;
	int j_start;
	double square_sum;

	PleiadesFullRHS(double g_, double softening_length_squared_, double masses_[7]) {
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

class PleiadesFastRHS : public RHS {
public:
	int n = 7;
	int d = 2;
	double g;
	double softening_length_squared;
	double masses[7];
	vec delta_pos;
	vec delta_accel;
	int i_start;
	int j_start;
	double square_sum;

	PleiadesFastRHS(double g_, double softening_length_squared_, double masses_[7]) {
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

class PleiadesSlowRHS : public RHS {
public:
	int n = 7;
	int d = 2;
	double g;
	double softening_length_squared;
	double masses[7];
	vec delta_pos;
	vec delta_accel;
	int i_start;
	int j_start;
	double square_sum;

	PleiadesSlowRHS(double g_, double softening_length_squared_, double masses_[7]) {
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

class PleiadesImplicitRHS : public RHS {
public:
	int n = 7;
	int d = 2;
	double g;
	double softening_length_squared;
	double masses[7];
	vec delta_pos;
	vec delta_accel;
	int i_start;
	int j_start;
	double square_sum;

	PleiadesImplicitRHS(double g_, double softening_length_squared_, double masses_[7]) {
		g = g_;
		softening_length_squared = softening_length_squared_;
		memcpy(masses, masses_, sizeof(masses));

		delta_pos = vec(d, fill::zeros);
		delta_accel = vec(d, fill::zeros);
	}

	void evaluate(double t, vec* y, vec* f) {}
};

class PleiadesExplicitRHS : public RHS {
public:
	int n = 7;
	int d = 2;
	double g;
	double softening_length_squared;
	double masses[7];
	vec delta_pos;
	vec delta_accel;
	int i_start;
	int j_start;
	double square_sum;

	PleiadesExplicitRHS(double g_, double softening_length_squared_, double masses_[7]) {
		g = g_;
		softening_length_squared = softening_length_squared_;
		memcpy(masses, masses_, sizeof(masses));

		delta_pos = vec(d, fill::zeros);
		delta_accel = vec(d, fill::zeros);
	}

	void evaluate(double t, vec* y, vec* f) {}
};

class PleiadesFullRHSJacobian : public RHSJacobian {
public:
	int n = 7;
	int d = 2;
	double g;
	double softening_length_squared;
	double masses[7];
	vec delta_pos;
	vec delta_accel;
	int i_start;
	int j_start;
	double square_sum;

	PleiadesFullRHSJacobian(double g_, double softening_length_squared_, double masses_[7]) {
		g = g_;
		softening_length_squared = softening_length_squared_;
		memcpy(masses, masses_, sizeof(masses));

		delta_pos = vec(d, fill::zeros);
		delta_accel = vec(d, fill::zeros);
	}

	void evaluate(double t, vec* y, mat* j) {}
};

class PleiadesFastRHSJacobian : public RHSJacobian {
public:
	int n = 7;
	int d = 2;
	double g;
	double softening_length_squared;
	double masses[7];
	vec delta_pos;
	vec delta_accel;
	int i_start;
	int j_start;
	double square_sum;

	PleiadesFastRHSJacobian(double g_, double softening_length_squared_, double masses_[7]) {
		g = g_;
		softening_length_squared = softening_length_squared_;
		memcpy(masses, masses_, sizeof(masses));

		delta_pos = vec(d, fill::zeros);
		delta_accel = vec(d, fill::zeros);
	}

	void evaluate(double t, vec* y, mat* j) {}
};

class PleiadesSlowRHSJacobian : public RHSJacobian {
public:
	int n = 7;
	int d = 2;
	double g;
	double softening_length_squared;
	double masses[7];
	vec delta_pos;
	vec delta_accel;
	int i_start;
	int j_start;
	double square_sum;

	PleiadesSlowRHSJacobian(double g_, double softening_length_squared_, double masses_[7]) {
		g = g_;
		softening_length_squared = softening_length_squared_;
		memcpy(masses, masses_, sizeof(masses));

		delta_pos = vec(d, fill::zeros);
		delta_accel = vec(d, fill::zeros);
	}

	void evaluate(double t, vec* y, mat* j) {}
};

class PleiadesImplicitRHSJacobian : public RHSJacobian {
public:
	int n = 7;
	int d = 2;
	double g;
	double softening_length_squared;
	double masses[7];
	vec delta_pos;
	vec delta_accel;
	int i_start;
	int j_start;
	double square_sum;

	PleiadesImplicitRHSJacobian(double g_, double softening_length_squared_, double masses_[7]) {
		g = g_;
		softening_length_squared = softening_length_squared_;
		memcpy(masses, masses_, sizeof(masses));

		delta_pos = vec(d, fill::zeros);
		delta_accel = vec(d, fill::zeros);
	}

	void evaluate(double t, vec* y, mat* j) {}
};

class PleiadesTrueSolution : public TrueSolution {
public:
	void evaluate(double t, vec* y) {}
};

class PleiadesProblem : public Problem {
public:
	double g = 1.0;
	double softening_length_squared = 0.0;
	double masses[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
	static const int problem_dimension_pleiades = 28;
	static constexpr double default_H = std::pow(2.0,-9.0);
	static constexpr double t_0 = 0.0;
	static constexpr double t_f = 3.0;

	PleiadesFullRHS pleiades_full_rhs;
	PleiadesFastRHS pleiades_fast_rhs;
	PleiadesSlowRHS pleiades_slow_rhs;
	PleiadesImplicitRHS pleiades_implicit_rhs;
	PleiadesExplicitRHS pleiades_explicit_rhs;
	PleiadesFullRHSJacobian pleiades_full_rhsjacobian;
	PleiadesFastRHSJacobian pleiades_fast_rhsjacobian;
	PleiadesSlowRHSJacobian pleiades_slow_rhsjacobian;
	PleiadesImplicitRHSJacobian pleiades_implicit_rhsjacobian;
	PleiadesTrueSolution pleiades_true_solution;
	
	PleiadesProblem() :
	pleiades_full_rhs(g, softening_length_squared, masses),
	pleiades_fast_rhs(g, softening_length_squared, masses),
	pleiades_slow_rhs(g, softening_length_squared, masses),
	pleiades_implicit_rhs(g, softening_length_squared, masses),
	pleiades_explicit_rhs(g, softening_length_squared, masses),
	pleiades_full_rhsjacobian(g, softening_length_squared, masses),
	pleiades_fast_rhsjacobian(g, softening_length_squared, masses),
	pleiades_slow_rhsjacobian(g, softening_length_squared, masses),
	pleiades_implicit_rhsjacobian(g, softening_length_squared, masses),
	pleiades_true_solution(),
	Problem("Pleiades", problem_dimension_pleiades, default_H, t_0, t_f, false, true,
		&pleiades_full_rhs,
		&pleiades_fast_rhs,
		&pleiades_slow_rhs,
		&pleiades_implicit_rhs,
		&pleiades_explicit_rhs,
		&pleiades_full_rhsjacobian,
		&pleiades_fast_rhsjacobian,
		&pleiades_slow_rhsjacobian,
		&pleiades_implicit_rhsjacobian,
		&pleiades_true_solution)
	{
		y_0 = { 3.0, 3.0, 3.0, -3.0, -1.0, 2.0, -3.0, 0.0, 2.0, 0.0, -2.0, -4.0, 2.0, 4.0,
    		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.25, 0.0, 1.0, 1.75, 0.0, -1.5, 0.0};
	}
};

#endif