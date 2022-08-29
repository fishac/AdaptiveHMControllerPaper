#ifndef EX2_EX2_2_1_A_COEFFICIENTS_DEFINED__
#define EX2_EX2_2_1_A_COEFFICIENTS_DEFINED__

#include "MRGARKCoefficients.hpp"

using namespace arma;

class EX2_EX2_2_1_A_Coefficients: public MRGARKCoefficients {
public:
	EX2_EX2_2_1_A_Coefficients() {
		name = "EX2_EX2_2_1_A";
		num_stages = 2;
		primary_order = 2.0;
		secondary_order = 1.0;
		
		A_ff = mat(2,2,fill::zeros);
		A_ff(1,0) = 2.0/3.0;

		A_ss = mat(2,2,fill::zeros);
		A_ss(1,0) = 2.0/3.0;

		b_f = vec(2,fill::zeros);
		b_f(0) = 1.0/4.0;
		b_f(1) = 3.0/4.0;

		b_s = vec(2,fill::zeros);
		b_s(0) = 1.0/4.0;
		b_s(1) = 3.0/4.0;

		d_f = vec(2,fill::zeros);
		d_f(0) = 1.0;
		d_f(1) = 0.0;

		d_s = vec(2,fill::zeros);
		d_s(0) = 1.0;
		d_s(1) = 0.0;

		c_ss = A_ss*vec(2,fill::ones);
		c_ff = A_ff*vec(2,fill::ones);
	}

	mat get_A_ss() {
		return A_ss;
	}

	mat get_A_ff() {
		return A_ff;
	}

	vec get_b_f() {
		return b_f;
	}

	vec get_b_s() {
		return b_s;
	}

	vec get_d_f() {
		return d_f;
	}

	vec get_d_s() {
		return d_s;
	}

	mat get_A_fsl(int lambda, int M) {
		mat A_fsl = mat(2,2,fill::zeros);
		if (lambda == 1) {
			A_fsl(1,0) = 2.0/(3.0*M);
		} else {
			A_fsl(0,0) = (3.0*M*M*M-11.0*M*M+20.0*lambda*M-20.0*M-20.0*lambda+20.0)/(20.0*(M-1.0)*M);
			A_fsl(0,1) = -(M*(3.0*M-11.0))/(20.0*(M-1.0));
			A_fsl(1,0) = (-3.0*M*M*M-9.0*M*M+60.0*lambda*M-20.0*M-60.0*lambda+20.0)/(60.0*(M-1.0)*M);
			A_fsl(1,1) = (M*(M+3.0))/(20.0*(M-1.0));
		}
		return A_fsl;
	}

	mat get_A_sfl(int lambda, int M) {
		mat A_sfl = mat(2,2,fill::zeros);
		if (lambda == 1) {
			A_sfl(1,0) = -1.0/3.0*(M-2.0)*M;
			A_sfl(1,1) = M*M/3.0;
		}
		return A_sfl;
	}

	vec get_c_ff() {
		return c_ff;
	}

	vec get_c_ss() {
		return c_ss;
	}

	vec get_c_fs(mat* A_fsl) {
		return (*A_fsl)*vec(2,fill::ones);
	}

	vec get_c_sf(mat* A_sfl) {
		return (*A_sfl)*vec(2,fill::ones);
	}
};

#endif