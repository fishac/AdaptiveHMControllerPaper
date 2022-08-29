#ifndef EX4_EX4_3_2_A_COEFFICIENTS_DEFINED__
#define EX4_EX4_3_2_A_COEFFICIENTS_DEFINED__

#include "MRGARKCoefficients.hpp"

using namespace arma;

class EX4_EX4_3_2_A_Coefficients: public MRGARKCoefficients {
public:
	EX4_EX4_3_2_A_Coefficients() {
		name = "EX4_EX4_3_2_A";
		num_stages = 4;
		primary_order = 3.0;
		secondary_order = 2.0;

		A_ff = mat(4,4,fill::zeros);
		A_ff(1,0) = 1.0/3.0;
		A_ff(2,1) = 5.0/9.0;
		A_ff(3,0) = 833.0/7680.0;
		A_ff(3,1) = 833.0/9216.0;
		A_ff(3,2) = 3213.0/5120.0;

		A_ss = mat(4,4,fill::zeros);
		A_ss(1,0) = 1.0/3.0;
		A_ss(2,1) = 5.0/9.0;
		A_ss(3,0) = 833.0/7680.0;
		A_ss(3,1) = 833.0/9216.0;
		A_ss(3,2) = 3213.0/5120.0;

		b_f = vec(4,fill::zeros);
		b_f(0) = 101.0/714.0;
		b_f(1) = 1.0/3.0;
		b_f(2) = 1.0/6.0;
		b_f(3) = 128.0/357.0;

		b_s = vec(4,fill::zeros);
		b_s(0) = 101.0/714.0;
		b_s(1) = 1.0/3.0;
		b_s(2) = 1.0/6.0;
		b_s(3) = 128.0/357.0;

		d_f = vec(4,fill::zeros);
		d_f(0) = 7.0/40.0;
		d_f(1) = -425.0/8784.0;
		d_f(2) = 100037.0/131760.0;
		d_f(3) = 188.0/1647.0;

		d_s = vec(4,fill::zeros);
		d_s(0) = 7.0/40.0;
		d_s(1) = -425.0/8784.0;
		d_s(2) = 100037.0/131760.0;
		d_s(3) = 188.0/1647.0;

		c_ss = A_ss*vec(4,fill::ones);
		c_ff = A_ff*vec(4,fill::ones);
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
		mat A_fsl = mat(4,4,fill::zeros);
		if (lambda == 1) {
			A_fsl(1,0) = 1.0/(3.0*M);
			A_fsl(2,0) = 5.0*(518.0*M*M*M-2140.0*M*M+2399.0*M-777.0)/(2331.0*M*(3.0*M-4.0));
			A_fsl(3,0) = 17.0*(141932.0*M*M*M-445231.0*M*M+481160.0*M-178710.0)/(852480.0*M*(3.0*M-4.0));
			A_fsl(2,1) = -5.0*(518.0*M*M*M-2140.0*M*M+1622.0*M+259.0)/(2331.0*M*(3.0*M-4.0));
			A_fsl(3,1) = -17.0*(94535.0*M*M*M-228442.0*M*M+142736.0*M-5180.0)/(340992.0*M*(3.0*M-4.0));
			A_fsl(3,2) = 3213.0*M/5120.0;
		} else {
			A_fsl(0,0) = (lambda-1.0)/M;
			A_fsl(1,0) = (3.0*lambda-2.0)/(3.0*M);
			A_fsl(2,0) = (-5965.0*M*M*M+6993.0*lambda*M*M+12092.0*M*M-16317.0*lambda*M-858.0*M+9324.0*lambda-5439.0)/(2331.0*M*(3.0*M*M-7.0*M+4.0));
			A_fsl(3,0) = (-867119.0*M*M*M+511488.0*lambda*M*M+1937719.0*M*M-1193472.0*lambda*M-1006056.0*M+681984.0*lambda-74370.0)/(170496.0*M*(3.0*M*M-7.0*M+4.0));
			A_fsl(2,1) = 5.0*(1193.0*M*M*M-3040.0*M*M+1622.0*M+259.0)/(2331.0*M*(3.0*M*M-7.0*M+4.0));
			A_fsl(3,1) = 17.0*(51007.0*M*M*M-119207.0*M*M+71368.0*M-2590.0)/(170496.0*M*(3.0*M*M-7.0*M+4.0));
		}
		return A_fsl;
	}

	mat get_A_sfl(int lambda, int M) {
		mat A_sfl = mat(4,4,fill::zeros);
		if (lambda == 1) {
			A_sfl(1,0) = (361.0*M-102.0*M*M)/1083.0;
			A_sfl(3,0) = M*(1480461.0*M*M-3944118.0*M+3007130.0)/2772480.0;
			A_sfl(1,1) = 34.0*M*M/361.0;
			A_sfl(2,1) = -5.0*M*(981.0*M-1805.0)/6498.0;
			A_sfl(3,1) = -119.0*M*(3249.0*M*M-20358.0*M+18050.0)/3326976.0;
			A_sfl(2,2) = 5.0*M*(327.0*M-361.0)/2166.0;
			A_sfl(3,2) = -119.0*M*(66063.0*M*M-78954.0*M-18050.0)/5544960.0;
			A_sfl(3,3) = (M-1.0)*M*M;
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
		return (*A_fsl)*vec(4,fill::ones);
	}

	vec get_c_sf(mat* A_sfl) {
		return (*A_sfl)*vec(4,fill::ones);
	}
};

#endif