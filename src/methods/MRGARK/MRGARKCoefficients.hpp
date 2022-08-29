#ifndef MRGARKCOEFFICIENTS_DEFINED__
#define MRGARKCOEFFICIENTS_DEFINED__

using namespace arma;

class MRGARKCoefficients {
public:
	const char* name;
	mat A_ff;
	mat A_ss;
	vec b_f;
	vec b_s;
	vec d_f;
	vec d_s;
	vec c_ss;
	vec c_ff;
	vec c_fs;
	vec c_sf;
	int num_stages;
	double primary_order;
	double secondary_order;

	virtual mat get_A_ss() {
		mat x(1,1);
		return x;
	}

	virtual mat get_A_ff() {
		mat x(1,1);
		return x;
	}

	virtual vec get_b_f() {
		vec x(1);
		return x;
	}

	virtual vec get_b_s() {
		vec x(1);
		return x;
	}

	virtual vec get_d_f() {
		vec x(1);
		return x;
	}

	virtual vec get_d_s() {
		vec x(1);
		return x;
	}

	virtual mat get_A_fsl(int lambda, int M) {
		mat x(1,1);
		return x;
	}

	virtual mat get_A_sfl(int lambda, int M) {
		mat x(1,1);
		return x;
	}

	virtual vec get_c_ff() {
		vec x(1);
		return x;
	}

	virtual vec get_c_ss() {
		vec x(1);
		return x;
	}

	virtual vec get_c_fs(mat* A_fsl) {
		vec x(1);
		return x;
	}

	virtual vec get_c_sf(mat* A_sfl) {
		vec x(1);
		return x;
	}
};

#endif