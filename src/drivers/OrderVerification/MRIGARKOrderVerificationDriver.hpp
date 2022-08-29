#ifndef OPTIMALITYSEARCHDRIVER_DEFINED__
#define OPTIMALITYSEARCHDRIVER_DEFINED__

#include <armadillo>
#include <math.h>

#include "MRIGARKCoefficients.hpp"

using namespace std;
using namespace arma;

class MRIGARKOrderVerificationDriver {
public:
	double tol = 0.0;
	void run(MRIGARKCoefficients* coeffs) {
		int status;
		printf("Verifying order of %s\n",coeffs->name);

		printf("Verifying order 0\n");
		status = verify_order_0(coeffs);
		//if (status == 0) {
		printf("Verifying order 1\n");
		status = verify_order_1(coeffs);
		//} 

		//if (status == 0) {
		printf("Verifying order 2\n");
		status = verify_order_2(coeffs);
		//} 

		//if (status == 0) {
		printf("Verifying order 3\n");
		status = verify_order_3(coeffs);
		//}

		printf("\n");
	}

	int verify_order_0(MRGARKCoefficients* coeffs) {
		int status = 0;

		return status;
	}

	int verify_order_1(MRGARKCoefficients* coeffs) {
		int status = 0;

		return status;
	}

	int verify_order_2(MRGARKCoefficients* coeffs) {
		int status = 0;

		return status;
	}

	int verify_order_3(MRGARKCoefficients* coeffs) {
		int status = 0;

		return status;
	}

	void vector_power(vec* input, vec* output, int k) {
		output->zeros();
		for(int i=0; i<k; i++) {
			output %= input;
		}
	}

	void zeta_k(vec* b, mat* A, )

};

#endif