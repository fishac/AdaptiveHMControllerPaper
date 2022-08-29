#ifndef OPTIMALITYSEARCHDRIVER_DEFINED__
#define OPTIMALITYSEARCHDRIVER_DEFINED__

#include <armadillo>
#include <math.h>

#include "MRGARKCoefficients.hpp"

using namespace std;
using namespace arma;

class MRGARKOrderVerificationDriver {
public:
	double tol = 1e-12;
	void run(MRGARKCoefficients* coeffs) {
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

		mat A_ss = coeffs->get_A_ss();
		mat c_ss = coeffs->get_c_ss();
		mat A_ff = coeffs->get_A_ff();
		mat c_ff = coeffs->get_c_ff();
		vec one_vec(A_ss.n_rows,fill::ones);
		vec residual_vec(A_ss.n_rows,fill::ones);

		residual_vec = A_ss*one_vec - c_ss;
		if (norm(residual_vec,2) != 0.0) {
			status = 1;
			printf("\tA_ss * 1s != c_ss\n");
		}

		residual_vec = A_ff*one_vec - c_ff;
		if (norm(residual_vec,2) != 0.0) {
			status = 1;
			printf("\tA_ff * 1s != c_ff\n");
		}

		return status;
	}

	int verify_order_1(MRGARKCoefficients* coeffs) {
		int status = 0;
		double residual;
		vec temp_val(1,fill::zeros);
		vec b_s = coeffs->get_b_s();
		vec one_vec(b_s.n_elem,fill::ones);
		vec b_f = coeffs->get_b_f();

		temp_val = b_s.t()*one_vec;
		residual = temp_val(0) - 1.0;
		if(abs(residual) > tol) {
			status = 1;
			printf("\tb_s.T * 1s != 1, residual: %.16f\n",residual);
		}

		temp_val = b_f.t()*one_vec; 
		residual = temp_val(0) - 1.0;
		if(abs(residual) > tol) {
			status = 1;
			printf("\tb_f.T * 1s != 1, residual: %.16f\n",residual);
		}

		return status;
	}

	int verify_order_2(MRGARKCoefficients* coeffs) {
		int status = 0;
		double residual;

		vec temp_val(1,fill::zeros);
		vec b_s = coeffs->get_b_s();
		vec one_vec(b_s.n_elem,fill::ones);
		mat A_ss = coeffs->get_A_ss();
		vec b_f = coeffs->get_b_f();
		mat A_ff = coeffs->get_A_ff();

		temp_val = b_s.t()*A_ss*one_vec;
		residual = temp_val(0) - 0.5;
		if(abs(residual) > tol) {
			status = 1;
			printf("\tb_s.T * A_ss * 1s != 0.5, residual: %.16f\n",residual);
		}

		temp_val = b_f.t()*A_ff*one_vec;
		residual = temp_val(0) - 0.5;
		if(abs(residual) > tol) {
			status = 1;
			printf("\tb_f.T * A_f * 1s != 0.5, residual: %.16f\n",residual);
		}

		mat A_sf(A_ss.n_rows, A_ss.n_cols, fill::zeros);
		mat A_fs(A_ff.n_rows, A_ff.n_cols, fill::zeros);
		for(int M=1; M<=10; M++) {
			A_sf.zeros();
			for(int l=1; l<=M; l++) {
				A_sf += coeffs->get_A_sfl(l,M);
			}
			temp_val = b_s.t()*A_sf*one_vec;
			residual = temp_val(0) - M/2.0;
			if(abs(residual) > tol) {
				status = 1;
				printf("\tb_s.T * (sum A_sfl) * 1s != M/2 for M: %d, residual: %.16f\n",M,residual);
			}
		}

		for(int M=1; M<=10; M++) {
			A_fs.zeros();
			for(int l=1; l<=M; l++) {
				A_fs += coeffs->get_A_fsl(l,M);
			}
			temp_val = b_s.t()*A_fs*one_vec;
			residual = temp_val(0) - M/2.0;
			if(abs(residual) > tol) {
				status = 1;
				printf("\tb_s.T * (sum A_fsl) * 1s != M/2 for M: %d, residual: %.16f\n",M,residual);
			}
		}

		return status;
	}

	int verify_order_3(MRGARKCoefficients* coeffs) {
		int status = 0;
		double residual;

		vec temp_val(1,fill::zeros);
		vec b_s = coeffs->get_b_s();
		vec one_vec(b_s.n_elem,fill::ones);
		vec b_f = coeffs->get_b_f();
		mat A_ss = coeffs->get_A_ss();
		mat A_ff = coeffs->get_A_ff();
		mat diag_temp(A_ss.n_rows,A_ss.n_cols,fill::zeros);
		mat A_sf(A_ss.n_rows,A_ss.n_cols,fill::zeros);
		mat A_fs(A_ss.n_rows,A_ss.n_cols,fill::zeros);
		mat iden = eye(A_ss.n_rows,A_ss.n_cols);
		mat temp_mat(A_ss.n_rows,A_ss.n_cols,fill::zeros);

		diag_temp = diagmat(A_ss*one_vec);
		temp_val = b_s.t()*diag_temp*A_ss*one_vec;
		residual = temp_val(0) - 1.0/3.0;
		if (abs(residual) > tol) {
			status = 1;
			printf("\tb_s.T * diag(A_ss*1s) * A_ss * 1s != 1/3, residual: %.16f\n",residual);
		}

		diag_temp = diagmat(A_ff*one_vec);
		temp_val = b_f.t()*diag_temp*A_ff*one_vec;
		residual = temp_val(0) - 1.0/3.0;
		if (abs(residual) > tol) {
			status = 1;
			printf("\tb_f.T * diag(A_ff*1s) * A_ff * 1s != 1/3, residual: %.16f\n",residual);
		}

		temp_val = b_s.t()*A_ss*A_ss*one_vec;
		residual = temp_val(0) - 1.0/6.0;
		if (abs(residual) > tol) {
			status = 1;
			printf("\tb_s.T * A_ss * A_ss * 1s != 1/6, residual: %.16f\n",residual);
		}

		temp_val = b_f.t()*A_ff*A_ff*one_vec;
		residual = temp_val(0) - 1.0/6.0;
		if (abs(residual) > tol) {
			status = 1;
			printf("\tb_f.T * A_ff * A_ff * 1s != 1/6, residual: %.16f\n",residual);
		}

		for(int M=1; M<=10; M++) {
			A_sf.zeros();
			for(int l=1; l<=M; l++) {
				A_sf += coeffs->get_A_sfl(l,M);
			}
			diag_temp = diagmat(A_ss*one_vec);
			temp_val = b_s.t()*diag_temp*A_sf*one_vec;
			residual = temp_val(0) - M/3.0;
			if (abs(residual) > tol) {
				status = 1;
				printf("\tb_s.T * diag(A_ss*1s) * sum(A_sfl) * 1s != M/3 for M: %d, residual: %.16f\n",M,residual);
			}
		}

		for(int M=1; M<=10; M++) {
			A_sf.zeros();
			for(int l=1; l<=M; l++) {
				A_sf += coeffs->get_A_sfl(l,M);
			}
			diag_temp = diagmat(A_sf*one_vec);
			temp_val = b_s.t()*diag_temp*A_ss*one_vec;
			residual = temp_val(0) - M/3.0;
			if (abs(residual) > tol) {
				status = 1;
				printf("\tb_s.T * diag(sum(A_sfl)*1s) * A_ss * 1s != M/3 for M: %d, residual: %.16f\n",M,residual);
			}
		}

		for(int M=1; M<=10; M++) {
			A_sf.zeros();
			for(int l=1; l<=M; l++) {
				A_sf += coeffs->get_A_sfl(l,M);
			}
			diag_temp = diagmat(A_sf*one_vec);
			temp_val = b_s.t()*diag_temp*A_sf*one_vec;
			residual = temp_val(0) - M*M/3.0;
			if (abs(residual) > tol) {
				status = 1;
				printf("\tb_s.T * diag(sum(A_sfl)*1s) * sum(A_sfl) * 1s != M^2/3 for M: %d, residual: %.16f\n",M,residual);
			}
		}

		for(int M=1; M<=10; M++) {
			A_sf.zeros();
			for(int l=1; l<=M; l++) {
				A_sf += coeffs->get_A_sfl(l,M);
			}
			temp_val = b_s.t()*A_ss*A_sf*one_vec;
			residual = temp_val(0) - M/6.0;
			if (abs(residual) > tol) {
				status = 1;
				printf("\tb_s.T * A_ss * sum(A_sfl) * 1s != M/6 for M: %d, residual: %.16f\n",M,residual);
			}
		}

		for(int M=1; M<=10; M++) {
			A_sf.zeros();
			for(int l=1; l<=M; l++) {
				A_sf += (coeffs->get_A_sfl(l,M)*coeffs->get_A_fsl(l,M));
			}
			temp_val = b_s.t()*A_sf*one_vec;
			residual = temp_val(0) - M/6.0;
			if (abs(residual) > tol) {
				status = 1;
				printf("\tb_s.T * sum(A_sfl*A_fsl) * 1s != M/6 for M: %d, residual: %.16f\n",M,residual);
			}
		}

		for(int M=1; M<=10; M++) {
			A_sf.zeros();
			for(int l=1; l<=M; l++) {
				A_sf += (coeffs->get_A_sfl(l,M)*(A_ff+(l-1)*iden));
			}
			temp_val = b_s.t()*A_sf*one_vec;
			residual = temp_val(0) - M*M/6.0;
			if (abs(residual) > tol) {
				status = 1;
				printf("\tb_s.T * sum(A_sfl*(A_ff+(l-1)*I)*1s) != M/6 for M: %d, residual: %.16f\n",M,residual);
			}
		}

		for(int M=1; M<=10; M++) {
			A_fs.zeros();
			for(int l=1; l<=M; l++) {
				A_fs += coeffs->get_A_fsl(l,M);
			}
			temp_val = b_f.t()*A_fs*A_ss*one_vec;
			residual = temp_val(0) - M/6.0;
			if (abs(residual) > tol) {
				status = 1;
				printf("\tb_f.T * sum(A_fsl) * A_ss * 1s != M/6 for M: %d, residual: %.16f\n",M,residual);
			}
		}

		for(int M=1; M<=10; M++) {
			A_fs.zeros();
			A_sf.zeros();
			for(int l=1; l<=M; l++) {
				A_sf.zeros();
				for(int mu=1; mu<=M; mu++) {
					A_sf += coeffs->get_A_sfl(mu,M);
				}
				A_fs += coeffs->get_A_fsl(l,M)*A_sf;
			}
			temp_val = b_f.t()*A_fs*one_vec;
			residual = temp_val(0) - M*M/6.0;
			if (abs(residual) > tol) {
				status = 1;
				printf("\tb_f.T * sum(A_fsl*sum(A_sfl)) * 1s != M*M/6 for M: %d, residual: %.16f\n",M,residual);
			}
		}

		for(int M=1; M<=10; M++) {
			A_fs.zeros();
			temp_mat.zeros();

			for(int l=1; l<=M; l++) {
				A_fs += coeffs->get_A_fsl(l,M);
			}
			for(int mu=1; mu<=M-1; mu++) {
				for(int l=1; l<=mu; l++) {
					temp_mat += coeffs->get_A_fsl(l,M);
				}
			}

			temp_val = b_f.t()*(A_ff*A_fs + temp_mat)*one_vec;
			residual = temp_val(0) - M*M/6.0;
			if (abs(residual) > tol) {
				status = 1;
				printf("\tb_f.T * (A_ss*sum(A_fsl) + sum(sum(A_fsl))) * 1s != M^2/6 for M: %d, residual: %.16f\n",M,residual);
			}
		}

		for(int M=1; M<=10; M++) {
			A_fs.zeros();
			for(int l=1; l<=M; l++) {
				A_fs += diagmat(coeffs->get_A_fsl(l,M)*one_vec)*coeffs->get_A_fsl(l,M);
			}
			temp_val = b_f.t()*A_fs*one_vec;
			residual = temp_val(0) - M/3.0;
			if (abs(residual) > tol) {
				status = 1;
				printf("\tb_f.T * sum(diag(A_fsl*1s)*A_fsl) * 1s != M/3 for M: %d, residual: %.16f\n",M,residual);
			}
		}

		for(int M=1; M<=10; M++) {
			A_fs.zeros();
			for(int l=1; l<=M; l++) {
				A_fs += (diagmat(coeffs->get_A_fsl(l,M)*one_vec) * (A_ff + (l-1)*coeffs->get_A_fsl(l,M)));
			}
			temp_val = b_f.t()*A_fs*one_vec;
			residual = temp_val(0) - M*M/3.0;
			if (abs(residual) > tol) {
				status = 1;
				printf("\tb_f.T * sum(diag(A_fsl*1s)*(A_ff + (l-1)A_fsl)) * 1s != M^2/3 for M: %d, residual: %.16f\n",M,residual);
			}
		}

		for(int M=1; M<=10; M++) {
			A_fs.zeros();
			temp_mat.zeros();
			for(int l=1; l<=M; l++) {
				A_fs += coeffs->get_A_fsl(l,M);
				temp_mat += (l-1)*coeffs->get_A_fsl(l,M);
			}
			temp_val = b_f.t()*(diagmat(A_ff*one_vec)*A_fs + temp_mat)*one_vec;
			residual = temp_val(0) - M*M/3.0;
			if (abs(residual) > tol) {
				status = 1;
				printf("\tb_f.T * (diag(A_ff*1s)sum(A_fsl) + sum((l-1)A_fsl)) * 1s != M^2/3 for M: %d, residual: %.16f\n",M,residual);
			}
		}

		return status;
	}
};

#endif