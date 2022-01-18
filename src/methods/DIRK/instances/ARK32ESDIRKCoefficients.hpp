#ifndef ARK32ESDIRKCOEFFICIENTS_DEFINED__
#define ARK32ESDIRKCOEFFICIENTS_DEFINED__

#include "SingleRateMethodCoefficients.hpp"

using namespace arma;

class ARK32ESDIRKCoefficients: public SingleRateMethodCoefficients {
public:
	ARK32ESDIRKCoefficients() {
		name = "ARK3(2)4L[2]SA-ESDIRK";
		
		num_stages = 4;
		primary_order = 3.0;
		secondary_order = 2.0;
		
		A = mat(num_stages,num_stages,fill::zeros);
		A(1,0) = 1767732205903.0/4055673282236.0;
		A(1,1) = 1767732205903.0/4055673282236.0;

		A(2,0) = 2746238789719.0/10658868560708.0;
		A(2,1) = -640167445237/6845629431997.0;
		A(2,2) = 1767732205903.0/4055673282236.0;

		A(3,0) = 1471266399579.0/7840856788654.0;
		A(3,1) = -4482444167858.0/7529755066697.0;
		A(3,2) = 11266239266428.0/11593286722821.0;
		A(3,3) = 1767732205903.0/4055673282236.0;

		b = vec(num_stages,fill::zeros);
		b(0) = 0.5;
		b(1) = 0.5;

		d = vec(num_stages,fill::zeros);
		d(0) = 1.0;
		d(1) = 0.0;

		c = A*vec(num_stages,fill::ones);
	}
};

#endif