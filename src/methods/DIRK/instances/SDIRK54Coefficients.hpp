#ifndef SDIRK54COEFFICIENTS_DEFINED__
#define SDIRK54COEFFICIENTS_DEFINED__

#include "SingleRateMethodCoefficients.hpp"

using namespace arma;

class SDIRK54Coefficients: public SingleRateMethodCoefficients {
public:
	SDIRK54Coefficients() {
		name = "SDIRK-5-4";
		
		num_stages = 5;
		primary_order = 4.0;
		secondary_order = 3.0;
		
		A = mat(num_stages,num_stages,fill::zeros);
		A(0,0) = 0.25;

		A(1,0) = 0.5;
		A(1,1) = 0.25;

		A(2,0) = 17.0/50.0;
		A(2,1) = -1.0/25.0;
		A(2,2) = 0.25;

		A(3,0) = 371.0/1360.0;
		A(3,1) = -137.0/2720.0;
		A(3,2) = 15.0/544.0;
		A(3,3) = 0.25;

		A(4,0) = 25.0/24.0;
		A(4,1) = -49.0/48.0;
		A(4,2) = 125.0/16.0;
		A(4,3) = -85.0/12.0;
		A(4,4) = 1.0/4.0;

		b = vec(num_stages,fill::zeros);
		b(0) = 25.0/24.0;
		b(1) = -49.0/48.0;
		b(2) = 125.0/16.0;
		b(3) = -85.0/12.0;
		b(4) = 1.0/4.0;

		d = vec(num_stages,fill::zeros);
		d(0) = 59.0/48.0;
		d(1) = -17.0/96.0;
		d(2) = 225.0/32.0;
		d(3) = -85.0/12.0;
		d(4) = 0.0;

		c = A*vec(num_stages,fill::ones);
	}
};

#endif