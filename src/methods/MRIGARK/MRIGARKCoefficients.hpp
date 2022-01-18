#ifndef MRIGARKCOEFFICIENTS_DEFINED__
#define MRIGARKCOEFFICIENTS_DEFINED__

using namespace arma;

class MRIGARKCoefficients {
public:
	const char* name;
	bool explicit_mrigark;
	std::vector<mat> gammas;
	vec c;
	int num_stages;
	int num_gammas;
	double primary_order;
	double secondary_order;
};

#endif