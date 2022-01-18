#ifndef RESIDUAL_DEFINED__
#define RESIDUAL_DEFINED__

using namespace arma;

class Residual {
public:
	virtual void evaluate(double t, vec* explicit_data, vec* y_0, vec* y, vec* f) {};
	virtual void evaluate_explicit_data(vec* explicit_data) {};
};

#endif