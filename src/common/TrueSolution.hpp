#ifndef TRUESOLUTION_DEFINED__
#define TRUESOLUTION_DEFINED__

using namespace arma;

class TrueSolution {
public:
	virtual void evaluate(double t, vec* y) {}
};

#endif