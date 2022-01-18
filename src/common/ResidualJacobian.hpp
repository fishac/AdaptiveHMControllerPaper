#ifndef RESIDUALJACOBIAN_DEFINED__
#define RESIDUALJACOBIAN_DEFINED__

using namespace arma;

class ResidualJacobian {
public:
	virtual void evaluate(double t, vec* y, mat* jac) {};
};

#endif