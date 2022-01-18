#ifndef RHSJACOBIAN_DEFINED__
#define RHSJACOBIAN_DEFINED__

using namespace arma;

class RHSJacobian {
public:
	virtual void evaluate(double t, vec* y, mat* f) {};
	void set_yhat_that(vec yhat_, double that_) {};
};

#endif