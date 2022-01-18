#ifndef RHS_DEFINED__
#define RHS_DEFINED__

using namespace arma;

class RHS {
public:
	virtual void evaluate(double t, vec* y, vec* f) {};
	void set_yhat_that(vec yhat_, double that_) {};
};

#endif