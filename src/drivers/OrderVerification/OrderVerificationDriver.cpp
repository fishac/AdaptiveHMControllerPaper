#include <armadillo>
#include <math.h>

#include "MRGARKOrderVerificationDriver.hpp"
#include "EX2_EX2_2_1_A_Coefficients.hpp"
#include "EX4_EX4_3_2_A_Coefficients.hpp"

using namespace std;
using namespace arma;

int main() {
	MRGARKOrderVerificationDriver driver;
	EX2_EX2_2_1_A_Coefficients ex2_ex2_2_1_A;
	EX4_EX4_3_2_A_Coefficients ex4_ex4_3_2_A;

	driver.run(&ex2_ex2_2_1_A);
	driver.run(&ex4_ex4_3_2_A);
	
	return 0;
}