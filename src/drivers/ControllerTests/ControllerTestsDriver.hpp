#ifndef CONTROLLERTESTSDRIVER_DEFINED__
#define CONTROLLERTESTSDRIVER_DEFINED__

#include "Problem.hpp"
#include "AdaptiveStepMultiRateMethod.hpp"
#include "MRIGARKAdaptiveMethod.hpp"
#include "MRIGARKAdaptiveStepSlowMeasurement.hpp"
#include "MRIGARKAdaptiveStepFastMeasurement.hpp"
#include "MRIGARKERK33Coefficients.hpp"
#include "MRIGARKIRK21aCoefficients.hpp"
#include "MRIGARKERK45aCoefficients.hpp"
#include "MRIGARKESDIRK34aCoefficients.hpp"
#include "SingleRateMethodCoefficients.hpp"
#include "AdaptiveDIRKMethod.hpp"
#include "HeunEulerERKCoefficients.hpp"
#include "BogackiShampineERKCoefficients.hpp"
#include "ZonneveldERKCoefficients.hpp"
#include "WeightedErrorNorm.hpp"
#include "ConstantConstantController.hpp"
#include "LinearLinearController.hpp"
#include "PIMRController.hpp"
#include "QuadraticQuadraticController.hpp"
#include "PIDMRController.hpp"
#include "IController.hpp"
#include "PIController.hpp"
#include "PIDController.hpp"
#include "GustafssonController.hpp"
#include "Controller.hpp"
#include "FastErrorMeasurementTypes.hpp"

using namespace std;
using namespace arma;

struct stats {
	int total_timesteps; int total_successful_timesteps; int total_microtimesteps; 
	int total_successful_microtimesteps; double rel_err; double abs_err;
	int full_function_evals; int fast_function_evals; int slow_function_evals; 
	int implicit_function_evals;  int explicit_function_evals; 
	int full_jacobian_evals; int fast_jacobian_evals; int slow_jacobian_evals; 
	int implicit_jacobian_evals;
	int status;
};

struct stats_over_time {
	std::vector<double> ts; std::vector<double> Hs; std::vector<int> Ms;
	std::vector<int> full_function_evals; std::vector<int> fast_function_evals; std::vector<int> slow_function_evals;
	std::vector<int> implicit_function_evals; std::vector<int> explicit_function_evals;
	std::vector<int> full_jacobian_evals; std::vector<int> fast_jacobian_evals; std::vector<int> slow_jacobian_evals;
	std::vector<int> implicit_jacobian_evals;
};

class ControllerTestsDriver {
public:
	void save_stats(const char* problem_name, const char* method_name, const char* controller_name, const char* tol_string, const char* measurement_type, stats* solve_stats) {
		char filename[200];
		sprintf(filename, "./output/%s/%s_ControllerTests_%s_%s_%s_%s_stats.csv", problem_name, problem_name, controller_name, tol_string, measurement_type, method_name);
		vec output = {
			(double) solve_stats->total_timesteps, (double) solve_stats->total_successful_timesteps, (double) solve_stats->total_microtimesteps, 
			(double) solve_stats->total_successful_microtimesteps, solve_stats->rel_err, solve_stats->abs_err,
			(double) solve_stats->full_function_evals, (double) solve_stats->fast_function_evals, (double) solve_stats->slow_function_evals,
			(double) solve_stats->implicit_function_evals, (double) solve_stats->explicit_function_evals, 
			(double) solve_stats->full_jacobian_evals, (double) solve_stats->fast_jacobian_evals, (double) solve_stats->slow_jacobian_evals, (double) solve_stats->implicit_jacobian_evals,
			(double) solve_stats->status
		};
		output.save(filename, csv_ascii);
	}
	
	void save_stats_over_time(const char* problem_name, const char* method_name, const char* controller_name, const char* tol_string, const char* measurement_type, stats_over_time* solve_stats) {
		char filename[200];
		sprintf(filename, "./output/%s/%s_ControllerTests_%s_%s_%s_%s_SOT.csv", problem_name, problem_name, controller_name, tol_string, measurement_type, method_name);
		int n_elem = (solve_stats->ts).size();
		mat output(n_elem,12,fill::zeros);
		for(int i=0; i<n_elem; i++) {
			output(i,0) = (solve_stats->ts)[i];
			output(i,1) = (solve_stats->Hs)[i];
			output(i,2) = (solve_stats->Ms)[i];
			output(i,3) = (solve_stats->full_function_evals)[i];
			output(i,4) = (solve_stats->fast_function_evals)[i];
			output(i,5) = (solve_stats->slow_function_evals)[i];
			output(i,6) = (solve_stats->implicit_function_evals)[i];
			output(i,7) = (solve_stats->explicit_function_evals)[i];
			output(i,8) = (solve_stats->full_jacobian_evals)[i];
			output(i,9) = (solve_stats->fast_jacobian_evals)[i];
			output(i,10) = (solve_stats->slow_jacobian_evals)[i];
			output(i,11) = (solve_stats->implicit_jacobian_evals)[i];
		}
		output.save(filename, csv_ascii);
	}

	void run(Problem* problem, double H_0, double M_0, vec* atol, double rtol, mat* Y_true, vec* output_tspan, const char* tol_string, bool allow_explicit, bool allow_implicit) {
		WeightedErrorNorm err_norm(atol, rtol);

		MRIGARKERK33Coefficients mrigarkerk33;
		MRIGARKIRK21aCoefficients mrigarkirk21a;
		MRIGARKERK45aCoefficients mrigarkerk45a;
		MRIGARKESDIRK34aCoefficients mrigarkesdirk34a;

		MRIGARKAdaptiveMethod mrigark_method(
			problem,
			problem->problem_dimension,
			&err_norm
		);


		/////////////////////////
		// MULTI RATE METHODS + MULTI RATE CONTROLLERS
		////////////////////////
		double k1_CC[1] = { 0.42 }; 
		double k2_CC[1] = { 0.44 }; 
		ConstantConstantController CCcontroller(
			1.0,
			1.0,
			1.0,
			0.85,
			k1_CC,
			k2_CC
		);
		
		if (allow_explicit) {
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkerk33, &CCcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkerk45a, &CCcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
		}
		if (!problem->explicit_only && allow_implicit) {
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkirk21a, &CCcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkesdirk34a, &CCcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
		}
		
		double k1_CC_inner[1] = { 0.42 }; 
		double k2_CC_inner[1] = { 0.44 }; 
		ConstantConstantController CCcontroller_inner(
			1.0,
			1.0,
			1.0,
			0.85,
			k1_CC_inner,
			k2_CC_inner
		);
		
		double k1_LL[2] = { 0.82, 0.54 }; 
		double k2_LL[2] = { 0.94, 0.9 }; 
		LinearLinearController LLcontroller(
			1.0,
			1.0,
			1.0,
			0.85,
			k1_LL,
			k2_LL,
			&CCcontroller_inner
		);

		if (allow_explicit) {
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkerk33, &LLcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkerk45a, &LLcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
		}
		if (!problem->explicit_only && allow_implicit) {
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkirk21a, &LLcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkesdirk34a, &LLcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
		}

		double k1_PIMR[2] = { 0.18, 0.86 };
		double k2_PIMR[2] = { 0.34, 0.8 }; 
		PIMRController pimrcontroller(
			1.0,
			1.0,
			1.0,
			0.85,
			k1_PIMR,
			k2_PIMR,
			&CCcontroller_inner
		);

		if (allow_explicit) {
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkerk33, &pimrcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkerk45a, &pimrcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
		}
		if (!problem->explicit_only && allow_implicit) {
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkirk21a, &pimrcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkesdirk34a, &pimrcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
		}

		double k1_PIDMR[3] = { 0.34, 0.1, 0.78 }; 
		double k2_PIDMR[3] = { 0.46, 0.42, 0.74 }; 
		PIDMRController pidmrcontroller(
			1.0,
			1.0,
			1.0,
			0.85,
			k1_PIDMR,
			k2_PIDMR,
			&CCcontroller_inner
		);

		if (allow_explicit) {
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkerk33, &pidmrcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkerk45a, &pidmrcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
		}
		if (!problem->explicit_only && allow_implicit) {
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkirk21a, &pidmrcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkesdirk34a, &pidmrcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
		}

		

		/////////////////////////
		// MULTI RATE METHODS + SINGLE RATE CONTROLLERS
		////////////////////////

		double k1_i[1] = { 1.0 };
		double k2_i[1] = { 0.0 }; 
		IController icontroller(
			1.0,
			1.0,
			1.0,
			0.85,
			k1_i,
			k2_i
		);

		double k1_pi[2] = { 0.6, 0.2 };
		double k2_pi[2] = { 0.0, 0.0 }; 
		PIController picontroller(
			1.0,
			1.0,
			1.0,
			0.85,
			k1_pi,
			k2_pi
		);

		double k1_pid[3] = { 0.49, 0.34, 0.1 };
		double k2_pid[3] = { 0.0, 0.0, 0.0 }; 
		PIDController pidcontroller(
			1.0,
			1.0,
			1.0,
			0.85,
			k1_pid,
			k2_pid
		);

		double k1_g[2] = { 0.6, 0.2 };
		double k2_g[2] = { 0.0, 0.0 }; 
		GustafssonController gcontroller(
			1.0,
			1.0,
			1.0,
			0.85,
			k1_g,
			k2_g,
			&icontroller
		);

		// i-controller
		if (allow_explicit) {
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkerk33, &icontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkerk45a, &icontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
		}
		if (!problem->explicit_only && allow_implicit) {
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkirk21a, &icontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkesdirk34a, &icontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
		}

		// pi-controller
		if (allow_explicit) {
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkerk33, &picontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkerk45a, &picontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
		}
		if (!problem->explicit_only && allow_implicit) {
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkirk21a, &picontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkesdirk34a, &picontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
		}

		// pid-controller
		if (allow_explicit) {
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkerk33, &pidcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkerk45a, &pidcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
		}
		if (!problem->explicit_only && allow_implicit) {
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkirk21a, &pidcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkesdirk34a, &pidcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
		}

		// gustafsson-controller
		if (allow_explicit) {
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkerk33, &gcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkerk45a, &gcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
		}
		if (!problem->explicit_only && allow_implicit) {
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkirk21a, &gcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
			run_single_mrigark_method(problem, &mrigark_method, &mrigarkesdirk34a, &gcontroller, H_0, M_0, Y_true, output_tspan, &err_norm, tol_string);
		}		



		/////////////////////////
		// SINGLE RATE METHODS + SINGLE RATE CONTROLLERS
		////////////////////////
		
		/*
		HeunEulerERKCoefficients heuneuler_coeffs;
		AdaptiveDIRKMethod heuneuler_method(
			&heuneuler_coeffs,
			problem, 
			problem->problem_dimension,
			&err_norm,
			true
		);
		
		BogackiShampineERKCoefficients bogackishampine_coeffs;
		AdaptiveDIRKMethod bogackishampine_method(
			&bogackishampine_coeffs,
			problem, 
			problem->problem_dimension,
			&err_norm,
			true
		);
		
		ZonneveldERKCoefficients zonneveld_coeffs;
		AdaptiveDIRKMethod zonneveld_method(
			&zonneveld_coeffs,
			problem, 
			problem->problem_dimension,
			&err_norm,
			true
		);
		
		double k1_i[1] = { 1.0 };
		double k2_i[1] = { 0.0 }; 
		IController icontroller(
			1.0,
			1.0,
			1.0,
			0.85,
			k1_i,
			k2_i
		);
		run_single_dirk_method(problem, &heuneuler_method, &heuneuler_coeffs, &icontroller, H_0, Y_true, output_tspan, &err_norm, tol_string);
		run_single_dirk_method(problem, &bogackishampine_method, &bogackishampine_coeffs, &icontroller, H_0, Y_true, output_tspan, &err_norm, tol_string);
		run_single_dirk_method(problem, &zonneveld_method, &zonneveld_coeffs, &icontroller, H_0, Y_true, output_tspan, &err_norm, tol_string);
		
		double k1_pi[2] = { 0.6, 0.2 };
		double k2_pi[2] = { 0.0, 0.0 }; 
		PIController picontroller(
			1.0,
			1.0,
			1.0,
			0.85,
			k1_pi,
			k2_pi
		);
		run_single_dirk_method(problem, &heuneuler_method, &heuneuler_coeffs, &picontroller, H_0, Y_true, output_tspan, &err_norm, tol_string);
		run_single_dirk_method(problem, &bogackishampine_method, &bogackishampine_coeffs, &picontroller, H_0, Y_true, output_tspan, &err_norm, tol_string);
		run_single_dirk_method(problem, &zonneveld_method, &zonneveld_coeffs, &picontroller, H_0, Y_true, output_tspan, &err_norm, tol_string);
		
		double k1_pid[3] = { 0.49, 0.34, 0.1 };
		double k2_pid[3] = { 0.0, 0.0, 0.0 }; 
		PIDController pidcontroller(
			1.0,
			1.0,
			1.0,
			0.85,
			k1_pid,
			k2_pid
		);
		run_single_dirk_method(problem, &heuneuler_method, &heuneuler_coeffs, &pidcontroller, H_0, Y_true, output_tspan, &err_norm, tol_string);
		run_single_dirk_method(problem, &bogackishampine_method, &bogackishampine_coeffs, &pidcontroller, H_0, Y_true, output_tspan, &err_norm, tol_string);
		run_single_dirk_method(problem, &zonneveld_method, &zonneveld_coeffs, &pidcontroller, H_0, Y_true, output_tspan, &err_norm, tol_string);
		
		double k1_ig[1] = { 1.0 };
		double k2_ig[1] = { 0.0 }; 
		IController igcontroller(
			1.0,
			1.0,
			1.0,
			0.85,
			k1_ig,
			k2_ig
		);
		
		double k1_g[2] = { 0.6, 0.2 };
		double k2_g[2] = { 0.0, 0.0 }; 
		GustafssonController gcontroller(
			1.0,
			1.0,
			1.0,
			0.85,
			k1_g,
			k2_g,
			&igcontroller
		);
		run_single_dirk_method(problem, &heuneuler_method, &heuneuler_coeffs, &gcontroller, H_0, Y_true, output_tspan, &err_norm, tol_string);
		run_single_dirk_method(problem, &bogackishampine_method, &bogackishampine_coeffs, &gcontroller, H_0, Y_true, output_tspan, &err_norm, tol_string);
		run_single_dirk_method(problem, &zonneveld_method, &zonneveld_coeffs, &gcontroller, H_0, Y_true, output_tspan, &err_norm, tol_string);
		
		*/
	}
	
	void run_single_dirk_method(Problem* problem, AdaptiveDIRKMethod* method, SingleRateMethodCoefficients* coeffs, Controller* controller, double H_0, mat* Y_true, vec* output_tspan, WeightedErrorNorm* err_norm, const char* tol_string) {
		double P = std::min(coeffs->primary_order,coeffs->secondary_order);
		double p = 0.0;
		controller->set_orders(P,p);
		controller->reset();

		AdaptiveSingleRateMethodReturnValue ret;
		controller->reset();
		err_norm->reset_weights();

		method->solve(problem->t_0, H_0, &(problem->y_0), output_tspan, controller, &ret);
		process_singlerate_method(&ret, problem, coeffs->name, controller->name, Y_true, tol_string);

	}

	void run_single_mrigark_method(Problem* problem, MRIGARKAdaptiveMethod* method, MRICoefficients* coeffs, Controller* controller, double H_0, int M_0, mat* Y_true, vec* output_tspan, WeightedErrorNorm* err_norm, const char* tol_string) {
		if (coeffs->primary_order == 1 || coeffs->primary_order == 2) {
			HeunEulerERKCoefficients inner_coeffs;
			run_single_mrigark_method_with_coeffs(problem, method, coeffs, &inner_coeffs, controller, H_0, M_0, Y_true, output_tspan, err_norm, tol_string);
		} else if (coeffs->primary_order == 3) {
			BogackiShampineERKCoefficients inner_coeffs;	
			run_single_mrigark_method_with_coeffs(problem, method, coeffs, &inner_coeffs, controller, H_0, M_0, Y_true, output_tspan, err_norm, tol_string);
		} else if (coeffs->primary_order == 4 || coeffs->primary_order == 5) {
			ZonneveldERKCoefficients inner_coeffs;
			run_single_mrigark_method_with_coeffs(problem, method, coeffs, &inner_coeffs, controller, H_0, M_0, Y_true, output_tspan, err_norm, tol_string);
		}
	}

	void run_single_mrigark_method_with_coeffs(Problem* problem, MRIGARKAdaptiveMethod* method, MRICoefficients* coeffs, SingleRateMethodCoefficients* inner_coeffs, Controller* controller, double H_0, int M_0, mat* Y_true, vec* output_tspan, WeightedErrorNorm* err_norm, const char* tol_string) {
		double P = std::min(coeffs->primary_order,coeffs->secondary_order);
		double p = std::min(inner_coeffs->primary_order,inner_coeffs->secondary_order);
		controller->set_orders(P,p);
		controller->reset();

		// Fast error measurement type to use
		const char* measurement_type = "LASA-mean";

		AdaptiveMultiRateMethodReturnValue ret;
		controller->reset();
		err_norm->reset_weights();

		if (FastError::is_slow_type(measurement_type)) {
			MRIGARKAdaptiveStepSlowMeasurement mrigark_step_sm(
				coeffs, 
				inner_coeffs, 
				problem,
				problem->problem_dimension, 
				err_norm
			);
			method->solve(problem->t_0, H_0, M_0, &(problem->y_0), output_tspan, &mrigark_step_sm, controller, measurement_type, &ret);
			process_multirate_method(&ret, problem, coeffs->name, controller->name, measurement_type, Y_true, tol_string);
		} else {
			MRIGARKAdaptiveStepFastMeasurement mrigark_step_fm(
				coeffs, 
				inner_coeffs, 
				problem,
				problem->problem_dimension, 
				err_norm
			);
			method->solve(problem->t_0, H_0, M_0, &(problem->y_0), output_tspan, &mrigark_step_fm, controller, measurement_type, &ret);
			process_multirate_method(&ret, problem, coeffs->name, controller->name, measurement_type, Y_true, tol_string);
		}
	}

	void process_singlerate_method(AdaptiveSingleRateMethodReturnValue* ret, Problem* problem, const char* instance_name, const char* controller_name, mat* Y_true, const char* tol_string) {
		std::vector<double> ts = ret->ts;
		std::vector<double> hs = ret->hs;
		std::vector<int> Ms(hs.size(), 0);
		mat Y = ret->Y;
		int status = ret->status;

		double abs_err = abs((*Y_true)-Y).max();
		double rel_err = norm((*Y_true)-Y,2)/norm((*Y_true),2);

		if (status == 0) {	
			printf("%s, %s. Total timesteps: %d, total microtimesteps: %d, total successful timesteps: %d, rel err: %.16f, abs err: %.16f\n", instance_name, controller_name, ret->total_timesteps, 0, ret->total_successful_timesteps, rel_err, abs_err);
		} else if (status == 1) {
			printf("%s, %s. Solver failure: h_new too small.\n", instance_name, controller_name);
		} else if (status == 2) {
			printf("%s, %s. Solver failure: h_new nonfinite.\n", instance_name, controller_name);
		} else if (status == 3) {
			printf("%s, %s. Solver failure: NewtonSolver linear solver failure.\n", instance_name, controller_name);
		}

		const char* measurement_type = "0";
		struct stats solve_stats = {
			ret->total_timesteps, ret->total_successful_timesteps, 0, 0, rel_err, abs_err,
			problem->full_function_evals, problem->fast_function_evals, problem->slow_function_evals,
			problem->implicit_function_evals, problem->explicit_function_evals, 
			problem->full_jacobian_evals, problem->fast_jacobian_evals, problem->slow_jacobian_evals, 
			problem->implicit_jacobian_evals, status
		};
		save_stats(problem->name, instance_name, controller_name, tol_string, measurement_type, &solve_stats);	

		struct stats_over_time solve_stats_over_time = {
			ret->ts, ret->hs, Ms,
			ret->full_function_evals, ret->fast_function_evals, ret->slow_function_evals,
			ret->implicit_function_evals, ret->explicit_function_evals,
			ret->full_jacobian_evals, ret->fast_jacobian_evals, ret->slow_jacobian_evals,
			ret->implicit_jacobian_evals
		};
		//save_stats_over_time(problem->name, instance_name, controller_name, tol_string, measurement_type, &solve_stats_over_time);
		problem->reset_eval_counts();
	}

	void process_multirate_method(AdaptiveMultiRateMethodReturnValue* ret, Problem* problem, const char* instance_name, const char* controller_name, const char* measurement_type, mat* Y_true, const char* tol_string) {
		std::vector<double> ts = ret->ts;
		std::vector<double> Hs = ret->Hs;
		std::vector<int> Ms = ret->Ms;
		mat Y = ret->Y;
		int status = ret->status;

		double abs_err = abs((*Y_true)-Y).max();
		double rel_err = norm((*Y_true)-Y,2)/norm((*Y_true),2);

		if (status == 0) {
			printf("%s, %s. Measurement Type: %s. Total timesteps: %d, total microtimesteps: %d, total successful timesteps: %d, rel err: %.16f, abs err: %.16f\n", instance_name, controller_name, measurement_type, ret->total_timesteps, ret->total_microtimesteps, ret->total_successful_timesteps, rel_err, abs_err);
		} else if (status == 1) {
			printf("%s, %s. Measurement Type: %s. Solver failure: H_new too small.\n", instance_name, controller_name, measurement_type);
		} else if (status == 2) {
			printf("%s, %s. Measurement Type: %s. Solver failure: H_new nonfinite.\n", instance_name, controller_name, measurement_type);
		} else if (status == 3) {
			printf("%s, %s. Measurement Type: %s. Solver failure: Excessive iterations without progressing in time.\n", instance_name, controller_name, measurement_type);
		} else if (status == 4) {
			printf("%s, %s. Measurement Type: %s. Solver failure: Inner solver h_new nonfinite.\n", instance_name, controller_name, measurement_type);
		} else if (status == 5) {
			printf("%s, %s. Measurement Type: %s. Solver failure: NewtonSolver linear solver failure.\n", instance_name, controller_name, measurement_type);
		} else if (status == 6) {
			printf("%s, %s. Measurement Type: %s. Solver failure: Excessive iterations with M=M_max, esf=esf_min.\n", instance_name, controller_name, measurement_type);
		} else if (status == 7) {
			printf("%s, %s. Measurement Type: %s. Solver failure: Excessive iterations.\n", instance_name, controller_name, measurement_type);
		} else if (status == 8) {
			printf("%s, %s. Measurement Type: %s. Solver failure: Excessive failures in recent steps.\n", instance_name, controller_name, measurement_type);
		}
		
		struct stats solve_stats = {
			ret->total_timesteps, ret->total_successful_timesteps, ret->total_microtimesteps, 
			ret->total_successful_microtimesteps, rel_err, abs_err,
			problem->full_function_evals, problem->fast_function_evals, problem->slow_function_evals,
			problem->implicit_function_evals, problem->explicit_function_evals, 
			problem->full_jacobian_evals, problem->fast_jacobian_evals, problem->slow_jacobian_evals, 
			problem->implicit_jacobian_evals, status
		};
		save_stats(problem->name, instance_name, controller_name, tol_string, measurement_type, &solve_stats);
			
		struct stats_over_time solve_stats_over_time = {
			ret->ts, ret->Hs, ret->Ms,
			ret->full_function_evals, ret->fast_function_evals, ret->slow_function_evals,
			ret->implicit_function_evals, ret->explicit_function_evals,
			ret->full_jacobian_evals, ret->fast_jacobian_evals, ret->slow_jacobian_evals,
			ret->implicit_jacobian_evals
		};
		//save_stats_over_time(problem->name, instance_name, controller_name, tol_string, measurement_type, &solve_stats_over_time);
		problem->reset_eval_counts();
	}
};

#endif