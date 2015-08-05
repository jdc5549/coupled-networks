/*
 Disclaimer : This is probably bad code.  Relies on osi.m for error checking.
 Author     : Paul Hines, University of Vermont (paul.hines@uvm.edu)
 Modified from code by Johan Loefberg, ETH Zurich, loefberg@control.ee.ethz.ch
 Known issues:
  * for some reason I can't get it to print messages from Clp/Osi??
  * Option handling is yet to be done...
  * QPs are not yet handled
 */

#include "mex.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <vector>

#include <coin/OsiClpSolverInterface.hpp>
#include <coin/OsiCbcSolverInterface.hpp>


static void error(char * msg) {
	mexPrintf("Error in mexosi.cpp: %s\n",msg);
	throw msg;
}

// Create a derived handler that ensures that fprintf sends data to 
// fprintf does not print to matlab buffer in Windows
class DerivedHandler : public CoinMessageHandler
{
	public: virtual int print() ;
};

int DerivedHandler::print()
{
	mexPrintf(messageBuffer());
	mexPrintf("\n");
	return 0;
}
// mex function usage:
//  [x,y,status] = mexosi(n_vars,n_cons,A,x_lb,x_ub,c,Ax_lb,Ax_ub,isMIP,isQP,vartype,Q,options)
//                        0      1      2 3    4    5 6     7     8     9    10     11 12
void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{
	// Enable printing in MATLAB
	int loglevel = 0;
	DerivedHandler *mexprinter = new DerivedHandler(); // assumed open	
	mexprinter->setLogLevel(loglevel);		 
	// check that we have the right number of inputs
	if(nrhs < 10) mexErrMsgTxt("At least 10 inputs required in call to mexosi. Bug in osi.m?...");
	// check that we have the right number of outputs
	if(nlhs < 3) mexErrMsgTxt("At least 3 ouptuts required in call to mexosi. Bug in osi.m?...");

    // Get pointers to input values
	const int    n_vars = (int)*mxGetPr(prhs[0]);
	const int    n_cons = (int)*mxGetPr(prhs[1]);
	const mxArray    *A =  prhs[2];
	const double  *x_lb =  mxGetPr(prhs[3]);
	const double  *x_ub =  mxGetPr(prhs[4]);
	const double     *c =  mxGetPr(prhs[5]);
	const double *Ax_lb =  mxGetPr(prhs[6]);
	const double *Ax_ub =  mxGetPr(prhs[7]);
	const bool    isMIP = (bool)*(mxLogical*)mxGetData(prhs[8]);
	const bool     isQP = (bool)*(mxLogical*)mxGetData(prhs[9]);
	mxLogical *isinteger = (mxLogical*)mxGetData(prhs[10]);
	const mxArray*    Q = prhs[11];
	// process the options
	int  returnStatus = 0;
	// extract row/col/value data from A
	const mwIndex * A_col_starts = mxGetJc(A);
	const mwIndex * A_row_index  = mxGetIr(A);
	const double  * A_data       = mxGetPr(A);
    // figure out the number of non-zeros in A
    int nnz = (int)(A_col_starts[n_vars] - A_col_starts[0]); // number of non-zeros
    //mexPrintf("nnz = %d, n_vars = %d, n_cons = %d\n",nnz,n_vars,n_cons);

    // we need to convert these into other types of indices for Coin to use them
    std::vector<CoinBigIndex> A_col_starts_coin(A_col_starts,A_col_starts+n_vars+1);
    std::vector<int>          A_row_index_coin(A_row_index,A_row_index+nnz);
    
    // declare the solver
    OsiSolverInterface* pSolver;
// 	initialize the solver
    if ( isMIP ) {
        pSolver = new OsiCbcSolverInterface;
    } else {
        pSolver = new OsiClpSolverInterface;
    }
   
//	OsiCbcSolverInterface is deprecated and CbcModel should be used instead but don't
//	know how to get that working with loadProblem. 
//     OsiCbcSolverInterface solver1;
//     CbcModel model(solver1);
//     CbcMain0(model);
//     OsiSolverInterface * pSolver = model.solver();
    
	if (nrhs>12) { // get stuff out of the options structure if provided
		// Finish me
	}
    
//     mexPrintf("Setting Log Level to 0.\n");
	// load the problem
	mexPrintf("Loading the problem.\n");
	pSolver->loadProblem( n_vars, n_cons, // problem size
						  &A_col_starts_coin[0], &A_row_index_coin[0], A_data, // the A matrix
						  x_lb,  x_ub, c, // the objective and bounds
						  Ax_lb, Ax_ub ); // the constraint bounds
    
//     pSolver->messageHandler()->setLogLevel(0); // This doesn't seem to work
    pSolver->setHintParam(OsiDoReducePrint,true,OsiHintTry);
    
	// deal with integer inputs
	if ( isMIP ) {
		for(int i=0;i<n_vars;i++) {
			if (isinteger[i]) pSolver->setInteger(i);
		}
	}
	if (isQP) {
		error("QP is not working yet");
		// need to call loadQuadraticObjective here ???
	}
    
//     CbcModel model(pSolver);
//     model.solver()->setHintParam(OsiDoReducePrint,true,OsiHintTry);
    
    
	// solve the problem
	//mexPrintf("Trying to solve the problem.\n");
    if (isMIP) {
        pSolver->branchAndBound();
//         model.branchAndBound();
    } else {
//         model.initialSolve();
        pSolver->initialSolve();
    }
	
	// Allocate memory for return data
    plhs[0] = mxCreateDoubleMatrix(n_vars,1, mxREAL); // for the solution
    plhs[1] = mxCreateDoubleMatrix(n_cons,1, mxREAL); // for the constraint prices
    plhs[2] = mxCreateDoubleMatrix(1,1, mxREAL);      // for the return status
    double *x = mxGetPr(plhs[0]);
    double *y = mxGetPr(plhs[1]);
    double *returncode = mxGetPr(plhs[2]);
	
	// Copy solutions if available
	if ( pSolver->isProvenOptimal() ) {
//     if ( model.isProvenOptimal() ) {
//     if ( model.solver()->isProvenOptimal() ) {
        //mexPrintf("Solution found.\n");
		// extract the solutions
		const double * solution = pSolver->getColSolution();
		const double * dualvars = pSolver->getRowPrice();
		// copy the solution to the outpus
		memcpy(x,solution,n_vars*sizeof(double));
		memcpy(y,dualvars,n_cons*sizeof(double));
		*returncode = 1;
	} else {
		if ( pSolver->isProvenPrimalInfeasible() ) {
			mexPrintf("Primal problem is proven infeasible.\n");
			*returncode = 0;
		} else if ( pSolver->isProvenDualInfeasible() ) {
			mexPrintf("Dual problem is proven infeasible.\n");
			*returncode = -1;
		} else if ( pSolver->isPrimalObjectiveLimitReached() ) {
			mexPrintf("The primal objective limit was reached.\n");
			*returncode = -2;
		} else if ( pSolver->isDualObjectiveLimitReached() ) {
			mexPrintf("The dual objective limit was reached.\n");
			*returncode = -3;
		} else if ( pSolver->isIterationLimitReached() ) {
			mexPrintf("The iteration limit was reached\n");
			*returncode = -4;
		}
	}
	// clean up memory
	if ( mexprinter!= NULL) delete mexprinter;	
	delete pSolver;
}
