/*
 ============================================================================
 Name        : GaussSeidel.c
 Author      : Gianluca Gerard
 Version     : v0.1
 Copyright   : Copyright (2015) Gianluca Gerard
 Licensing   : This code is distributed under the GNU LGPL license.
 Description : Solves the steady state heat equation with Gauss-Seidel iterations.
 Reference(s):
 ============================================================================
 */

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>

#include "HeatedPlate2D.h"

double GaussSeidelV1(double **u, int m, int n,
		double eps, int maxit,
		int iterations_print, int* iterations) {
	int i, j;
	double v;

	double diff = eps;

	while (eps <= diff && (*iterations) < maxit) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 */
		diff = 0.0;

		for (i = 1; i < m - 1; i++) {
			for (j = 1; j < n - 1; j ++) {
				v = (u[i - 1][j] + u[i + 1][j] + u[i][j - 1]
													  + u[i][j + 1]) / 4.0;
				if ( diff < fabs(v - u[i][j]))
					diff = fabs(v - u[i][j]);
				u[i][j] = v;
			}
		}

		(*iterations)++;
#ifdef __VERBOSE
		if (*iterations == iterations_print) {
			printf("  %8d  %f\n", *iterations, diff);
			iterations_print = 2 * iterations_print;
		}
#endif
	}
	return diff;
}

double GaussSeidelV1Err(double **u, int m, int n,
		double eps, int maxit,
		int iterations_print, int* iterations) {
	int i, j;
	double v;

	double error = 10.0*eps;

	while (error >= eps  && (*iterations) < maxit) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 */
		error = 0.0;

		for (i = 1; i < m - 1; i++) {
			for (j = 1; j < n - 1; j ++) {
				v = (u[i - 1][j] + u[i + 1][j] + u[i][j - 1]
													  + u[i][j + 1]) / 4.0;
				error += (v - u[i][j])*(v - u[i][j]);
				u[i][j] = v;
			}
		}

		(*iterations)++;
		error = sqrt(error)/(m*n);

#ifdef __VERBOSE
		if (*iterations == iterations_print) {
			printf("  %8d  %f\n", *iterations, error);
			iterations_print = 2 * iterations_print;
		}
#endif
	}
	return error;
}

double GaussSeidel(double **u, int m, int n, int sqrerr,
		double eps, int maxit,
		int iterations_print, int* iterations, double* wtime) {
	double err;

	/*
     * iterate until the  error is less than the tolerance.
	 */
	*iterations = 0;

#ifdef __VERBOSE
	printf ( "\n" );
	printf ( " Iteration  Change\n" );
	printf ( "\n" );
#endif

	*wtime = omp_get_wtime();
	if (sqrerr == 0)
		err = GaussSeidelV1(u, m, n, eps, maxit, iterations_print, iterations);
	else
		err = GaussSeidelV1Err(u, m, n, eps, maxit, iterations_print, iterations);
	*wtime = omp_get_wtime() - *wtime;
	return err;
}
