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

double GaussSeidel(double **u, int m, int n, double eps,
		int iterations_print, int* iterations, double* wtime) {
	int i, j;
	double diff;
	double v;

	diff = eps;

	/*
     * iterate until the  error is less than the tolerance.
	 */
	*iterations = 0;

	printf ( "\n" );
	printf ( " Iteration  Change\n" );
	printf ( "\n" );

	*wtime = omp_get_wtime();
	while (eps <= diff) {
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
		if (*iterations == iterations_print) {
			printf("  %8d  %f\n", *iterations, diff);
			iterations_print = 2 * iterations_print;
		}
	}
	*wtime = omp_get_wtime() - *wtime;
	return diff;
}
