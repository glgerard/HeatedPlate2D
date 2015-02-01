/*
 ============================================================================
 Name        : Jacobi.c
 Author      : Gianluca Gerard
 Version     : v0.2
 Copyright   : Copyright (2015) Gianluca Gerard
 Licensing   : This code is distributed under the GNU LGPL license.
 Description : Solves the steady state heat equation with Jacobi iterations.
 Reference(s):
   http://people.sc.fsu.edu/~jburkardt/c_src/heated_plate_openmp/heated_plate_openmp.html
 ============================================================================
 */

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>

#include "HeatedPlate2D.h"

double JacobiSeqLoop(double** up, double** wp, int m, int n,
		double eps, int iterations_print, int* iterations) {
	int i;
	int j;
	double **t;
	double diff;

	diff = eps;

	/*
     * iterate until the  error is less than the tolerance.
	 */
	while (eps <= diff) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 * The new solution W is the average of north, south, east and west neighbors.
		 */
		diff = 0.0;
		for (i = 1; i < m - 1; i++) {
			for (j = 1; j < n - 1; j++) {
				wp[i][j] = (up[i - 1][j] + up[i + 1][j] + up[i][j - 1]
						+ up[i][j + 1]) / 4.0;
				if ( diff < fabs(wp[i][j] - up[i][j]) )
					diff = fabs(wp[i][j] - up[i][j]);
			}
		}
		/*
		 * Swap wp and up so that at the next iteration up points to the new
		 * solution.
		 */
		t = up;
		up = wp;
		wp = t;
		(*iterations)++;
		if (*iterations == iterations_print) {
			printf("  %8d  %f\n", *iterations, diff);
			iterations_print = 2 * iterations_print;
		}
	}
	return diff;
}

double JacobiOmpLoop(double** up, double** wp, int m, int n,
		double eps, int iterations_print, int* iterations) {
	int i;
	int j;
	double **t;
	double diff, my_diff;

	diff = eps;

	/*
     * iterate until the  error is less than the tolerance.
	 */
	while (eps <= diff) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 * The new solution W is the average of north, south, east and west neighbors.
		 */
		diff=0.0;
#pragma omp parallel shared private (j, my_diff)
		{
			my_diff = 0.0;
#pragma omp for
			for (i = 1; i < m - 1; i++) {
				for (j = 1; j < n - 1; j++) {
					wp[i][j] = (up[i - 1][j] + up[i + 1][j] + up[i][j - 1]
																	+ up[i][j + 1]) / 4.0;
					if ( my_diff < fabs(wp[i][j] - up[i][j]) )
						my_diff = fabs(wp[i][j] - up[i][j]);
				}
			}
#pragma omp critical
			{
				if ( diff < my_diff )
					diff = my_diff;
			}
		}
		/*
		 * Swap wp and up so that at the next iteration up points to the new
		 * solution.
		 */
		t = up;
		up = wp;
		wp = t;
		(*iterations)++;
		if (*iterations == iterations_print) {
			printf("  %8d  %f\n", *iterations, diff);
			iterations_print = 2 * iterations_print;
		}
	}
	return diff;
}

double JacobiOmpLoopV2(double** up, double** wp, int m, int n,
		double eps, int iterations_print, int* iterations) {
	int i;
	int j;
	double **t;
	double diff;
	double my_diff;

	diff = eps;

	/*
     * iterate until the  error is less than the tolerance.
	 */
#pragma omp parallel private (j, my_diff)
	{
		while (eps <= diff) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 * The new solution W is the average of north, south, east and west neighbors.
		 */
			my_diff = 0.0;
#pragma omp for
			for (i = 1; i < m - 1; i++) {
				for (j = 1; j < n - 1; j++) {
					wp[i][j] = (up[i - 1][j] + up[i + 1][j] + up[i][j - 1]
																+ up[i][j + 1]) / 4.0;
					if (my_diff < fabs(wp[i][j] - up[i][j]))
						my_diff = fabs(wp[i][j] - up[i][j]);
				}
			}

#ifdef __DEBUG
			printf("tid = %d, my_diff = %g\n", omp_get_thread_num(), my_diff);
#endif

#pragma omp master
			{
				diff = my_diff;
				/*
				 * Swap wp and up so that at the next iteration up points to the new
				 * solution.
				 */
				t = up;
				up = wp;
				wp = t;
			} /* end master */

#pragma omp barrier

#pragma omp critical
			{
				if(my_diff > diff)
					diff = my_diff;
			}

#ifdef __DEBUG
			printf("tid = %d, diff = %g\n", omp_get_thread_num(), diff);
#endif

#pragma omp master
			{
				(*iterations)++;
				if (*iterations == iterations_print) {
					printf("  %8d  %f\n", *iterations, diff);
					iterations_print = 2 * iterations_print;
				}
			} /* end master */
		} /* end while */
	} /* end parallel */
	return diff;
}

double JacobiOmpLoopV3(double** up, double** wp, int m, int n,
		double eps, int iterations_print, int* iterations) {
	int i;
	int j;
	double **t;
	double *diff;
	double max_diff;
	int tid, nthreads;

	nthreads=omp_get_max_threads();
	diff = malloc(sizeof(double)*nthreads);
	for (tid=0; tid<nthreads; tid++)
		diff[tid] = eps;

	/*
     * iterate until the  error is less than the tolerance.
	 */
#pragma omp parallel private (j, tid)
	{
		tid = omp_get_thread_num();

		while (eps <= diff[tid]) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 * The new solution W is the average of north, south, east and west neighbors.
		 */

			diff[tid] = 0.0;
#pragma omp for
			for (i = 1; i < m - 1; i++) {
				for (j = 1; j < n - 1; j++) {
					wp[i][j] = (up[i - 1][j] + up[i + 1][j] + up[i][j - 1]
																+ up[i][j + 1]) / 4.0;
					if (diff[tid] < fabs(wp[i][j] - up[i][j]))
						diff[tid] = fabs(wp[i][j] - up[i][j]);
				}
			}

#ifdef __DEBUG
			printf("tid = %d, diff = %g\n", tid, diff[tid]);
#endif

#pragma omp master
			{
				/*
				 * Swap wp and up so that at the next iteration up points to the new
				 * solution.
				 */
				t = up;
				up = wp;
				wp = t;

				max_diff = 0.0;
				for(j = 0; j < nthreads; j++)
					if (diff[j] > max_diff)
						max_diff = diff[j];

			} /* end master */

#pragma omp barrier
			diff[tid] = max_diff;

#ifdef __DEBUG
			printf("tid = %d, diff = %g\n", tid, diff[tid]);
#endif

#pragma omp master
			{
				(*iterations)++;
				if (*iterations == iterations_print) {
					printf("  %8d  %f\n", *iterations, max_diff);
					iterations_print = 2 * iterations_print;
				}
			} /* end master */
		} /* end while */
	} /* end parallel */
	return max_diff;
}

double Jacobi(double **u, int m, int n, double eps, int omp,
		int iterations_print, int* iterations, double* wtime) {
	double **w;
	double err;

	w = InitGrid(m, n, 0);

	*iterations = 0;

	printf ( "\n" );
	printf ( " Iteration  Change\n" );
	printf ( "\n" );

	*wtime = omp_get_wtime();
	if (omp == 0)
		err = JacobiSeqLoop(u, w, m, n, eps, iterations_print, iterations);
	else
		err = JacobiOmpLoopV2(u, w, m, n, eps, iterations_print, iterations);
	*wtime = omp_get_wtime() - *wtime;

	FreeGrid(w, m);

	return err;
}
