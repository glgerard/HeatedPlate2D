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

double JacobiOmpLoopV1(double** up, double** wp, int m, int n,
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
#pragma omp parallel private (j, my_diff)
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
			if (diff < my_diff)
#pragma omp atomic write
				diff = my_diff;
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

double JacobiOmpLoopV1err(double** up, double** wp, int m, int n,
		double eps, int iterations_print, int* iterations) {
	int i;
	int j;
	double **t;
	double error;

	/*
     * iterate until the  error is less than the tolerance.
	 */
	error = 10.0*eps;
	while (error >= eps) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 * The new solution W is the average of north, south, east and west neighbors.
		 */
#pragma omp parallel for private(j) reduction(+:error)
			for (i = 1; i < m - 1; i++) {
				for (j = 1; j < n - 1; j++) {
					wp[i][j] = (up[i - 1][j] + up[i + 1][j] + up[i][j - 1]
																	+ up[i][j + 1]) / 4.0;
					error += (wp[i][j] - up[i][j])*(wp[i][j] - up[i][j]);
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
		error = sqrt(error)/(m*n);
		if (*iterations == iterations_print) {
			printf("  %8d  %f\n", *iterations, error);
			iterations_print = 2 * iterations_print;
		}
	}
	return error;
}

double JacobiOmpLoopV2(double** up, double** wp, int m, int n,
		double eps, int iterations_print, int* iterations) {
	int i;
	int j;
	double **t;
	double diff;
	double my_diff;

	my_diff = eps;

	/*
     * iterate until the  error is less than the tolerance.
	 */
#pragma omp parallel private (j) firstprivate(my_diff)
	{
		while (eps <= my_diff) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 * The new solution W is the average of north, south, east and west neighbors.
		 */
			diff = 0.0;
			my_diff = 0.0;
#pragma omp for schedule(static)
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

			if (diff < my_diff)
#pragma omp atomic write
				diff = my_diff;

#pragma omp single
			{
				/*
				 * Swap wp and up so that at the next iteration up points to the new
				 * solution.
				 */
				t = up;
				up = wp;
				wp = t;
			} /* end single, implicit barrier */

#ifdef __DEBUG
			printf("tid = %d, diff = %g\n", omp_get_thread_num(), diff);
#endif

			my_diff = diff;

#pragma omp single
			{
				(*iterations)++;
				if (*iterations == iterations_print) {
					printf("  %8d  %f\n", *iterations, diff);
					iterations_print = 2 * iterations_print;
				}
			} /* end single, implicit barrier */
		} /* end while */
	} /* end parallel */
	return diff;
}

double JacobiOmpLoopV2err(double** up, double** wp, int m, int n,
		double eps, int iterations_print, int* iterations) {
	int i;
	int j;
	double **t;
	double error;

	error = 10.0*eps;

	/*
     * iterate until the  error is less than the tolerance.
	 */
#pragma omp parallel private (j)
	{
		while (error >= eps) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 * The new solution W is the average of north, south, east and west neighbors.
		 */
#pragma omp for reduction(+:error)
			for (i = 1; i < m - 1; i++) {
				for (j = 1; j < n - 1; j++) {
					wp[i][j] = (up[i - 1][j] + up[i + 1][j] + up[i][j - 1]
																+ up[i][j + 1]) / 4.0;
					error += (wp[i][j] - up[i][j])*(wp[i][j] - up[i][j]);
				}
			}

#ifdef __DEBUG
			printf("tid = %d, my_diff = %g\n", omp_get_thread_num(), error);
#endif

#pragma omp single
			{
				error = sqrt(error)/(m*n);
				/*
				 * Swap wp and up so that at the next iteration up points to the new
				 * solution.
				 */
				t = up;
				up = wp;
				wp = t;

				(*iterations)++;
				if (*iterations == iterations_print) {
					printf("  %8d  %f\n", *iterations, error);
					iterations_print = 2 * iterations_print;
				}
			} /* end single, implicit barrier */
#ifdef __DEBUG
			printf("tid = %d, diff = %g\n", omp_get_thread_num(), error);
#endif
		} /* end while */
	} /* end parallel */
	return error;
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

	max_diff = eps;

	/*
     * iterate until the  error is less than the tolerance.
	 */
#pragma omp parallel private (j, tid)
	{
		tid = omp_get_thread_num();

		while (eps <= max_diff) {
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

#pragma omp single
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
			} /* end single, implicit barrier */

#ifdef __DEBUG
			printf("tid = %d, diff = %g\n", tid, max_diff);
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

	free(diff);

	return max_diff;
}

double JacobiOmpLoopV3err(double** up, double** wp, int m, int n,
		double eps, int iterations_print, int* iterations) {
	int i;
	int j;
	double **t;
	double *my_error;
	double error;
	int tid, nthreads;

	nthreads=omp_get_max_threads();
	my_error = malloc(sizeof(double)*nthreads);

	error = 10.0*eps;

	/*
     * iterate until the  error is less than the tolerance.
	 */
#pragma omp parallel private (j, tid)
	{
		tid = omp_get_thread_num();

		while (error >= eps) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 * The new solution W is the average of north, south, east and west neighbors.
		 */

			my_error[tid] = 0.0;
#pragma omp for
			for (i = 1; i < m - 1; i++) {
				for (j = 1; j < n - 1; j++) {
					wp[i][j] = (up[i - 1][j] + up[i + 1][j] + up[i][j - 1]
																+ up[i][j + 1]) / 4.0;
					my_error[tid] += (wp[i][j] - up[i][j])*(wp[i][j] - up[i][j]);
				}
			}

#ifdef __DEBUG
			printf("tid = %d, diff = %g\n", tid, my_error[tid]);
#endif

#pragma omp single
			{
				/*
				 * Swap wp and up so that at the next iteration up points to the new
				 * solution.
				 */
				t = up;
				up = wp;
				wp = t;
				for(j = 0; j < nthreads; j++)
						error += my_error[j];
				error = sqrt(error)/(n*m);
			} /* end single, implicit barrier */

#ifdef __DEBUG
//			printf("tid = %d, diff = %g\n", tid, my_error[tid]);
			printf("tid = %d, diff = %g\n", tid, error);
#endif

#pragma omp master
			{
				(*iterations)++;
				if (*iterations == iterations_print) {
					printf("  %8d  %f\n", *iterations, error);
					iterations_print = 2 * iterations_print;
				}
			} /* end master */
		} /* end while */
	} /* end parallel */

	free(my_error);

	return error;
}


double JacobiOmpLoopV4(double** up, double** wp, int m, int n,
		double eps, int iterations_print, int* iterations) {
	int i;
	int j;
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
#pragma omp parallel private (j, my_diff)
		{
			my_diff = 0.0;
#pragma omp for
			for (i = 1; i < m - 1; i++) {
				for (j = 1; j < n - 1; j++) {
					wp[i][j] = (up[i - 1][j] + up[i + 1][j] + up[i][j - 1]
																	+ up[i][j + 1]) / 4.0;
				}
			}
#pragma omp for
			for (i = 1; i < m - 1; i++) {
				for (j = 1; j < n - 1; j++) {
					up[i][j] = (wp[i - 1][j] + wp[i + 1][j] + wp[i][j - 1]
																	+ wp[i][j + 1]) / 4.0;
					if ( my_diff < fabs(up[i][j] - wp[i][j]) )
						my_diff = fabs(up[i][j] - wp[i][j]);
				}
			}
			if ( diff < my_diff )
#pragma omp atomic write
				diff = my_diff;
		}
		*iterations += 2;
		if (*iterations >> 1 == iterations_print) {
			printf("  %8d  %f\n", *iterations, diff);
			iterations_print = 2 * iterations_print;
		}
	}
	return diff;
}

double JacobiOmpLoopV4err(double** up, double** wp, int m, int n,
		double eps, int iterations_print, int* iterations) {
	int i;
	int j;
	double error;

	/*
     * iterate until the  error is less than the tolerance.
	 */
	error = 10.0*eps;
	while (error >= eps) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 * The new solution W is the average of north, south, east and west neighbors.
		 */
#pragma omp parallel private(j)
		{
#pragma omp for
			for (i = 1; i < m - 1; i++) {
				for (j = 1; j < n - 1; j++) {
					wp[i][j] = (up[i - 1][j] + up[i + 1][j] + up[i][j - 1]
																	+ up[i][j + 1]) / 4.0;
				}
			}
#pragma omp for reduction(+:error)
			for (i = 1; i < m - 1; i++) {
				for (j = 1; j < n - 1; j++) {
					up[i][j] = (wp[i - 1][j] + wp[i + 1][j] + wp[i][j - 1]
																	+ wp[i][j + 1]) / 4.0;
					error += (up[i][j] - wp[i][j])*(up[i][j] - wp[i][j]);
				}
			}
		} /* end parallel */
		*iterations +=2;
		error = sqrt(error)/(m*n);
		if (*iterations >> 1 == iterations_print) {
			printf("  %8d  %f\n", *iterations, error);
			iterations_print = 2 * iterations_print;
		}
	}
	return error;
}

double Jacobi(double **u, int m, int n, double eps, int omp, int sqrerr, int version,
		int iterations_print, int* iterations, double* wtime) {
	double **w;
	double err;

	w = InitGrid(m, n, 0);

	*iterations = 0;

	printf ( "\n" );
	printf ( " Iteration  Change\n" );
	printf ( "\n" );

	*wtime = omp_get_wtime();
	if (sqrerr == 0)
		if (omp == 0)
			err = JacobiSeqLoop(u, w, m, n, eps, iterations_print, iterations);
		else
			switch(version) {
			case 1:
				err = JacobiOmpLoopV1(u, w, m, n, eps, iterations_print, iterations);
				break;
			case 2:
				err = JacobiOmpLoopV2(u, w, m, n, eps, iterations_print, iterations);
				break;
			case 3:
				err = JacobiOmpLoopV3(u, w, m, n, eps, iterations_print, iterations);
				break;
			case 4:
				err = JacobiOmpLoopV4(u, w, m, n, eps, iterations_print, iterations);
				break;
			default:
				err = -1.0;
				break;
			}
	else
		if (omp == 0)
			err = -1.0;
		else
			switch(version) {
			case 1:
				err = JacobiOmpLoopV1err(u, w, m, n, eps, iterations_print, iterations);
				break;
			case 2:
				err = JacobiOmpLoopV2err(u, w, m, n, eps, iterations_print, iterations);
				break;
			case 3:
				err = JacobiOmpLoopV3err(u, w, m, n, eps, iterations_print, iterations);
				break;
			case 4:
				err = JacobiOmpLoopV4err(u, w, m, n, eps, iterations_print, iterations);
				break;
			default:
				err = -1.0;
				break;
			}
	*wtime = omp_get_wtime() - *wtime;

	FreeGrid(w, m);

	return err;
}
