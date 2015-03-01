/*
 ============================================================================
 Name        : RedBlack_GaussSeidel.c
 Author      : Gianluca Gerard
 Version     : v0.2
 Copyright   : Copyright (2015) Gianluca Gerard
 Licensing   : This code is distributed under the GNU LGPL license.
 Description : Solves the steady state heat equation with Red Black
               Gauss-Seidel iterations.
 Reference(s):
 ============================================================================
 */

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>
# include "HeatedPlate2D.h"

double RB_GS_SeqLoop(double** black, double **red, int m, int n,
		double eps, int maxit,
		int iterations_print, int* iterations) {
	int i, j;
	double v;

	double diff = eps;

	/*
     * Iterate until the  error is less than the tolerance or
     * we reach the maximum number of iterations.
	 */
	while (eps <= diff && *iterations < maxit ) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 */
		diff = 0.0;

		/*
		 * Black sweep
		 */
		for (i = 1; i < m - 1; i++)
			for (j = i%2; j < n/2 - (i+1)%2; j++) {
				v = 0.25*(red[i - 1][j] + red[i + 1][j] +
						red[i][j - i%2] + red[i][j + (i+1)%2]);
				if (diff < fabs(v - black[i][j]))
					diff = fabs(v - black[i][j]);
				black[i][j] = v;
			}

		/*
		 * Red sweep
		 */
		for (i = 1; i < m - 1; i++)
			for (j = (i+1)%2; j < n/2 - i%2; j++) {
				v = 0.25*(black[i - 1][j] + black[i + 1][j] +
						black[i][j - (i+1)%2] + black[i][j + i%2]);
				if (diff < fabs(v - red[i][j]))
					diff = fabs(v - red[i][j]);
				red[i][j] = v;
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

double RB_GS_SeqLoopErr(double** black, double **red, int m, int n,
		double eps, int maxit,
		int iterations_print, int* iterations) {
	int i, j;
	double v;

	double error = 10.0*eps;

	/*
     * Iterate until the  error is less than the tolerance or
     * we reach the maximum number of iterations.
	 */
	while (error >= eps && *iterations < maxit ) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 */

		/*
		 * Black sweep
		 */
		for (i = 1; i < m - 1; i++)
			for (j = i%2; j < n/2 - (i+1)%2; j++) {
				v = 0.25*(red[i - 1][j] + red[i + 1][j] +
						red[i][j - i%2] + red[i][j + (i+1)%2]);
				error += (v - black[i][j])*(v - black[i][j]);
				black[i][j] = v;
			}

		/*
		 * Red sweep
		 */
		for (i = 1; i < m - 1; i++)
			for (j = (i+1)%2; j < n/2 - i%2; j++) {
				v = 0.25*(black[i - 1][j] + black[i + 1][j] +
						black[i][j - (i+1)%2] + black[i][j + i%2]);
				error += (v - red[i][j])*(v - red[i][j]);
				red[i][j] = v;
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

double RB_GS_OmpLoopV1(double** black, double** red, int m, int n,
		double eps, int maxit,
		int iterations_print, int* iterations) {
	int i, j;
	double v;
	double my_diff, my_diffR, my_diffB;

	double diff = eps;

	/*
     * Iterate until the  error is less than the tolerance or
     * we reach the maximum number of iterations.
	 */
	while (eps <= diff && *iterations < maxit) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 */
		diff = 0.0;

#pragma omp parallel private(j, v, my_diff, my_diffR, my_diffB)
		{
			my_diffB = 0.0;
			/*
			 * Black sweep
			 */
#pragma omp for schedule(OMP_SCHED)
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = 0.25*(red[i - 1][j] + red[i + 1][j] +
							red[i][j - i%2] + red[i][j + (i+1)%2]);
					if (my_diffB < fabs(v - black[i][j]))
						my_diffB = fabs(v - black[i][j]);
					black[i][j] = v;
				}
// implicit barrier

			my_diffR = 0.0;
			/*
			 * Red sweep
			 */
#pragma omp for schedule(OMP_SCHED) nowait
 			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = 0.25*(black[i - 1][j] + black[i + 1][j] +
							black[i][j - (i+1)%2] + black[i][j + i%2]);
					if (my_diffR < fabs(v - red[i][j]))
						my_diffR = fabs(v - red[i][j]);
					red[i][j] = v;
				}

			my_diff = my_diffB < my_diffR ? my_diffR : my_diffB;
#pragma omp barrier

			if (diff < my_diff)
#pragma omp atomic write
				diff = my_diff;
		} /* end parallel */

		(*iterations)++;
#ifdef __VERBOSE
		if (*iterations == iterations_print) {
			printf("  %8d  %f\n", *iterations, diff);
			iterations_print = 2 * iterations_print;
		}
#endif
	} /* end while */
	return diff;
}

double RB_GS_OmpLoopV1err(double** black, double** red, int m, int n,
		double eps, int maxit,
		int iterations_print, int* iterations) {
	int i, j;
	double v;

	double error = 10.0*eps;

	/*
     * Iterate until the  error is less than the tolerance or
     * we reach the maximum number of iterations.
	 */
	while (error >= eps && *iterations < maxit) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 */
#pragma omp parallel private(j, v)
		{
			/*
			 * Black sweep
			 */
#pragma omp for schedule(OMP_SCHED) reduction(+:error)
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = 0.25*(red[i - 1][j] + red[i + 1][j] +
							red[i][j - i%2] + red[i][j + (i+1)%2]);
					error += (v - black[i][j])*(v - black[i][j]);
					black[i][j] = v;
				}
// implicit barrier

			/*
			 * Red sweep
			 */
#pragma omp for schedule(OMP_SCHED) reduction(+:error) nowait
			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = 0.25*(black[i - 1][j] + black[i + 1][j] +
							black[i][j - (i+1)%2] + black[i][j + i%2]);
					error += (v - red[i][j])*(v - red[i][j]);
					red[i][j] = v;
				}
		} /* end parallel */

		error = sqrt(error)/(m*n);
		(*iterations)++;

#ifdef __VERBOSE
		if (*iterations == iterations_print) {
			printf("  %8d  %f\n", *iterations, error);
			iterations_print = 2 * iterations_print;
		}
#endif
	} /* end while */
	return error;
}

double RB_GS_OmpLoopV2(double** black, double** red, int m, int n,
		double eps, int maxit,
		int iterations_print, int* iterations) {
	int i, j;
	double v;
	double diff, my_diffR, my_diffB;

	int my_iterations = *iterations;
	double my_diff = eps;

	/*
     * Iterate until the  error is less than the tolerance or
     * we reach the maximum number of iterations.
	 */
#pragma omp parallel private(j, v) firstprivate(my_diff, my_iterations)
	{
		while (eps <= my_diff && my_iterations < maxit) {
			/*
			 * Determine the new estimate of the solution at the interior points.
			 */
			diff = 0.0;
			my_diffB = 0.0;
			/*
			 * Black sweep
			 */
#pragma omp for schedule(OMP_SCHED)
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = 0.25*(red[i - 1][j] + red[i + 1][j] +
							red[i][j - i%2] + red[i][j + (i+1)%2]);
					if (my_diffB < fabs(v - black[i][j]))
						my_diffB = fabs(v - black[i][j]);
					black[i][j] = v;
				}
// implicit barrier

			my_diffR = 0.0;
			/*
			 * Red sweep
			 */
#pragma omp for schedule(OMP_SCHED) nowait
			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = 0.25*(black[i - 1][j] + black[i + 1][j] +
							black[i][j - (i+1)%2] + black[i][j + i%2]);
					if (my_diffR < fabs(v - red[i][j]))
						my_diffR = fabs(v - red[i][j]);
					red[i][j] = v;
				}

			my_diff = my_diffB < my_diffR ? my_diffR : my_diffB;
#pragma omp barrier

			if (diff < my_diff)
#pragma omp critical (diff_update)
				diff = my_diff;

			my_iterations++;
			my_diff = diff;

#ifdef __VERBOSE
#pragma omp single
			{
				if (my_iterations == iterations_print) {
					printf("  %8d  %f\n", my_iterations, diff);
					iterations_print = 2 * iterations_print;
				}
			} /* end single, implicit barrier */
#else
#pragma omp barrier
#endif
		} /* end while */
#pragma omp master
		*iterations = my_iterations;
	} /* end parallel */
	return diff;
}

double RB_GS_OmpLoopV2err(double** black, double** red, int m, int n,
		double eps, int maxit,
		int iterations_print, int* iterations) {
	int i, j;
	double v;

	int my_iterations = *iterations;
	double error= 10.0*eps;

#pragma omp parallel private(j, v) firstprivate(my_iterations)
	{
		/*
	     * Iterate until the  error is less than the tolerance or
	     * we reach the maximum number of iterations.
		 */
		while (error >= eps && my_iterations < maxit) {
			/*
			 * Determine the new estimate of the solution.
			 */
			/*
			 * Black sweep
			 */
#pragma omp for schedule(OMP_SCHED) reduction(+:error)
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = 0.25*(red[i - 1][j] + red[i + 1][j] +
							red[i][j - i%2] + red[i][j + (i+1)%2]);
					error += (v - black[i][j])*(v - black[i][j]);
					black[i][j] = v;
				}
// implicit barrier

			/*
			 * Red sweep
			 */
#pragma omp for schedule(OMP_SCHED) reduction(+:error)
			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = 0.25*(black[i - 1][j] + black[i + 1][j] +
							black[i][j - (i+1)%2] + black[i][j + i%2]);
					error += (v - red[i][j])*(v - red[i][j]);
					red[i][j] = v;
				}
// implicit barrier

			my_iterations++;

#pragma omp single
			{
				error = sqrt(error)/(m*n);
#ifdef __VERBOSE
				if (my_iterations == iterations_print) {
					printf("  %8d  %f\n", my_iterations, error);
					iterations_print = 2 * iterations_print;
				}
#endif
			} /* end single, implicit barrier */
		} /* end while */
#pragma omp master
		*iterations = my_iterations;
	} /* end parallel */

	return error;
}

double RB_GS_OmpLoopV3(double** black, double** red, int m, int n,
		double eps, int maxit,
		int iterations_print, int* iterations) {
	int i, j;
	double v;
	double *my_diff, *my_diffR, *my_diffB;
	int tid, nthreads;
	int my_iterations = *iterations;

	double diff = eps;

	nthreads=omp_get_max_threads();
	my_diff = malloc(sizeof(double)*nthreads);
	my_diffR = malloc(sizeof(double)*nthreads);
	my_diffB = malloc(sizeof(double)*nthreads);

#pragma omp parallel private(j, v, tid) firstprivate(my_iterations)
	{
		tid = omp_get_thread_num();

		/*
	     * Iterate until the  error is less than the tolerance or
	     * we reach the maximum number of iterations.
		 */
		while (eps <= diff && my_iterations < maxit) {
			/*
			 * Determine the new estimate of the solution at the interior points.
			 */
			/*
			 * Black sweep
			 */
			my_diffB[tid] = 0.0;
#pragma omp for schedule(OMP_SCHED)
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = 0.25*(red[i - 1][j] + red[i + 1][j] +
							red[i][j - i%2] + red[i][j + (i+1)%2]);
					if (my_diffB[tid] < fabs(v - black[i][j]))
						my_diffB[tid] = fabs(v - black[i][j]);
					black[i][j] = v;
				}
// implicit barrier

			/*
			 * Red sweep
			 */
			my_diffR[tid] = 0.0;
#pragma omp for schedule(OMP_SCHED) nowait
			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = 0.25*(black[i - 1][j] + black[i + 1][j] +
							black[i][j - (i+1)%2] + black[i][j + i%2]);
					if (my_diffR[tid] < fabs(v - red[i][j]))
						my_diffR[tid] = fabs(v - red[i][j]);
					red[i][j] = v;
				}

			my_diff[tid] = my_diffR[tid] < my_diffB[tid] ?
					my_diffB[tid] : my_diffR[tid];
#pragma omp barrier
			my_iterations++;

#pragma omp single
			{
				diff = 0.0;
				for(j = 0; j < nthreads; j++)
					if (my_diff[j] > diff)
						diff = my_diff[j];
			}

#ifdef __VERBOSE
#pragma omp master
			{
				if (my_iterations == iterations_print) {
					printf("  %8d  %f\n", my_iterations, diff);
					iterations_print = 2 * iterations_print;
				}
			}
#endif
		} /* end while */
#pragma omp master
		*iterations = my_iterations;
	} /* end parallel */

	free(my_diff);

	return diff;
}

double RB_GS_OmpLoopV3err(double** black, double** red, int m, int n,
		double eps, int maxit,
		int iterations_print, int* iterations) {
	int i, j;
	double v;
	double *my_error;

	int tid, nthreads;

	int my_iterations = *iterations;
	double error = 10.0*eps;

	nthreads=omp_get_max_threads();
	my_error = malloc(sizeof(double)*nthreads);

#pragma omp parallel private(j, v, tid) firstprivate(my_iterations)
	{
		tid = omp_get_thread_num();

		/*
	     * Iterate until the  error is less than the tolerance or
	     * we reach the maximum number of iterations.
		 */
		while (error >= eps && my_iterations < maxit) {
			/*
			 * Determine the new estimate of the solution at the interior points.
			 */
			my_error[tid] = 0.0;
			/*
			 * Black sweep
			 */
#pragma omp for schedule(OMP_SCHED)
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = 0.25*(red[i - 1][j] + red[i + 1][j] +
							red[i][j - i%2] + red[i][j + (i+1)%2]);
					my_error[tid] += (v - black[i][j])*(v - black[i][j]);
					black[i][j] = v;
				}
			/*
			 * Red sweep
			 */
#pragma omp for schedule(OMP_SCHED)
			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = 0.25*(black[i - 1][j] + black[i + 1][j] +
							black[i][j - (i+1)%2] + black[i][j + i%2]);
					my_error[tid] += (v - red[i][j])*(v - red[i][j]);
					red[i][j] = v;
				}

#pragma omp single
			{
				for(j = 0; j < nthreads; j++)
						error += my_error[j];
				error = sqrt(error)/(n*m);
			}

			my_iterations++;
#ifdef __VERBOSE
#pragma omp master
			{
				if (*iterations == iterations_print) {
					printf("  %8d  %f\n", *iterations, error);
					iterations_print = 2 * iterations_print;
				}
			}
#endif
		} /* end while */
#pragma omp master
		*iterations = my_iterations;
	} /* end parallel */

	free(my_error);

	return error;
}

double RB_GS_OmpLoopV4(double** black, double** red, int m, int n,
		double eps, int maxit,
		int iterations_print, int* iterations) {
	int i, j;
	double v;
	double my_diff, my_diffR, my_diffB;

	double diff = eps;

	/*
     * Iterate until the  error is less than the tolerance or
     * we reach the maximum number of iterations.
	 */
	while (eps <= diff && *iterations < maxit) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 */
		diff = 0.0;

#pragma omp parallel private(j, v, my_diff, my_diffR, my_diffB)
		{
			/*
			 * Black sweep
			 */
#pragma omp for schedule(OMP_SCHED)
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = 0.25*(red[i - 1][j] + red[i + 1][j] +
							red[i][j - i%2] + red[i][j + (i+1)%2]);
					black[i][j] = v;
				}
// implicit barrier

			/*
			 * Red sweep
			 */
#pragma omp for schedule(OMP_SCHED)
			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = 0.25*(black[i - 1][j] + black[i + 1][j] +
							black[i][j - (i+1)%2] + black[i][j + i%2]);
					red[i][j] = v;
				}
// implicit barrier

			/*
			 * Second iteration
			 * Black sweep
			 */
			my_diffB = 0.0;
#pragma omp for schedule(OMP_SCHED)
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = 0.25*(red[i - 1][j] + red[i + 1][j] +
							red[i][j - i%2] + red[i][j + (i+1)%2]);
					if (my_diffB < fabs(v - black[i][j]))
						my_diffB = fabs(v - black[i][j]);
					black[i][j] = v;
				}
// implicit barrier

			/*
			 * Second iteration
			 * Red sweep
			 */
			my_diffR = 0.0;
#pragma omp for schedule(OMP_SCHED) nowait
			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = 0.25*(black[i - 1][j] + black[i + 1][j] +
							black[i][j - (i+1)%2] + black[i][j + i%2]);
					if (my_diffR < fabs(v - red[i][j]))
						my_diffR = fabs(v - red[i][j]);
					red[i][j] = v;
				}

			my_diff = my_diffR < my_diffB ? my_diffB : my_diffR;
#pragma omp barrier

			if (diff < my_diff)
#pragma omp atomic write
				diff = my_diff;
		} /* end parallel */

		*iterations += 2;
#ifdef __VERBOSE
		if (*iterations >> 1 == iterations_print) {
			printf("  %8d  %f\n", *iterations, diff);
			iterations_print = 2 * iterations_print;
		}
#endif
	} /* end while */
	return diff;
}

double RB_GS_OmpLoopV4err(double** black, double** red, int m, int n,
		double eps, int maxit,
		int iterations_print, int* iterations) {
	int i, j;
	double v;

	double error = 10.0*eps;

	/*
     * Iterate until the  error is less than the tolerance or
     * we reach the maximum number of iterations.
	 */
	while (error >= eps && *iterations < maxit) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 */
#pragma omp parallel private(j, v)
		{
			/*
			 * Black sweep
			 */
#pragma omp for schedule(OMP_SCHED) reduction(+:error)
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = 0.25*(red[i - 1][j] + red[i + 1][j] +
							red[i][j - i%2] + red[i][j + (i+1)%2]);
					black[i][j] = v;
				}
//implicit barrier

			/*
			 * Red sweep
			 */
#pragma omp for schedule(OMP_SCHED) reduction(+:error)
			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = 0.25*(black[i - 1][j] + black[i + 1][j] +
							black[i][j - (i+1)%2] + black[i][j + i%2]);
					red[i][j] = v;
				}
// implicit barrier

			/*
			 * Second iteration
			 * Black sweep
			 */
#pragma omp for schedule(OMP_SCHED) reduction(+:error)
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = 0.25*(red[i - 1][j] + red[i + 1][j] +
							red[i][j - i%2] + red[i][j + (i+1)%2]);
					error += (v - black[i][j])*(v - black[i][j]);
					black[i][j] = v;
				}
// implicit barrier

			/*
			 * Second iteration
			 * Red sweep
			 */
#pragma omp for schedule(OMP_SCHED) reduction(+:error) nowait
			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = 0.25*(black[i - 1][j] + black[i + 1][j] +
							black[i][j - (i+1)%2] + black[i][j + i%2]);
					error += (v - red[i][j])*(v - red[i][j]);
					red[i][j] = v;
				}
		} /* end parallel */

		error = sqrt(error)/(m*n);
		*iterations += 2;
#ifdef __VERBOSE
		if (*iterations >> 1 == iterations_print) {
			printf("  %8d  %f\n", *iterations, error);
			iterations_print = 2 * iterations_print;
		}
#endif
	} /* end while */
	return error;
}


double RedBlack_GaussSeidel(double **u, int m, int n, double eps,
		int maxit, int omp, int sqrerr, int version,
		int iterations_print, int* iterations, double* wtime) {
	double err;

	double **red;
	double **black;

	int i, j;

	red = CreateGrid(m, n/2);
	black = CreateGrid(m, n/2);

	*iterations = 0;

#ifdef __VERBOSE
	printf ( "\n" );
	printf ( " Iteration  Change\n" );
	printf ( "\n" );
#endif

	*wtime = omp_get_wtime();
#pragma omp parallel shared(u, m, n, red, black) private(i, j,)
	{
		/*
		 * Black sweep
		 */
#pragma omp for
		for (i = 0; i < m; i++)
			for (j = (i+1)%2; j < n; j += 2) {
				black[i][j/2] = u[i][j];
			}

		/*
		 * Red sweep
		 */
#pragma omp for
		for (i = 0; i < m; i++)
			for (j = i%2; j < n; j += 2) {
				red[i][j/2] = u[i][j];
			}
	} /* end parallel */

	if (sqrerr == 0)
		if (omp == 0)
			err = RB_GS_SeqLoop(black, red, m, n, eps, maxit, iterations_print, iterations);
		else
			switch(version) {
			case 1:
				err = RB_GS_OmpLoopV1(black, red, m, n, eps, maxit, iterations_print, iterations);
				break;
			case 2:
				err = RB_GS_OmpLoopV2(black, red, m, n, eps, maxit, iterations_print, iterations);
				break;
			case 3:
				err = RB_GS_OmpLoopV3(black, red, m, n, eps, maxit, iterations_print, iterations);
				break;
			case 4:
				err = RB_GS_OmpLoopV4(black, red, m, n, eps, maxit, iterations_print, iterations);
				break;
			default:
				err = -1.0;
				break;
			}
	else
		if (omp == 0)
			err = RB_GS_SeqLoopErr(black, red, m, n, eps, maxit, iterations_print, iterations);
		else
			switch(version) {
			case 1:
				err = RB_GS_OmpLoopV1err(black, red, m, n, eps, maxit, iterations_print, iterations);
				break;
			case 2:
				err = RB_GS_OmpLoopV2err(black, red, m, n, eps, maxit, iterations_print, iterations);
				break;
			case 3:
				err = RB_GS_OmpLoopV3err(black, red, m, n, eps, maxit, iterations_print, iterations);
				break;
			case 4:
				err = RB_GS_OmpLoopV4err(black, red, m, n, eps, maxit, iterations_print, iterations);
				break;
			default:
				err = -1.0;
				break;
			}

#pragma omp parallel private(j)
	{
		/*
		 * Black sweep
		 */
#pragma omp for
		for (i = 1; i < m-1; i++)
			for (j = i%2+1; j < n-1; j += 2) {
				u[i][j] = black[i][j/2];
			}

		/*
		 * Red sweep
		 */
#pragma omp for
		for (i = 1; i < m-1; i++)
			for (j = (i+1)%2+1; j < n-1; j += 2) {
				u[i][j] = red[i][j/2];
			}
	} /* end parallel */
	*wtime = omp_get_wtime() - *wtime;
	return err;
}
