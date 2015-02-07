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

double RB_GS_SeqLoop(double** black, double **red, int m, int n, double eps,
		int iterations_print, int* iterations) {
	int i, j;
	double diff;
	double v;

	diff = eps;

	/*
     * iterate until the  error is less than the tolerance.
	 */
	while (eps <= diff) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 */
		diff = 0.0;

		/*
		 * Black sweep
		 */
		for (i = 1; i < m - 1; i++)
			for (j = i%2; j < n/2 - (i+1)%2; j++) {
				v = (red[i - 1][j] + red[i + 1][j] + red[i][j - i%2] + red[i][j + (i+1)%2]) / 4.0;
				if (diff < fabs(v - black[i][j]))
					diff = fabs(v - black[i][j]);
				black[i][j] = v;
			}

		/*
		 * Red sweep
		 */
		for (i = 1; i < m - 1; i++)
			for (j = (i+1)%2; j < n/2 - i%2; j++) {
				v = (black[i - 1][j] + black[i + 1][j] + black[i][j - (i+1)%2] + black[i][j + i%2]) / 4.0;
				if (diff < fabs(v - red[i][j]))
					diff = fabs(v - red[i][j]);
				red[i][j] = v;
			}

		(*iterations)++;
		if (*iterations == iterations_print) {
			printf("  %8d  %f\n", *iterations, diff);
			iterations_print = 2 * iterations_print;
		}
	}
	return diff;
}

double RB_GS_OmpLoopV1(double** black, double** red, int m, int n, double eps,
		int iterations_print, int* iterations) {
	int i, j;
	double diff, my_diff;
	double v;

	diff = eps;

	/*
     * iterate until the  error is less than the tolerance.
	 */
	while (eps <= diff) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 */
		diff = 0.0;

#pragma omp parallel private(j, v, my_diff)
		{
			my_diff = 0.0;
			/*
			 * Black sweep
			 */
#pragma omp for
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = (red[i - 1][j] + red[i + 1][j] + red[i][j - i%2] + red[i][j + (i+1)%2]) / 4.0;
					if (my_diff < fabs(v - black[i][j]))
						my_diff = fabs(v - black[i][j]);
					black[i][j] = v;
				}
			if (diff < my_diff)
#pragma omp atomic write
				diff = my_diff;
			/*
			 * Red sweep
			 */
#pragma omp for
			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = (black[i - 1][j] + black[i + 1][j] + black[i][j - (i+1)%2] + black[i][j + i%2]) / 4.0;
					if (my_diff < fabs(v - red[i][j]))
						my_diff = fabs(v - red[i][j]);
					red[i][j] = v;
				}
			if (diff < my_diff)
#pragma omp atomic write
				diff = my_diff;
		} /* end parallel */

		(*iterations)++;
		if (*iterations == iterations_print) {
			printf("  %8d  %f\n", *iterations, diff);
			iterations_print = 2 * iterations_print;
		}
	} /* end while */
	return diff;
}

double RB_GS_OmpLoopV1err(double** black, double** red, int m, int n, double eps,
		int iterations_print, int* iterations) {
	int i, j;
	double v;
	double error;

	error = 10.0*eps;

	/*
     * iterate until the  error is less than the tolerance.
	 */
	while (error > eps) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 */
#pragma omp parallel private(j, v)
		{
			/*
			 * Black sweep
			 */
#pragma omp for reduction(+:error)
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = (red[i - 1][j] + red[i + 1][j] + red[i][j - i%2] + red[i][j + (i+1)%2]) / 4.0;
					error += (v - black[i][j])*(v - black[i][j]);
					black[i][j] = v;
				}

			/*
			 * Red sweep
			 */
#pragma omp for reduction(+:error)
			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = (black[i - 1][j] + black[i + 1][j] + black[i][j - (i+1)%2] + black[i][j + i%2]) / 4.0;
					error += (v - red[i][j])*(v - red[i][j]);
					red[i][j] = v;
				}
		} /* end parallel */

		error = sqrt(error)/(m*n);
		(*iterations)++;

		if (*iterations == iterations_print) {
			printf("  %8d  %f\n", *iterations, error);
			iterations_print = 2 * iterations_print;
		}
	} /* end while */
	return error;
}

double RB_GS_OmpLoopV2(double** black, double** red, int m, int n, double eps,
		int iterations_print, int* iterations) {
	int i, j;
	double diff, my_diff, max_diff;
	double v;

	diff = eps;

	/*
     * iterate until the  error is less than the tolerance.
	 */
#pragma omp parallel private(j, v, my_diff)
	{
		while (eps <= diff) {
			/*
			 * Determine the new estimate of the solution at the interior points.
			 */
#pragma omp critical (diff_update)
			max_diff = 0.0;
// #pragma omp barrier
			my_diff = 0.0;
			/*
			 * Black sweep
			 */
#pragma omp for
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = (red[i - 1][j] + red[i + 1][j] + red[i][j - i%2] + red[i][j + (i+1)%2]) / 4.0;
					if (my_diff < fabs(v - black[i][j]))
						my_diff = fabs(v - black[i][j]);
					black[i][j] = v;
				}

			if (max_diff < my_diff)
#pragma omp critical (diff_update)
				max_diff = my_diff;

			/*
			 * Red sweep
			 */
#pragma omp for
			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = (black[i - 1][j] + black[i + 1][j] + black[i][j - (i+1)%2] + black[i][j + i%2]) / 4.0;
					if (my_diff < fabs(v - red[i][j]))
						my_diff = fabs(v - red[i][j]);
					red[i][j] = v;
				}

				if (max_diff < my_diff)
#pragma omp critical (diff_update)
					max_diff = my_diff;

#pragma omp single
				diff = max_diff;

// #pragma omp barrier

#pragma omp master
			{
				(*iterations)++;
				if (*iterations == iterations_print) {
					printf("  %8d  %f\n", *iterations, diff);
					iterations_print = 2 * iterations_print;
				}
			}
#pragma omp barrier
		} /* end while */
	} /* end parallel */
	return diff;
}

double RB_GS_OmpLoopV2err(double** black, double** red, int m, int n, double eps,
		int iterations_print, int* iterations) {
	int i, j;
	double v;
	double error;

	error = 10.0*eps;

	/*
     * iterate until the  error is less than the tolerance.
	 */
#pragma omp parallel private(j, v)
	{
		while (error > eps) {
			/*
			 * Determine the new estimate of the solution at the interior points.
			 */
			/*
			 * Black sweep
			 */
#pragma omp for reduction(+:error)
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = (red[i - 1][j] + red[i + 1][j] + red[i][j - i%2] + red[i][j + (i+1)%2]) / 4.0;
					error += (v - black[i][j])*(v - black[i][j]);
					black[i][j] = v;
				}
			/*
			 * Red sweep
			 */
#pragma omp for reduction(+:error)
			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = (black[i - 1][j] + black[i + 1][j] + black[i][j - (i+1)%2] + black[i][j + i%2]) / 4.0;
					error += (v - red[i][j])*(v - red[i][j]);
					red[i][j] = v;
				}

#pragma omp single
			error = sqrt(error)/(m*n);

// #pragma omp barrier

#pragma omp master
			{
				(*iterations)++;
				if (*iterations == iterations_print) {
					printf("  %8d  %f\n", *iterations, error);
					iterations_print = 2 * iterations_print;
				}
			}
		} /* end while */
	} /* end parallel */
	return error;
}

double RB_GS_OmpLoopV3(double** black, double** red, int m, int n, double eps,
		int iterations_print, int* iterations) {
	int i, j;
	double *diff;
	double max_diff;
	double v;

	int tid, nthreads;

	nthreads=omp_get_max_threads();
	diff = malloc(sizeof(double)*nthreads);

	max_diff = eps;

	/*
     * iterate until the  error is less than the tolerance.
	 */
#pragma omp parallel private(j, v, tid)
	{
		tid = omp_get_thread_num();

		while (eps <= max_diff) {
			/*
			 * Determine the new estimate of the solution at the interior points.
			 */
			diff[tid] = 0.0;
			/*
			 * Black sweep
			 */
#pragma omp for
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = (red[i - 1][j] + red[i + 1][j] + red[i][j - i%2] + red[i][j + (i+1)%2]) / 4.0;
					if (diff[tid] < fabs(v - black[i][j]))
						diff[tid] = fabs(v - black[i][j]);
					black[i][j] = v;
				}
			/*
			 * Red sweep
			 */
#pragma omp for
			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = (black[i - 1][j] + black[i + 1][j] + black[i][j - (i+1)%2] + black[i][j + i%2]) / 4.0;
					if (diff[tid] < fabs(v - red[i][j]))
						diff[tid] = fabs(v - red[i][j]);
					red[i][j] = v;
				}

// #pragma omp barrier

#pragma omp single
			{
				max_diff = 0.0;
				for(j = 0; j < nthreads; j++)
					if (diff[j] > max_diff)
						max_diff = diff[j];
			}

// #pragma omp barrier

#pragma omp master
			{
				(*iterations)++;
				if (*iterations == iterations_print) {
					printf("  %8d  %f\n", *iterations, max_diff);
					iterations_print = 2 * iterations_print;
				}
			}
		} /* end while */
	} /* end parallel */

	free(diff);

	return max_diff;
}

double RB_GS_OmpLoopV3err(double** black, double** red, int m, int n, double eps,
		int iterations_print, int* iterations) {
	int i, j;
	double *my_error;
	double error;
	double v;

	int tid, nthreads;

	nthreads=omp_get_max_threads();
	my_error = malloc(sizeof(double)*nthreads);

	error = 10.0*eps;

	/*
     * iterate until the  error is less than the tolerance.
	 */
#pragma omp parallel private(j, v, tid)
	{
		tid = omp_get_thread_num();

		while (error > eps) {
			/*
			 * Determine the new estimate of the solution at the interior points.
			 */
			my_error[tid] = 0.0;
			/*
			 * Black sweep
			 */
#pragma omp for
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = (red[i - 1][j] + red[i + 1][j] + red[i][j - i%2] + red[i][j + (i+1)%2]) / 4.0;
					my_error[tid] += (v - black[i][j])*(v - black[i][j]);
					black[i][j] = v;
				}
			/*
			 * Red sweep
			 */
#pragma omp for
			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = (black[i - 1][j] + black[i + 1][j] + black[i][j - (i+1)%2] + black[i][j + i%2]) / 4.0;
					my_error[tid] += (v - red[i][j])*(v - red[i][j]);
					red[i][j] = v;
				}

// #pragma omp barrier

#pragma omp single
			{
				for(j = 0; j < nthreads; j++)
						error += my_error[j];
				error = sqrt(error)/(n*m);
			}

// #pragma omp barrier

#pragma omp master
			{
				(*iterations)++;
				if (*iterations == iterations_print) {
					printf("  %8d  %f\n", *iterations, error);
					iterations_print = 2 * iterations_print;
				}
			}
		} /* end while */
	} /* end parallel */

	free(my_error);

	return error;
}

double RB_GS_OmpLoopV4(double** black, double** red, int m, int n, double eps,
		int iterations_print, int* iterations) {
	int i, j;
	double diff, my_diff;
	double v;

	diff = eps;

	/*
     * iterate until the  error is less than the tolerance.
	 */
	while (eps <= diff) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 */
		diff = 0.0;

#pragma omp parallel private(j, v, my_diff)
		{
			my_diff = 0.0;
			/*
			 * Black sweep
			 */
#pragma omp for
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = (red[i - 1][j] + red[i + 1][j] + red[i][j - i%2] + red[i][j + (i+1)%2]) / 4.0;
					black[i][j] = v;
				}
			/*
			 * Red sweep
			 */
#pragma omp for
			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = (black[i - 1][j] + black[i + 1][j] + black[i][j - (i+1)%2] + black[i][j + i%2]) / 4.0;
					red[i][j] = v;
				}
			/*
			 * Second iteration
			 * Black sweep
			 */
#pragma omp for
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = (red[i - 1][j] + red[i + 1][j] + red[i][j - i%2] + red[i][j + (i+1)%2]) / 4.0;
					if (my_diff < fabs(v - black[i][j]))
						my_diff = fabs(v - black[i][j]);
					black[i][j] = v;
				}
			if (diff < my_diff)
#pragma omp atomic write
				diff = my_diff;
			/*
			 * Second iteration
			 * Red sweep
			 */
#pragma omp for
			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = (black[i - 1][j] + black[i + 1][j] + black[i][j - (i+1)%2] + black[i][j + i%2]) / 4.0;
					if (my_diff < fabs(v - red[i][j]))
						my_diff = fabs(v - red[i][j]);
					red[i][j] = v;
				}
			if (diff < my_diff)
#pragma omp atomic write
				diff = my_diff;
		} /* end parallel */

		*iterations += 2;
		if (*iterations >> 1 == iterations_print) {
			printf("  %8d  %f\n", *iterations, diff);
			iterations_print = 2 * iterations_print;
		}
	} /* end while */
	return diff;
}

double RB_GS_OmpLoopV4err(double** black, double** red, int m, int n, double eps,
		int iterations_print, int* iterations) {
	int i, j;
	double v;
	double error;

	error = 10.0*eps;

	/*
     * iterate until the  error is less than the tolerance.
	 */
	while (error > eps) {
		/*
		 * Determine the new estimate of the solution at the interior points.
		 */
#pragma omp parallel private(j, v)
		{
			/*
			 * Black sweep
			 */
#pragma omp for reduction(+:error)
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = (red[i - 1][j] + red[i + 1][j] + red[i][j - i%2] + red[i][j + (i+1)%2]) / 4.0;
					black[i][j] = v;
				}

			/*
			 * Red sweep
			 */
#pragma omp for reduction(+:error)
			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = (black[i - 1][j] + black[i + 1][j] + black[i][j - (i+1)%2] + black[i][j + i%2]) / 4.0;
					red[i][j] = v;
				}
			/*
			 * Second iteration
			 * Black sweep
			 */
#pragma omp for reduction(+:error)
			for (i = 1; i < m - 1; i++)
				for (j = i%2; j < n/2 - (i+1)%2; j++) {
					v = (red[i - 1][j] + red[i + 1][j] + red[i][j - i%2] + red[i][j + (i+1)%2]) / 4.0;
					error += (v - black[i][j])*(v - black[i][j]);
					black[i][j] = v;
				}

			/*
			 * Second iteration
			 * Red sweep
			 */
#pragma omp for reduction(+:error)
			for (i = 1; i < m - 1; i++)
				for (j = (i+1)%2; j < n/2 - i%2; j++) {
					v = (black[i - 1][j] + black[i + 1][j] + black[i][j - (i+1)%2] + black[i][j + i%2]) / 4.0;
					error += (v - red[i][j])*(v - red[i][j]);
					red[i][j] = v;
				}
		} /* end parallel */

		error = sqrt(error)/(m*n);
		*iterations += 2;

		if (*iterations >> 1 == iterations_print) {
			printf("  %8d  %f\n", *iterations, error);
			iterations_print = 2 * iterations_print;
		}
	} /* end while */
	return error;
}


double RedBlack_GaussSeidel(double **u, int m, int n, double eps, int omp, int sqrerr, int version,
		int iterations_print, int* iterations, double* wtime) {
	double err;

	double **red;
	double **black;

	int i, j;

	red = CreateGrid(m, n/2);
	black = CreateGrid(m, n/2);

	*iterations = 0;

	printf ( "\n" );
	printf ( " Iteration  Change\n" );
	printf ( "\n" );

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
			err = RB_GS_SeqLoop(black, red, m, n, eps, iterations_print, iterations);
		else
			switch(version) {
			case 1:
				err = RB_GS_OmpLoopV1(black, red, m, n, eps, iterations_print, iterations);
				break;
			case 2:
				err = RB_GS_OmpLoopV2(black, red, m, n, eps, iterations_print, iterations);
				break;
			case 3:
				err = RB_GS_OmpLoopV3(black, red, m, n, eps, iterations_print, iterations);
				break;
			case 4:
				err = RB_GS_OmpLoopV4(black, red, m, n, eps, iterations_print, iterations);
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
				err = RB_GS_OmpLoopV1err(black, red, m, n, eps, iterations_print, iterations);
				break;
			case 2:
				err = RB_GS_OmpLoopV2err(black, red, m, n, eps, iterations_print, iterations);
				break;
			case 3:
				err = RB_GS_OmpLoopV3err(black, red, m, n, eps, iterations_print, iterations);
				break;
			case 4:
				err = RB_GS_OmpLoopV4err(black, red, m, n, eps, iterations_print, iterations);
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
