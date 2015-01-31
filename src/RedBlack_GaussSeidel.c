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

double RB_GS_OmpLoop(double** black, double** red, int m, int n, double eps,
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

#pragma omp parallel shared(red, black, m, n, diff) private(i, j, v, my_diff)
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
#pragma omp critical
			if (diff < my_diff)
				diff = my_diff;
		}
		(*iterations)++;
		if (*iterations == iterations_print) {
			printf("  %8d  %f\n", *iterations, diff);
			iterations_print = 2 * iterations_print;
		}
	}
	return diff;
}

double RedBlack_GaussSeidel(double **u, int m, int n, double eps, int omp,
		int iterations_print, int* iterations, double* wtime) {
	double err;

	double **red;
	double **black;

	int i, j;

	red = CreateGrid(m,n/2);
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
}
	if (omp == 0)
		err = RB_GS_SeqLoop(black, red, m, n, eps, iterations_print, iterations);
	else
		err = RB_GS_OmpLoop(black, red, m, n, eps, iterations_print, iterations);
#pragma omp parallel shared(u, m, n, red, black) private(i, j,)
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
}
	*wtime = omp_get_wtime() - *wtime;
	return err;
}
