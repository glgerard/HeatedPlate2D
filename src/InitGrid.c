/*
 ============================================================================
 Name        : InitGrid.c
 Author      : Gianluca Gerard
 Version     : v0.1
 Copyright   : Copyright (2015) Gianluca Gerard
 Licensing   : This code is distributed under the GNU LGPL license.
 Description : Create and initialize the temperature grid.
 Reference(s):
   http://people.sc.fsu.edu/~jburkardt/c_src/heated_plate_openmp/heated_plate_openmp.html
 ============================================================================
 */

# include <stdlib.h>
# include <stdio.h>
# include <errno.h>
# include <unistd.h>

#define DCACHE_L1_SIZE 64

double **CreateGrid(int m, int n)
{
	double **u;
	double *urow;

	int i;
	int errno;

#ifdef DCACHE_L1_SIZE
	size_t alignment = DCACHE_L1_SIZE;
#else
	size_t alignment = sysconf(_SC_PAGESIZE) ;
#endif

	if ( ( errno = posix_memalign((void**)&u, alignment, sizeof(double*)*m)) != 0) {
		if (errno == EINVAL)
			fprintf(stderr, "ERROR: The value of the alignment parameter is not a power of two multiple of sizeof(void *).\n" );
		else if (errno == ENOMEM)
			fprintf(stderr, "ERROR: There is insufficient memory available with the requested alignment.\n" );
		else
			fprintf(stderr, "ERROR: Error number %d\n", errno);
		return NULL;
	}

	for (i = 0; i < m; i++) {
		if ( ( errno = posix_memalign((void**)&urow, alignment, sizeof(double*)*n)) != 0) {
			if (errno == EINVAL)
				fprintf(stderr, "ERROR: The value of the alignment parameter is not a power of two multiple of sizeof(void *).\n" );
			else if (errno == ENOMEM)
				fprintf(stderr, "ERROR: There is insufficient memory available with the requested alignment.\n" );
			else
				fprintf(stderr, "ERROR: Error number %d\n", errno);
			return NULL;
		}
		u[i] = urow;
	}

	return u;
}

/*
 * Create and initialize the temperature grid.
 */
double **InitGrid(int m, int n, int output)
{
	double **u;

	double mean = 0.0;
	int i, j;

	if ( (u = CreateGrid(m, n)) == NULL)
		return NULL;

#pragma omp parallel shared(u, m, n) private(i, j)
	{
		/*
		 * Set the boundary values, which don't change.
		 */
#pragma omp for
		for (i = 1; i < m - 1; i++) {
			u[i][0] = 100.0;
			u[i][n - 1] = 100.0;
		}
#pragma omp for
		for (j = 0; j < n; j++) {
			u[0][j] = 0.0;
			u[m - 1][j] = 100.0;
		}
		/*
		 * Average the boundary values, to come up with a reasonable
		 * initial value for the interior.
		 */
#pragma omp for reduction(+:mean)
		for (i = 1; i < m - 1; i++) {
			mean += u[i][0] + u[i][n - 1];
		}
#pragma omp for reduction(+:mean)
		for (j = 0; j < n; j++) {
			mean += u[m - 1][j] + u[0][j];
		}
	}

	mean = mean / (double) (2 * m + 2 * n - 4);
	if (output == 1) {
		printf("\n");
		printf("  MEAN = %f\n", mean);
	}
	/*
	 * Initialize the interior solution to the mean value.
	 */
#pragma omp parallel for shared(u, m, n, mean) private(i,j)
	for (i = 1; i < m - 1; i++) {
		for (j = 1; j < n - 1; j++) {
			u[i][j] = mean;
		}
	}

	return u;
}

void FreeGrid(double **u, int m)
{
	int i;

#pragma omp parallel for shared (m) private(i)
	for (i = 0; i < m; i++)
		free(u[i]);

	free(u);
}
