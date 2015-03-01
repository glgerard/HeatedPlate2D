/*
 * HeatedPlate2D.h
 *
 *  Created on: Jan 1, 2015
 *      Author: Gianluca Gerard
 */

#ifndef HEATEDPLATELOCAL_SRC_HEATEDPLATE_H_
#define HEATEDPLATELOCAL_SRC_HEATEDPLATE_H_

#define OMP_SCHED static

# define M 8192
# define N 8192

# define NVERSIONS 4

double **CreateGrid(int m, int n);

double **InitGrid(int m, int n, int output);

void FreeGrid(double **u, int m);

double GaussSeidel(double **u, int m, int n, int sqrerr, double eps, int maxit,
		int iterations_print, int* iterations, double* wtime);

double Jacobi(double **u, int m, int n, double eps, int maxit, int omp, int sqrerr, int version,
		int iterations_print, int* iterations, double* wtime);

double RedBlack_GaussSeidel(double **u, int m, int n, double eps, int maxit, int omp, int sqrerr, int version,
		int iterations_print, int* iterations, double* wtime);

#endif /* HEATEDPLATELOCAL_SRC_HEATEDPLATE_H_ */
