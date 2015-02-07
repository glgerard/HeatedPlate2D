/*
 ============================================================================
 Name        : Heated_Plate_2D.c
 Author      : Gianluca Gerard
 Version     : v0.2
 Copyright   : Copyright (2015) Gianluca Gerard
 Licensing   : This code is distributed under the GNU LGPL license.
 Description : Solves the steady state heat equation with a choice of algorithms.
 Reference(s):
   http://people.sc.fsu.edu/~jburkardt/c_src/heated_plate_openmp/heated_plate_openmp.html
 ============================================================================
 */

#include "HeatedPlate2D.h"

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>
# include <getopt.h>
# include <string.h>
# include <errno.h>
#ifdef __linux__
# include <bsd/string.h>
#endif

# define MAXLEN 256

void SaveGrid(double **u, int m, int n, FILE *fout)
{
	int i,j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++)
			fprintf(fout, "%lf ",u[i][j]);
		fprintf(fout,"\n");
	}
}

int main ( int argc, char *argv[] )

{
	double epsilon = 0.0;
	int m = 512;
	int n = 512;

	double **u;

	int iterations;
	int nthreads = 1;
	double err;
	double wtime;

    int c;
    int gauss_seidel = 0;
    int red_black = 0;
    int openmp = 0;
    int sqrerr = 0;
    int version = 1;

    char outfile[MAXLEN];
    FILE *fd = NULL;

    static struct option long_options[] = {
			{"gs", no_argument, 0, 'g'},
			{"rb", no_argument, 0, 'r'},
			{"sqrerr", no_argument, 0, 's'},
			{"version", required_argument, 0, 'v'},
    		{"openmp", no_argument, 0, 'p'},
			{"mrows", required_argument, 0, 'm'},
			{"ncolumns", required_argument, 0, 'n'},
			{"epsilon", required_argument, 0, 'e'},
			{"output", required_argument, 0, 'o'},
			{"threads", required_argument, 0, 't'},
			{"help", no_argument, 0, '?'},
			{NULL, 0, NULL, 0}
    };

    outfile[0]='\0';

    int option_index = 0;
    errno = 0;
    while (1)
    {
    	c = getopt_long(argc, argv, "grspo:m:n:e:t:v:?",
    	                 long_options, &option_index);
    	if ( c == -1)
    		break;

        switch (c) {
        case 'g':
            gauss_seidel = 1;
            break;
        case 'r':
            red_black = 1;
            break;
        case 's':
        	sqrerr = 1;
        	break;
        case 'v':
        	version = atoi(optarg);
            if (version < 1 || version > NVERSIONS) {
            	fprintf(stderr, "ERROR: version must be between 1 and %d\n", NVERSIONS);
            	exit(-1);
            }
        	break;
        case 'p':
            openmp = 1;
            break;
        case 'm':
            m = strtol(optarg, NULL, 10);
            if (errno == EINVAL) {
            	fprintf(stderr, "ERROR: option m with value '%s'\n", optarg);
            	exit(-1);
            }
            if (m < 2 || m > M) {
            	fprintf(stderr, "ERROR: option m, rows must be between 2 and %d\n", M);
            	exit(-1);
            }
            break;
        case 'n':
            n = strtol(optarg, NULL, 10);
            if (errno == EINVAL) {
            	fprintf(stderr, "ERROR: option n with value '%s'\n", optarg);
            	exit(-1);
            }
            if (n < 2 || n > N) {
            	fprintf(stderr, "ERROR: option n, columns must be between 2 and %d\n", N);
            	exit(-1);
            }
            break;
        case 'e':
            epsilon = strtod(optarg, NULL);
            if (errno == EINVAL) {
            	fprintf(stderr, "ERROR: option n with value '%s'\n", optarg);
            	exit(-1);
            }
            if (epsilon <= 0.0) {
            	fprintf(stderr, "ERROR: option e, columns must be greater than 0\n");
            	exit(-1);
            }
            break;
        case 'o':
        	strlcpy(outfile, optarg, sizeof(outfile));
        	break;
        case 't':
        	nthreads = atoi(optarg);
        	break;
        case '?':
        	printf("Usage: HeatedPlate [OPTION]\n");
        	printf("Solves the steady state heat equation with various solvers.\n");
        	printf("\n");
        	printf("Mandatory arguments to long options are mandatory for short options too.\n");
        	printf("  -g, --gs\t\tUse Gauss-Seidel iterations (default Jacobi iterations).\n");
        	printf("  -r, --rb\t\tUse Red-Black ordering (only for Gauss-Seidel).\n");
        	printf("  -v, --version\t\tUse the V version of the algorithm.\n");
        	printf("  -s, --sqrerr\t\tCompute the squared mean error to test for convergence.\n");
        	printf("  -p, --openmp\t\tUse the OpenMP variants (RB G-S and Jacobi only).\n");
        	printf("  -t, --threads\t\tUse T threads.\n");
        	printf("  -m, --mrows=M\t\tUse a grid with M rows.\n");
        	printf("  -n, --ncols=N\t\tUse a grid with N columns.\n");
        	printf("  -e, --epsilon=E\tStops when the error is less than E*M*N.\n");
        	exit(-1);
            break;
        default:
            printf ("?? getopt returned character code 0%o ??\n", c);
        }
    }
    if (optind < argc) {
        printf ("non-option ARGV-elements: ");
        while (optind < argc)
            printf ("%s ", argv[optind++]);
        printf ("\n");
    }

    if (epsilon == 0.0) {
    	if (sqrerr == 0)
    		epsilon = 0.001;
    	else
    		epsilon = 0.000001;
    }

    /*
     * If an output file is provided it will store the outcome U in that file.
     */
    if (outfile[0] != '\0') {
    	errno=0;
    	if ((fd = fopen(outfile,"w+")) == NULL) {
    		fprintf(stderr, "ERROR: Could not create output file '%s'. Error number = %d\n", outfile, errno);
    		exit(-1);
    	}
    }

	printf ( "\n" );
	printf ( "HEATED_PLATE\n");
	if (openmp==1) {
		printf( " OpenMP version.\n");
		omp_set_num_threads(nthreads);
	}
	printf ( "  A program to solve for the steady state temperature distribution\n" );
	printf ( "  over a rectangular plate.\n" );

	if (red_black == 1)
		printf ( "  Red Black Gauss-Seidel implementation.\n" );
	else if (gauss_seidel == 1)
		printf ( "  Gauss-Seidel implementation.\n" );
	else
		printf ( "  Jacobi implementation.\n" );

	if (sqrerr == 0)
		printf("      Maximum error tolerance.\n");
	else
		printf("      Average squared error tolerance.\n");

	printf("      Version = %d\n", version);

	printf ( "\n" );
	printf ( "  Spatial grid of %d by %d points.\n", m, n );
	printf ( "  The iteration will be repeated until the change is <= %e\n", epsilon );
	printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
	nthreads = omp_get_max_threads();
	printf ( "  Number of threads =              %d\n", nthreads );

	/*
	 * Create and initialize the temperature grid u
	 */
	u = InitGrid(m, n, 1);
	if (u == NULL) {
		fprintf(stderr, "ERROR: Couldn't allocate memory to store the grid U.\n");
		exit(-1);
	}

	if (red_black == 1)
		err = RedBlack_GaussSeidel(u, m, n, epsilon, openmp, sqrerr, version,
					1, &iterations, &wtime);
	else if (gauss_seidel == 1)
		if (openmp == 0 && version == 1)
			err = GaussSeidel(u, m, n, sqrerr, epsilon, 1, &iterations, &wtime);
		else
			err = -1.0;
	else
		err = Jacobi(u, m, n, epsilon, openmp, sqrerr, version,
				1, &iterations, &wtime);

	if (err < 0.0) {
		fprintf(stderr, "ERROR: Method not implemented!\n");
		exit(-1);
	}

	printf ( "\n" );
	printf ( "  Error tolerance achieved.\n" );
	printf ( "  Threads = %d, Wallclock time = %f, Iterations = %d, Error = %f\n", nthreads, wtime, iterations, err);

	if (fd != NULL) {
		SaveGrid(u, m, n, fd);
		fclose(fd);
	}

	/*
     * Terminate.
	 */
	printf ( "\n" );
	printf ( "HEATED_PLATE:\n" );
	printf ( "  Normal end of execution.\n" );

	FreeGrid(u,m);

	return 0;
}

