#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "heat.h"

void usage(char *s) {
    fprintf(stderr, 
            "Usage: %s <input file> [result file]\n\n", s);
}

int do_computation(double *u, double *uhelp, int np, int algorithm, int maxiter, int id) {
    double residual;
    int iter = 0;

    while (1) {
        switch (algorithm) {
            case 0: // JACOBI
                residual = relax_jacobi(u, uhelp, np, np);
                for (int i = 0; i < np; i++)
                    for (int j = 0; j < np; j++)
                        u[i * np + j] = uhelp[i * np + j];
                break;
            case 1: // RED-BLACK
                residual = relax_redblack(u, np, np);
                break;
            case 2: // GAUSS
                residual = relax_gauss(u, np, np);
                break;
        }

        iter++;

        if (residual < 0.00005) break;
        if (maxiter > 0 && iter >= maxiter) break;
    }

    fprintf(stdout, "Process %d finished computing %d iterations with residual value = %f\n", id, iter, residual);
    return iter;
}

int main(int argc, char *argv[]) {
    unsigned iter;
    FILE *infile, *resfile;
    char *resfilename;
    int myid, numprocs;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid == 0) {
        printf("I am the master (%d) and going to distribute work to %d additional workers ...\n", myid, numprocs-1);

        // algorithmic parameters
        algoparam_t param;
        int np;
        double runtime, flop;
        //double residual=0.0;

        if (argc < 2) {
            usage(argv[0]);
            return 1;
        }

        if (!(infile=fopen(argv[1], "r"))) {
            fprintf(stderr, "\nError: Cannot open \"%s\" for reading.\n\n", argv[1]);
            usage(argv[0]);
            return 1;
        }

        resfilename= (argc >= 3) ? argv[2] : "heat.ppm";
        if (!(resfile=fopen(resfilename, "w"))) {
            fprintf(stderr, "\nError: Cannot open \"%s\" for writing.\n\n", resfilename);
            usage(argv[0]);
            return 1;
        }

        if (!read_input(infile, &param)) {
            fprintf(stderr, "\nError: Error parsing input file.\n\n");
            usage(argv[0]);
            return 1;
        }
        print_params(&param);

        param.u     = 0;
        param.uhelp = 0;
        param.uvis  = 0;
        param.visres = param.resolution;

        if (!initialize(&param)) {
            fprintf(stderr, "Error in Solver initialization.\n\n");
            usage(argv[0]);
            return 1;
        }

        np = param.resolution + 2;
        runtime = wtime();

        for (int i = 0; i < numprocs; i++) {
            if (i > 0) {
                MPI_Send(&param.maxiter, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.resolution, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.algorithm, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.u[0], (np) * (np), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.uhelp[0], (np) * (np), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
        }

        iter = do_computation(param.u, param.uhelp, np, param.algorithm, param.maxiter, myid);

        flop = iter * 11.0 * param.resolution * param.resolution;
        runtime = wtime() - runtime;

        fprintf(stdout, "Time: %04.3f ", runtime);
        fprintf(stdout, "(%3.3f GFlop => %6.2f MFlop/s)\n", 
            flop/1000000000.0,
            flop/runtime/1000000);

        coarsen(param.u, np, np, param.uvis, param.visres + 2, param.visres + 2);
        write_image(resfile, param.uvis, param.visres + 2, param.visres + 2);
        finalize(&param);

        MPI_Finalize();
        return 0;

    } else {
        printf("I am worker %d and ready to receive work to do ...\n", myid);

        int columns, rows, np;
        int algorithm;
        int maxiter;

        MPI_Recv(&maxiter, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&columns, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&algorithm, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        rows = columns;
        np = columns + 2;

        double *u = calloc(sizeof(double), (rows + 2) * (columns + 2));
        double *uhelp = calloc(sizeof(double), (rows + 2) * (columns + 2));
        if (!u || !uhelp) {
            fprintf(stderr, "Error: Cannot allocate memory\n");
            return 0;
        }

        MPI_Recv(&u[0], (rows + 2) * (columns + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&uhelp[0], (rows + 2) * (columns + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

        iter = do_computation(u, uhelp, np, algorithm, maxiter, myid);

        if (u) free(u);
        if (uhelp) free(uhelp);

        MPI_Finalize();
        exit(0);
    }
}
