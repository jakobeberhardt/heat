#include "/opt/homebrew/Cellar/open-mpi/4.1.6/include/mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include "heat.h"

void usage(char *s) {
    fprintf(stderr, "Usage: %s <input file> [result file]\n\n", s);
}

void exchange_ghosts_jacobi(double *u, int np, int rows, int rank, int numprocs) {
    MPI_Status status;
    MPI_Barrier(MPI_COMM_WORLD);

    // Send to the next rank your last row only if you are not the last process
    if (rank < numprocs - 1) {
        MPI_Send(&u[(rows) * np], np, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
    }

    // Receive from the previous member their last row only if you are not the first process
    if (rank > 0) {
        MPI_Recv(u, np, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
    }

    // Send the first row to the previous rank only if you are not rank 0
    if (rank > 0) {
        MPI_Send(&u[np], np, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD);
    }

    // Receive the first row from the next rank only if you are not the last rank
    if (rank < numprocs - 1) {
        MPI_Recv(&u[(rows + 1) * np], np, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &status);
    }
}

void exchange_boundaries_gauss(double *u, int np, int rows, int rank, int numprocs, int iteration) {
    int block_size = (np) / numprocs;
    int block_index = iteration * block_size + 1;
    MPI_Status status;

    MPI_Barrier(MPI_COMM_WORLD);

    // Send to the next rank your last row only if you are not the last process
    if (rank < numprocs - 1) {
        MPI_Send(&u[(rows) * np + block_index], block_size, MPI_DOUBLE, rank + 1, iteration, MPI_COMM_WORLD);
    }

    // Receive from the previous member their last row only if you are not the first process
    if (rank > 0) {
        MPI_Recv(&u[block_index], block_size, MPI_DOUBLE, rank - 1, iteration, MPI_COMM_WORLD, &status);
    }

    // Send the first row to the previous rank only if you are not rank 0
    if (rank > 0) {
        MPI_Send(&u[np + block_index], block_size, MPI_DOUBLE, rank - 1, iteration + 1, MPI_COMM_WORLD);
    }

    // Receive the first row from the next rank only if you are not the last rank
    if (rank < numprocs - 1) {
        MPI_Recv(&u[(rows + 1) * np + block_index], block_size, MPI_DOUBLE, rank + 1, iteration + 1, MPI_COMM_WORLD, &status);
    }
}

void do_calculations_master_blocking(algoparam_t *param, int rows, int np, int rank, int numprocs, int algorithm) {
    double residual = 0.0;
    double global_residual = 0.0;
    unsigned iter = 0;
    int ghost_rows = rows + 2;
    int block_size = (np) / numprocs;
    int flag; // For MPI_Test
    int send_tag = rank * numprocs + (rank + 1);

    MPI_Status statuses [numprocs];
    MPI_Request r_master[numprocs];

    while (1) {
        switch (algorithm) {
            case 0: // JACOBI
                residual = relax_jacobi(param->u, param->uhelp, ghost_rows, np);
                // Copy uhelp into u
                for (int i = 0; i < ghost_rows; i++) {
                    for (int j = 0; j < np; j++) {
                        param->u[i * np + j] = param->uhelp[i * np + j];
                    }
                }
                break;
            case 1: // RED-BLACK
                residual = relax_redblack(param->u, np, np);
                break;
            case 2: // GAUSS
                residual = 0.0;
                for (int i = 0; i < numprocs; i++) {
                    residual += relax_gauss(param->u, ghost_rows, np, numprocs, i);
                    MPI_Send(&param->u[(rows * np) + (i * block_size + 1)], block_size, MPI_DOUBLE, rank + 1, send_tag, MPI_COMM_WORLD);
                    
                }
                
                break;
        }

        MPI_Allreduce(&residual, &global_residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Recv(&param->u[(rows + 1) * np], np, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &statuses[0]);

        iter++;

        if (algorithm == 0) {
            MPI_Allreduce(&residual, &global_residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            exchange_ghosts_jacobi(param->u, np, rows, rank, numprocs);
        }

        // Solution good enough?
        if (global_residual < 0.00005) break;

        // Max iteration reached? (no limit with maxiter=0)
        if (param->maxiter > 0 && iter >= param->maxiter) break;
    }
}
void do_calculations_master_nonblocking(algoparam_t *param, int rows, int np, int rank, int numprocs, int algorithm) {
    double residual = 0.0;
    double global_residual = 0.0;
    unsigned iter = 0;
    int ghost_rows = rows + 2;
    int block_size = (np) / numprocs;
    int flag; // For MPI_Test
    int send_tag = rank * numprocs + (rank + 1);

    MPI_Status statuses [numprocs];
    MPI_Request r_master[numprocs];

    while (1) {
        switch (algorithm) {
            case 0: // JACOBI
                residual = relax_jacobi(param->u, param->uhelp, ghost_rows, np);
                // Copy uhelp into u
                for (int i = 0; i < ghost_rows; i++) {
                    for (int j = 0; j < np; j++) {
                        param->u[i * np + j] = param->uhelp[i * np + j];
                    }
                }
                break;
            case 1: // RED-BLACK
                residual = relax_redblack(param->u, np, np);
                break;
            case 2: // GAUSS
                residual = 0.0;
                for (int i = 0; i < numprocs; i++) {
                    residual += relax_gauss(param->u, ghost_rows, np, numprocs, i);
                    MPI_Isend(&param->u[(rows * np) + (i * block_size + 1)], block_size, MPI_DOUBLE, rank + 1, send_tag, MPI_COMM_WORLD,&r_master[i]);
                    MPI_Wait(&r_master[i],&statuses[i]);
                    
                }
                
                break;
        }

        MPI_Allreduce(&residual, &global_residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


        MPI_Recv(&param->u[(rows + 1) * np], np, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &statuses[0]);

        iter++;

        if (algorithm == 0) {
            MPI_Allreduce(&residual, &global_residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            exchange_ghosts_jacobi(param->u, np, rows, rank, numprocs);
        }

        // Solution good enough?
        if (global_residual < 0.00005) break;

        // Max iteration reached? (no limit with maxiter=0)
        if (param->maxiter > 0 && iter >= param->maxiter) break;
    }
}

void do_calculations_worker_nonblocking(double *u, double *uhelp, int rank, int numprocs, int rows, int np, int maxiter, int algorithm) {
    double residual = 0.0;
    double global_residual = 0.0;
    unsigned iter = 0;
    int ghost_rows = rows + 2;
    int block_size = (np) / numprocs;
    MPI_Status statuses[numprocs];
    MPI_Request r_send[numprocs];
    MPI_Request r_recv[numprocs];
    int recv_tag = (rank - 1) * numprocs + rank;
    int send_tag = rank * numprocs + (rank + 1);

    while (1) {
        switch (algorithm) {
            case 0: // JACOBI
                residual = relax_jacobi(u, uhelp, ghost_rows, np);
                // Copy uhelp into u
                for (int i = 0; i < ghost_rows; i++) {
                    for (int j = 0; j < np; j++) {
                        u[i * np + j] = uhelp[i * np + j];
                    }
                }
                break;
            case 1: // RED-BLACK
                residual = relax_redblack(u, np, np);
                break;
            case 2: // GAUSS
                residual = 0.0;
                for (int i = 0; i < numprocs; i++) {
                    MPI_Irecv(&u[i * block_size + 1], block_size, MPI_DOUBLE, rank - 1, recv_tag, MPI_COMM_WORLD, &r_recv[i]);
                    MPI_Wait(&r_recv[i],&statuses[i]);
                    residual += relax_gauss(u, ghost_rows, np, numprocs, i);
                    if (rank != numprocs - 1) {
                        MPI_Isend(&u[(rows * np) + (i * block_size + 1)], block_size, MPI_DOUBLE, rank + 1, send_tag, MPI_COMM_WORLD,&r_send[i]);
                        MPI_Wait(&r_send[i],&statuses[i]);
                    }
                }
                
                break;
        }
        MPI_Allreduce(&residual, &global_residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     
        MPI_Send(&u[np], np, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD);
        if (rank != numprocs - 1) {
            MPI_Recv(&u[(rows + 1) * np], np, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &statuses[0]);
        }

        if (algorithm == 0) {
            MPI_Allreduce(&residual, &global_residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            exchange_ghosts_jacobi(u, np, rows, rank, numprocs);
        }

        iter++;

        // Solution good enough?
        if (global_residual < 0.00005) break;

        // Max iteration reached? (no limit with maxiter=0)
        if (maxiter > 0 && iter >= maxiter) break;
    }
}

void do_calculations_worker_blocking(double *u, double *uhelp, int rank, int numprocs, int rows, int np, int maxiter, int algorithm) {
    double residual = 0.0;
    double global_residual = 0.0;
    unsigned iter = 0;
    int ghost_rows = rows + 2;
    int block_size = (np) / numprocs;
    MPI_Status statuses[numprocs];
    MPI_Request r_send[numprocs];
    MPI_Request r_recv[numprocs];
    int recv_tag = (rank - 1) * numprocs + rank;
    int send_tag = rank * numprocs + (rank + 1);

    while (1) {
        switch (algorithm) {
            case 0: // JACOBI
                residual = relax_jacobi(u, uhelp, ghost_rows, np);
                // Copy uhelp into u
                for (int i = 0; i < ghost_rows; i++) {
                    for (int j = 0; j < np; j++) {
                        u[i * np + j] = uhelp[i * np + j];
                    }
                }
                break;
            case 1: // RED-BLACK
                residual = relax_redblack(u, np, np);
                break;
            case 2: // GAUSS
                residual = 0.0;
                for (int i = 0; i < numprocs; i++) {
                    MPI_Recv(&u[i * block_size + 1], block_size, MPI_DOUBLE, rank - 1, recv_tag, MPI_COMM_WORLD, &statuses[i]);
                    residual += relax_gauss(u, ghost_rows, np, numprocs, i);
                    if (rank != numprocs - 1) {
                        MPI_Send(&u[(rows * np) + (i * block_size + 1)], block_size, MPI_DOUBLE, rank + 1, send_tag, MPI_COMM_WORLD);
                        
                    }
                }
                
                break;
        }
        MPI_Allreduce(&residual, &global_residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Send(&u[np], np, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD);
        if (rank != numprocs - 1) {
            MPI_Recv(&u[(rows + 1) * np], np, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &statuses[0]);
        }

        if (algorithm == 0) {
            MPI_Allreduce(&residual, &global_residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            exchange_ghosts_jacobi(u, np, rows, rank, numprocs);
        }

        iter++;

        // Solution good enough?
        if (global_residual < 0.00005) break;

        // Max iteration reached? (no limit with maxiter=0)
        if (maxiter > 0 && iter >= maxiter) break;
    }
}

int main(int argc, char *argv[]) {
    int iter = 0;
    FILE *infile, *resfile;
    char *resfilename;
    int myid, numprocs;
    MPI_Status status = {.MPI_SOURCE = 0, .MPI_TAG = 0, .MPI_ERROR = 0};

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid == 0) {
        printf("I am the master (%d) and going to distribute work to %d additional workers ...\n", myid, numprocs - 1);

        // Algorithmic parameters
        algoparam_t param;
        int np;
        double runtime, flop;
        double residual = 0.0;

        // Check arguments
        if (argc < 2) {
            usage(argv[0]);
            return 1;
        }

        // Check input file
        if (!(infile = fopen(argv[1], "r"))) {
            fprintf(stderr, "\nError: Cannot open \"%s\" for reading.\n\n", argv[1]);
            usage(argv[0]);
            return 1;
        }

        // Check result file
        resfilename = (argc >= 3) ? argv[2] : "heat.ppm";

        if (!(resfile = fopen(resfilename, "w"))) {
            fprintf(stderr, "\nError: Cannot open \"%s\" for writing.\n\n", resfilename);
            usage(argv[0]);
            return 1;
        }

        // Check input
        if (!read_input(infile, &param)) {
            fprintf(stderr, "\nError: Error parsing input file.\n\n");
            usage(argv[0]);
            return 1;
        }
        print_params(&param);

        // Set the visualization resolution
        param.u = 0;
        param.uhelp = 0;
        param.uvis = 0;
        param.visres = param.resolution;

        if (!initialize(&param)) {
            fprintf(stderr, "Error in Solver initialization.\n\n");
            usage(argv[0]);
            return 1;
        }

        // Full size (param.resolution are only the inner points)
        np = param.resolution + 2;
        int rows = param.resolution / numprocs;

        // Starting time
        runtime = wtime();

        // Send data to workers
        for (int i = 0; i < numprocs; i++) {
            if (i > 0) {
                MPI_Send(&param.maxiter, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.resolution, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.algorithm, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.u[i * np * rows], (rows + 2) * (np), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.uhelp[i * np * rows], (rows + 2) * (np), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        do_calculations_master_nonblocking(&param, rows, np, myid, numprocs, param.algorithm);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        for (int i = 0; i < numprocs; i++) {
            if (i > 0) {
                MPI_Recv(&param.u[i * np * rows + np], np * rows, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &status);
            }
        }

        // Flop count after iter iterations
        flop = iter * 11.0 * param.resolution * param.resolution;
        // Stopping time
        runtime = wtime() - runtime;

        fprintf(stdout, "Time: %04.3f ", runtime);
        fprintf(stdout, "(%3.3f GFlop => %6.2f MFlop/s)\n", flop / 1000000000.0, flop / runtime / 1000000);
        //fprintf(stdout, "Convergence to residual=%f: %d iterations\n", residual, iter);

        // For plot...
        coarsen(param.u, np, np, param.uvis, param.visres + 2, param.visres + 2);

        write_image(resfile, param.u, param.visres + 2, param.visres + 2);

        finalize(&param);

        //fprintf(stdout, "Process %d finished computing with residual value = %f\n", myid, residual);

        MPI_Finalize();

        return 0;
    } else {
        printf("I am worker %d and ready to receive work to do ...\n", myid);

        int columns, rows, np;
        int algorithm;
        double residual;
        int maxiter;

        MPI_Recv(&maxiter, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&columns, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&algorithm, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        np = columns + 2;
        rows = columns / numprocs;

        // Allocate memory for worker
        double *u = calloc(sizeof(double), (rows+2) * (np));
        double *uhelp = calloc(sizeof(double), (rows+2 ) * (np));
        if ((!u) || (!uhelp)) {
            fprintf(stderr, "Error: Cannot allocate memory\n");
            return 0;
        }
         // Fill initial values for the matrix with values received from the master
        MPI_Recv(&u[0], (rows + 2) * (np), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&uhelp[0], (rows + 2) * (np), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

        MPI_Barrier(MPI_COMM_WORLD);

        do_calculations_worker_nonblocking(u, uhelp, myid, numprocs, rows, np, maxiter, algorithm);

        MPI_Send(&u[np], (np) * (rows), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);

        if (u) free(u);
        if (uhelp) free(uhelp);

        fprintf(stdout, "Process %d finished computing %d iterations with residual value = %f\n", myid, iter, residual);

        MPI_Finalize();
        exit(0);
    }
}
