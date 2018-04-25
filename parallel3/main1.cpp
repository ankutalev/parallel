#include <mpi.h>
#include <iostream>
#include <cmath>
//#include <mpich/mpi.h>

#define N1 12
#define N2 12
#define N3 12
#define N4 12


int main(int argc, char**argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double matrixA[N1 * N2];
    double matrixB[N3 * N4];


    MPI_Comm twodims;
    int d[2] = {0, 0};
    int periods[2] = {0, 0};

    MPI_Dims_create(size, 2, d);
    MPI_Cart_create(MPI_COMM_WORLD, 2, d, periods, 1, &twodims);
    MPI_Comm_rank(twodims, &rank);

    int coords[2];
    MPI_Cart_coords(twodims, rank, 2, coords);

    int dims_col[2] = {0, 1};
    int dims_row[2] = {1, 0};


    MPI_Comm column_comm, row_comm;
    MPI_Cart_sub(twodims, dims_col, &column_comm);
    MPI_Cart_sub(twodims, dims_row, &row_comm);


    MPI_Datatype temp1, column, row;
    MPI_Type_vector(N3, N4 / d[1], N4, MPI_DOUBLE, &temp1);
    MPI_Type_create_resized(temp1, 0, N4 / d[1] * sizeof(double), &column);
    MPI_Type_commit(&column);

    MPI_Type_contiguous(N2, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);


    double localB[N3 * N4 / d[1]];
    double localA[N1 * N2 / d[0]];
    double localC[N1 / d[0] * N4 / d[1]] = { 0};


    if (!rank) {
        std::cout<<d[0]<<" "<<d[1]<<std::endl;

        for (int i = 0,k=0; i < N1; ++i,k++) {
            for (int j = 0; j < N2; ++j,k++) {
                matrixA[i*N2+j] = rand() % 15;
                printf("%f ",matrixA[i*N2+j]);
            }
            printf("\n");
        }
//mpi irecv mpi i  recv; mpi i wait - коммуникация между строками

        for (int i = 0,k=0; i < N3; ++i,k++) {
            for (int j = 0; j < N4; ++j,k++) {
                matrixB[i*N4+j] = rand() % 100;
                printf("%f ",matrixB[i*N4+j]);
            }
            printf("\n");
        }
    }


    if (coords[0] == 0)
        MPI_Scatter(matrixB, 1, column, localB, N3 * N4 / d[1], MPI_DOUBLE, 0, column_comm);
    if (coords[1] == 0)
        MPI_Scatter(matrixA, N1 / d[0], row, localA, N1 * N2 / d[0], MPI_DOUBLE, 0, row_comm);






    MPI_Bcast(localB, N3 * N4 / d[1], MPI_DOUBLE, 0, row_comm);
    MPI_Bcast(localA, N1 * N2 / d[0], MPI_DOUBLE, 0, column_comm);


    for (int i = 0; i < N1 / d[0]; ++i)
        for (int j = 0; j < N4 / d[1]; ++j) {
            for (int k = 0; k < N2; ++k) {
                localC[i * N4 / d[1] + j] += localA[i * N2 + k] * localB[k * N4 / d[1] + j];
            }
        }

    for (int n = 0; n < size; ++n) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == n) {
            printf("Rank == %d \n", rank);
            for (int i = 0; i < N1 / d[0]; ++i) {
                for (int j = 0; j < N4 / d[1]; ++j) {
                    printf("%f ",localC[i * N4 / d[1] + j]);
                }
                printf("\n");
            }
        }
    }

    //3 bloka po 6 elementov blocki : N1/D[0], N4/d[1];
    double matrixC[N1 * N4];

    int sendcounts[size];
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = 1;
    }


//   int displs[size] = {0,1,2,12,13,14,24,25,26}; //for 6
//    int displs[size] = {0,1,8,9,16,17}; // for 4 process
int displs[size];
    for (int i = 0; i < d[0]; ++i) {
        for (int j = 0; j < d[1]; ++j) {
            displs[i*d[1]+j] = j + i*N1/d[0]*d[1];
        }
    }

    MPI_Datatype temp, cmatrix;
    MPI_Type_vector(N1, N4/d[1], N4, MPI_DOUBLE, &temp);
    MPI_Type_create_resized(temp, 0, N4/d[1]* sizeof(double), &cmatrix);
    MPI_Type_commit(&cmatrix);
    MPI_Type_commit(&temp);


    MPI_Gatherv(localC, N1 / d[0]*N2/d[1], MPI_DOUBLE, matrixC, sendcounts, displs, cmatrix, 0, twodims);

    if (!rank) {
        printf("REZUZLT IS \n");
        for (int i = 0; i < N1; ++i) {
            for (int j = 0; j < N4; ++j) {
                printf("%.0f ", matrixC[i * N4 + j]);
            }
            printf("\n");
        }
    }

    MPI_Type_free(&column);
    MPI_Finalize();
}