#include <iostream>
#include <mpi/mpi.h>
#include <fstream>
#include <cmath>

#define N 10

void processInit(double *matrixPart, double *vectorPart, const std::string &matrixFileName,
                 const std::string &vectorFileName, int *recv, int *displs, int rank) {

    std::ifstream fin;
    fin.open(matrixFileName);
    std::ofstream fout;
    fout.open(std::to_string(rank));
    std::string tmp;

    int stringsInOneProcess = recv[rank];
    int i = 0;
    for (i = 0; i < displs[rank]; i++) {
        fin.ignore(256, '\n');
    }
    for (i = 0; i < N * stringsInOneProcess; i++) {
        fin >> matrixPart[i];
        fout << matrixPart[i] << " ";
    }

    fout << std::endl;
    fin.close();



    fin.open(vectorFileName);
    for (int i = 0; i < displs[rank]; i++) {
        fin.ignore(256, ' ');
    }

    for (i = 0; i < recv[rank]; i++) {
        fin >> vectorPart[i];
        fout << vectorPart[i] << " ";
    }
}

void multiply(const double *matrix, double *vector, double *rez,int* recv,int* displs, int rank, int size) {

    int stringsInOneProcess = recv[rank];
    MPI_Status st;
    int startPosition = displs[rank];
    int nextRank = (rank -1 + size) %size;
    int vectorCellsInOneProcess = stringsInOneProcess;

    for (int i = 0; i < recv[0]; i++)
        rez[i] = 0;

    for (int j = 0; j < size; j++) {
        for (int k = 0; k < stringsInOneProcess; k++) {
            for (int i = 0; i < vectorCellsInOneProcess; i++) {
                rez[k] += vector[i] * matrix[i + startPosition + N * k];
            }
        }

        if (size != 1) {
            vectorCellsInOneProcess = recv[nextRank];
            startPosition= displs[nextRank];
            nextRank = (!nextRank) ? size -1 : nextRank-1;
            MPI_Sendrecv_replace(vector, recv[0], MPI_DOUBLE, (rank + 1) % size, 10,
                          (rank-1+size)%size, 10, MPI_COMM_WORLD, &st);
        }
    }
}


void vectorSub(double *fvector, double *svector, double *rezvector, int vectorSize) {
    for (int i = 0; i < vectorSize; i++)
        rezvector[i] = fvector[i] - svector[i];
}

void scalarMultiply(double *vector, double scalar, double *rez, int vectorSize) {
    for (int i = 0; i < vectorSize; ++i) {
        rez[i] = vector[i] * scalar;
    }
}

double parametherCalcuation(const double *fvector, const double *svector, int vectorSize, int rank, int size) {
    double a = 0, b = 0;
    double total = 0;
    double allA = 0, allB = 0;
    for (int i = 0; i < vectorSize; i++) {
        a += fvector[i] * svector[i];
        b += svector[i] * svector[i];
    }
    MPI_Reduce(&a, &allA, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&b, &allB, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (!rank) {
        total = allA / allB;
    }
    MPI_Bcast(&total, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return total;
}

double getSummOfPartOfVector(const double *partOfVector, int stringsInOneProcess) {
    double sum = 0;
    for (int i = 0; i < stringsInOneProcess; ++i) {
        sum += partOfVector[i] * partOfVector[i];
    }
    return sum;
}

bool algoCondition(const double *matrix, double *partOfXVector, double *partOfBVector, int* recv, int* displs, int rank,
                   int size) {
    auto tmp = new double[recv[0]];
    double epsilon = 0.01;
    double NumeratorNorm = 0;
    static double DenominatorNorm = 0;
    static int lock = 0;
    bool cond;

    multiply(matrix,partOfXVector,tmp,recv,displs,rank,size);
    vectorSub(tmp, partOfBVector, tmp, recv[rank]);
    double partOfNumeratorNorm = getSummOfPartOfVector(tmp, recv[rank]);
    MPI_Reduce(&partOfNumeratorNorm, &NumeratorNorm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (!lock) {
        double partOfDenominatorNorm = getSummOfPartOfVector(partOfBVector, recv[rank]);
        MPI_Reduce(&partOfDenominatorNorm, &DenominatorNorm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        lock++;
    }

    if (!rank) {
        cond = sqrt(NumeratorNorm /DenominatorNorm) > epsilon;
    }

    MPI_Bcast(&cond, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    delete[] tmp;
    return cond;
}

void algorithm(double *matrix, double *xvector, double *bvector, double *yvector, double *rezult, int *recv,
               int *displs, int rank, int size, size_t *iter_count) {
    auto tmpVector = new double[recv[0]];
    double methodParamether;
    while (algoCondition(matrix, xvector, bvector, recv,displs, rank, size)) {
        multiply(matrix, xvector, tmpVector,recv,displs, rank, size);
        vectorSub(tmpVector, bvector, yvector, recv[rank]);
        multiply(matrix, yvector, tmpVector,recv,displs, rank, size);
        methodParamether = parametherCalcuation(yvector, tmpVector, recv[rank], rank, size);
        scalarMultiply(yvector, methodParamether, tmpVector, recv[rank]);
        vectorSub(xvector, tmpVector, xvector, recv[rank]);
        if (!rank) {
            (*iter_count)++;
        }
    }
    delete[] tmpVector;

    MPI_Allgatherv(xvector, recv[rank], MPI_DOUBLE, rezult, recv, displs, MPI_DOUBLE, MPI_COMM_WORLD);
}
void set_recvs_and_displs(int *recv, int *displs, int size) {
    int initialValue = N / size;
    int rest = N % size;
    for (int i = 0; i < size; i++) {
        recv[i] = initialValue;
    }
    int i = 0;
    while (rest--) {
        recv[i]++;
        i++;
    }
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i - 1] + recv[i - 1];
    }

}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);      /* starts MPI */

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);        /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &size);        /* get number of processes */


    if (size > N) {
        std::cout << "Too much process!!";
        return -1;
    }

    auto *recv = new int[size];
    auto *displs = new int[size];


    set_recvs_and_displs(recv, displs, size);


    auto matrix = new double[recv[rank] * N];
    auto partOfBvector = new double[recv[0]];
    auto partOfXVector = new double[recv[0]];
    auto partOfYVector = new double[recv[0]];

    std::fill_n(partOfBvector,recv[rank],0);
    std::fill_n(partOfYVector,recv[rank],0);
    std::fill_n(partOfXVector,recv[rank],0);

    size_t counter = 0;
    auto rezult = new double[N];

    processInit(matrix, partOfBvector, "matrix.txt", "vector.txt", recv, displs, rank);
    algorithm(matrix, partOfXVector, partOfBvector, partOfYVector, rezult, recv, displs, rank, size, &counter);

    if (!rank) {
        for (int i = 0; i < N; i++) {
            std::cout << rezult[i] << " ";
        }
        std::cout<<std::endl<<"iterations :"<<counter;
    }
    MPI_Finalize();
    return 0;
}