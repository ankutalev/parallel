#include <iostream>
#include <mpi/mpi.h>
#include <fstream>
#include <cmath>

#define N 10

void processInit(double *matrixPart, double *vector, const std::string &matrixFileName,
                 const std::string &vectorFileName, int rank, int size) {

    int stringsInOneProcess = N / size;
    std::ifstream fin;
    fin.open(matrixFileName);
    std::ofstream fout;
    fout.open(std::to_string(rank));
    std::string tmp;
    int i = 0;
    for (i = 0; i < rank * stringsInOneProcess; i++) {
        fin.ignore(256, '\n');
    }
    for (i = 0; i < N * stringsInOneProcess; i++) {
        fin >> matrixPart[i];
        fout << matrixPart[i] << " ";
    }

    if (rank == size - 1) {
        for (int j = 0; j < N % size * N; j++, i++) {
            fin >> matrixPart[i];
            fout << matrixPart[i] << " ";
        }
    }
    fout << std::endl;
    fin.close();

    fin.open(vectorFileName);
//    for (int i = 0; i < rank * stringsInOneProcess; i++) {
//        fin.ignore(256, ' ');
//    }

    for (i = 0; i < N; i++) {
        fin >> vector[i];
        fout << vector[i] << " ";
    }
}

void multiply(const double *matrix, double *vector, double *rez, int rank, int size) {

    int stringsInOneProcess = N / size;

    if (rank == size - 1)
        stringsInOneProcess += N % size;

    for (int i = 0; i < N / size + N % size; i++)
        rez[i] = 0;

        for (int k = 0; k < stringsInOneProcess; k++) {
            for (int i = 0; i < N; i++) {
                rez[k] += vector[i] * matrix[i + N * k];
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

double parametherCalcuation(const double *fvector, const double *svector, int vectorSize, int rank) {
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

double getSummOfVector(const double *vector, int stringsInOneProcess) {
    double sum = 0;
    for (int i = 0; i < stringsInOneProcess; ++i) {
        sum += vector[i] * vector[i];
    }
    return sum;
}

bool algoCondition(const double *matrix, double *xvector, double *bvector, int partOfVectorSize, int rank,
                   int size) {
    auto tmp = new double[partOfVectorSize];
    double epsilon = 0.001;
    double NumeratorNorm = 0;
    static double DenominatorNorm = 0;
    bool cond;

    multiply(matrix, xvector, tmp, rank, size);
    vectorSub(tmp, bvector, tmp, partOfVectorSize);
    double partOfNumeratorNorm = getSummOfVector(tmp, partOfVectorSize);
    static int lock = 0;
    if (!lock) {
        DenominatorNorm = getSummOfVector(bvector, N);
        lock++;
    }

    MPI_Reduce(&partOfNumeratorNorm, &NumeratorNorm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (!rank) {
        cond = sqrt(NumeratorNorm) / sqrt(DenominatorNorm) > epsilon;
    }
    MPI_Bcast(&cond, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    delete[] tmp;
    return cond;
}

void algo(double *matrix, double *xvector, double *bvector, double *yvector, double *rezult, int rank, int size) {

    int stringsInOneProcess = N / size;
    if (rank == size - 1) {
        stringsInOneProcess += (N % size);
    }

    int recv[size];
    int displs[size];

    for (int i = 0; i < size; i++) {
        recv[i] = stringsInOneProcess;
        displs[i] = i * stringsInOneProcess;
    }

    recv[size - 1] += (N % size);

    auto tmpVector = new double[stringsInOneProcess];
    double methodParamether;
    while (algoCondition(matrix, xvector, bvector, stringsInOneProcess, rank, size)) {
        multiply(matrix, xvector, tmpVector, rank, size);
        vectorSub(tmpVector, bvector, yvector, stringsInOneProcess);
        multiply(matrix, yvector, tmpVector, rank, size);
        methodParamether = parametherCalcuation(yvector, tmpVector, stringsInOneProcess, rank);
        scalarMultiply(yvector, methodParamether, tmpVector, stringsInOneProcess);
        vectorSub(xvector, tmpVector, xvector, stringsInOneProcess);
    }
    delete[] tmpVector;
    MPI_Allgatherv(xvector, stringsInOneProcess, MPI_DOUBLE, rezult, recv, displs, MPI_DOUBLE, MPI_COMM_WORLD);
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

    int stringsInOneProcess =  N / size;

    if (rank ==size-1)
        stringsInOneProcess+= N % size;


    auto matrix = new double[(rank == size - 1) ? N * stringsInOneProcess : N * N / size];
    auto bvector = new double[N];
    auto xvector = new double[N];
    auto partOfYVector = new double[stringsInOneProcess];

    for (int i = 0; i < stringsInOneProcess; ++i) {
        xvector[i] = 0;
    }

    auto rezult = new double[N];
    processInit(matrix,bvector, "matrix.txt", "vector.txt", rank, size);
    algo(matrix, xvector, bvector, partOfYVector, rezult, rank, size);

    if (!rank) {
        for (int i = 0; i < N; i++) {
            std::cout << rezult[i] << " ";
        }
        std::cout<<std::endl<<"heh";
    }

    MPI_Finalize();
    return 0;
}
