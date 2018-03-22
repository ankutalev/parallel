#include <iostream>
#include <fstream>
#include <cmath>

#define N 12

void multiply(const double* matrix, double* vector, double*rezult) {
#pragma omp parallel for
    for (int i = 0; i< N; i++) {
        for (int j = 0; j< N;j++)
            rezult[i] += vector[j]* matrix[j + N*i];
    }
}

void sub(double* fvector, double* svector, double* rezult) {
#pragma omp parallel for
    for (int j = 0; j< N;j++)
        rezult[j] = fvector[j] -  svector[j];
}

void scalarMultiply(double *vector, double scalar, double *rez) {
#pragma omp paralell for
    for (int i = 0; i < N; ++i) {
        rez[i] = vector[i] * scalar;
    }
}

double parametherCalcuation(const double *fvector, const double *svector) {
    double a = 0, b = 0;
#pragma omp paralell for reduction(+:a,+:b)
    for (int i = 0; i < N; i++) {
        a += fvector[i] * svector[i];
        b += svector[i] * svector[i];
    }
    return a/b;
}
double getSummOfPartOfVector(const double *partOfVector) {
    double sum = 0;
#pragma omp paralell for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += partOfVector[i] * partOfVector[i];
    }
    return sum;
}

bool algoCondition(const double *matrix, double *xvector, double *bvector) {
    auto tmp = new double[N];
    for (int i = 0;i<N;i++)
        tmp[i] = 0;
    const  double epsilon = 0.001;
    multiply(matrix, xvector, tmp);
    sub(tmp, bvector, tmp);
    double Numerator = getSummOfPartOfVector(tmp);
    double Denominator = getSummOfPartOfVector(bvector);

    delete[] tmp;
    return sqrt(Numerator/Denominator) > epsilon;
}

void algo(double *matrix, double *xvector, double *bvector, double *yvector) {

    auto tmpVector = new double[N];
    double methodParamether;
    while (algoCondition(matrix, xvector, bvector)) {
        multiply(matrix, xvector, tmpVector);
        sub(tmpVector, bvector, yvector);
        multiply(matrix, yvector, tmpVector);
        methodParamether = parametherCalcuation(yvector, tmpVector);
        scalarMultiply(yvector, methodParamether, tmpVector);
        sub(xvector, tmpVector, xvector);
    }
    delete[] tmpVector;
}

int main() {

    std::cout << "Hello, World!" << std::endl;
    std::ifstream fin;
    fin.open("matrix12.txt");

    auto matrix = new double[N*N];
    auto bvector = new double[N];
    auto yvector = new double[N];
    auto xvector = new double[N];
    for (int j = 0; j < N*N; ++j) {
        fin>>matrix[j];
    }

    fin.close();
    fin.open("vector12.txt");
    for (int j = 0; j < N; ++j) {
        fin>>bvector[j];
        xvector[j]=0;
    }
    algo(matrix,xvector,bvector,yvector);

    for (int j = 0; j < N; ++j) {
        std::cout<<xvector[j]<<" ";
    }

    fin.close();


    return 0;
}