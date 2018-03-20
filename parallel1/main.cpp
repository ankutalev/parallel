#include <iostream>
#include <mpi/mpi.h>
#include <fstream>
#include <cmath>

#define N 12

void processInit (double* matrixPart,double* vectorPart,const std::string& matrixFileName,const std::string& vectorFileName, int rank,int size) {

    int stringsInOneProcess = N/size;
    std::ifstream fin;
    fin.open(matrixFileName);
    std::ofstream fout;
    fout.open(std::to_string(rank));
    std::string tmp;
    int i = 0;
    for ( i = 0; i < rank*stringsInOneProcess; i++) {
        fin.ignore(256,'\n');
    }
    for (i = 0; i < N*stringsInOneProcess;i++) {
        fin>>matrixPart[i];
        fout<<matrixPart[i]<<" ";
    }

    if (rank == size-1) {
        for (int j = 0; j < N % size * N;j++,i++) {
            fin >> matrixPart[i];
            fout << matrixPart[i]<<" ";
        }
    }
    fout<<std::endl;
    fin.close();

    fin.open(vectorFileName);
    for (int i = 0; i< rank*stringsInOneProcess;i++) {
        fin.ignore(256,' ');
    }

    for (i = 0 ; i< stringsInOneProcess;i++) {
        fin >> vectorPart[i];
        fout << vectorPart[i] << " ";
    }
    if (rank == size-1) {
        for (int j=0; j < N % size;j++,i++) {
            fin >> vectorPart[i];
            fout << vectorPart[i] << " ";
        }
    }
}

void multiply(const double* matrix, double* vector,double* rez,int rank,int size) {

    int stringsInOneProcess = N / size;
    MPI_Status st;
    int startPosition = rank * stringsInOneProcess;
    if (rank == size - 1)
        stringsInOneProcess += N % size;
    int vectorCellsInOneProcess = stringsInOneProcess;

    for (int i = 0; i < N/size + N % size; i++)
        rez[i] = 0;


    for (int j = 0; j < size; j++) {
        for (int k = 0; k < stringsInOneProcess; k++) {
            for (int i = 0; i < vectorCellsInOneProcess; i++) {
                rez[k] += vector[i] * matrix[i + startPosition + N * k];
            }
        }

        if (size != 1) {
            MPI_Sendrecv(&vectorCellsInOneProcess, 1, MPI_INT, (rank + 1) % size, 10, &vectorCellsInOneProcess, 1, MPI_INT,
                         (rank - 1 + size) % size, 10, MPI_COMM_WORLD, &st);
            MPI_Sendrecv(&startPosition, 1, MPI_INT, (rank + 1) % size, 10, &startPosition, 1, MPI_INT,
                         (rank - 1 + size) % size, 10, MPI_COMM_WORLD, &st);
            MPI_Sendrecv(vector, N / size + N % size, MPI_DOUBLE, (rank + 1) % size, 10, vector, N / size + N % size,
                         MPI_DOUBLE, (rank - 1 + size) % size, 10, MPI_COMM_WORLD, &st);
        }
    }

}


void vectorSub(double* fvector, double* svector, double* rezvector,int vectorSize) {
    for (int i = 0; i < vectorSize;i++)
        rezvector[i] = fvector[i]-svector[i];
}

void scalarMultiply(double* vector,double scalar,double * rez,int vectorSize) {
    for (int i = 0; i < vectorSize; ++i) {
        rez[i]=vector[i]*scalar;
    }
}

double parametherCalcuation (const double* fvector, const double* svector,int vectorSize,int rank,int size) {
    double a=0,b=0;
    double total = 0;
    double allA=0, allB = 0;
    for (int i = 0; i< vectorSize;i++) {
        a+=fvector[i]*svector[i];
        b+=svector[i]*svector[i];
    }
    MPI_Reduce(&a,&allA,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&b,&allB,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    if (!rank) {
        total = allA / allB;
    }
    MPI_Bcast(&total,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        return total;
}

double getSummOfPartOfVector(const double* partOfVector,int stringsInOneProcess) {
    double sum = 0;
    for (int i = 0; i < stringsInOneProcess; ++i) {
        sum+= partOfVector[i]*partOfVector[i];
    }
    return sum;
}

bool algoCondition(const double* matrix,double* partOfXVector,double* partOfBVector,int partOfVectorSize,int rank,int size) {
    auto tmp = new double[N/ size + N%size];
    double epsilon = 0.001;
    double NumeratorNorm = 0;
    double DenominatorNorm = 0;
    bool cond;
    multiply(matrix,partOfXVector,tmp,rank,size);
    vectorSub(tmp,partOfBVector,tmp,partOfVectorSize);
    double partOfNumeratorNorm =  getSummOfPartOfVector(tmp,partOfVectorSize);
    //for (int i = 0; i < partOfVectorSize ; ++i) {
        //std::cout<<partOfNumeratorNorm<<std::endl;
    //}
    double partOfDenominatorNorm = getSummOfPartOfVector(partOfBVector,partOfVectorSize);
//    MPI_Reduce(&b,&allB,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
//    if (!rank)
//        total = allA/allB;
//    MPI_Bcast(&total,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Reduce(&partOfNumeratorNorm,&NumeratorNorm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&partOfDenominatorNorm,&DenominatorNorm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    if (!rank) {

        cond = sqrt(NumeratorNorm)/sqrt(DenominatorNorm) > epsilon;
    }
    MPI_Bcast(&cond,1,MPI_CXX_BOOL,0,MPI_COMM_WORLD);
    //todo: return mpi reduce all
//    double sum = getSummOfPartOfVector(tmp,partOfVectorSize);
  //  MPI_Allgather(tmp,partOfVectorSize,MPI_DOUBLE,checkVector,partOfVectorSize,MPI_DOUBLE,MPI_COMM_WORLD);
  //  MPI_Allgather(partOfBVector,partOfVectorSize,MPI_DOUBLE,fullBVector,partOfVectorSize,MPI_DOUBLE,MPI_COMM_WORLD);
  //  bool cond = (normOfFullVector(checkVector)/ normOfFullVector(fullBVector) > epsilon);
//    if (!rank) {
//               std::cout<<"YA OTNOWENIE NORM "<<normOfFullVector(checkVector)/ normOfFullVector(fullBVector)<<std::endl;
//
//        for (int i = 0; i<partOfVectorSize;i++)
//            std::cout<< partOfXVector[i]<<" ";
//        std::cout<<std::endl;
//    }
    delete[] tmp;
    //delete[] fullBVector;
    //delete[] checkVector;
    return cond;
}

void algo(double* matrix, double* xvector, double* bvector,double* yvector,double* rezult,int rank,int size) {

    int stringsInOneProcess = N / size;

    int recv[size];
    int displs[size];

    for (int i = 0; i< size; i++) {
        recv[i] = stringsInOneProcess;
        displs[i] = i*stringsInOneProcess;
    }

    recv[size-1]+= (N % size);

    if (rank==size-1) {
        stringsInOneProcess += (N % size);
    }

    auto tmpVector = new double[N/size + N%size];

    double methodParamether;
    MPI_Status st;
    while (algoCondition(matrix, xvector, bvector, stringsInOneProcess, rank, size)) {

        multiply(matrix, xvector, tmpVector, rank, size);

        vectorSub(tmpVector, bvector, yvector, stringsInOneProcess);



//    //if (rank==size-1)
//      //  std::cout<<*yvector<<"   "<<*(yvector+1);
          multiply(matrix, yvector, tmpVector, rank, size);
          methodParamether = parametherCalcuation(yvector, tmpVector, stringsInOneProcess, rank,size);
          scalarMultiply(yvector, methodParamether, tmpVector, stringsInOneProcess);
//////        //if (!rank)
//////        //    std::cout<<methodParamether<<std::endl;
//////            //std::cout<< *tmpVector<<"  "<< *(tmpVector+1)<<std::endl;
        vectorSub(xvector, tmpVector, xvector, stringsInOneProcess);
//// //        if (!rank) {
//// //            for (int i = 0; i < N; ++i) {
//// //                std::cout << rezult[i] << " ";
//// //            }
////  //            std::cout<<std::endl;
////  //        }
////    }

//        if (rank==size-1) {
//            for (int i = 0; i < stringsInOneProcess; i++) {
//                std::cout << xvector[i] << " ";
//                std::cout << std::endl;
//
//            }
//            std::cout << " PARAMETR= " << methodParamether << std::endl;
//        }

//    for (int i = 0; i < size;i++ ) {
//        if (i==rank) {
//            std::cout<<"RANK  = "<<i<<"  vectorcesll = "<<stringsInOneProcess;//<<std::endl;
//            for (int j = 0; j < stringsInOneProcess;j++)
//                std::cout<<" "<<yvector[j]<<" ";
//        }
//        std::cout<<std::endl;
    }

//    if (rank==size-1)
//        for (int j = 0; j < stringsInOneProcess;j++)
//            std::cout<<" "<<yvector[j]<<" ";

   //здесь  первый вектор - хвектор
    //MPI_Allgather(yvector, N/size + N%size, MPI_DOUBLE, rezult, N/size + N%size, MPI_DOUBLE, MPI_COMM_WORLD);
     delete[] tmpVector;
     MPI_Allgatherv(xvector,stringsInOneProcess,MPI_DOUBLE,rezult,recv,displs,MPI_DOUBLE,MPI_COMM_WORLD);


}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);      /* starts MPI */

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);        /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &size);        /* get number of processes */

    if (size>N) {
        std::cout<<"Too much process!!";
        return  -1;
    }

    int stringsInLastProcess = N / size + N % size;



    auto matrix = new double[(rank==size-1) ? N* stringsInLastProcess : N * N / size]; //?????? todo: wtf dude
    auto partOfBvector = new double[stringsInLastProcess];
    auto partOfXVector = new double[stringsInLastProcess];
    auto partOfYVector = new double[stringsInLastProcess];

    for (int i = 0; i < stringsInLastProcess; ++i) {
        partOfXVector[i] = 0;
    }

    auto rezult = new double[N];
    processInit(matrix, partOfBvector, "matrix12.txt", "vector12.txt", rank, size);
    algo(matrix, partOfXVector, partOfBvector, partOfYVector, rezult, rank, size);

//    if (rank==size-1)
//        for (int i = 0; i < stringsInLastProcess* N; i++) {
//            std::cout<<matrix[i]<<" ";
//        }

    if (!rank ) {
        for (int i = 0; i < N; i++) {
               std::cout<<rezult[i]<<" ";
        }
    }

    MPI_Finalize();
    return 0;
}