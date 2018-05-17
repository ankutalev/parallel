#include <iostream>
#include <mpi.h>
#include <cstdint>
#include <fstream>
#include <cmath>
#define Nx 10
#define Ny 10
#define Nz 10
#define xLen 2
#define yLen 2
#define zLen 2
#define x0 (0)
#define y0 (0)
#define z0 (0)
#define h(a,b) ((double)(b)/((a)-1))
#define A 100000
#define EPSILON 0.000001
#define xCOORD(a) ((x0) + ((a)*(h(Nx,xLen))))
#define yCOORD(a) ((y0) + (a)*(h(Ny,yLen)))
#define zCOORD(a) ((z0) + (a)*(h(Nz,zLen)))

double Ro (double x, double y, double z) {
    return 6 - A*(x*x + y*y + z*z);
}

double calculateNode(int x, int y, int z, int height, double oldNode[Nx][Ny][Nz], double recvEdge,int rank) {
    double tmp1, tmp2, tmp3;
    if (x == Nx -1 || y == Ny-1 || x == 0 || y ==0)
        return oldNode[x][y][z];
    if (z> 0 && z < height-1) {
        tmp1= (oldNode[x-1][y][z] + oldNode[x+1][y][z])/(h(Nx,xLen)*h(Nx,xLen));
        tmp2= (oldNode[x][y-1][z] + oldNode[x][y+1][z])/(h(Ny,yLen)*h(Ny,yLen));
        tmp3= (oldNode[x][y][z-1] + oldNode[x][y][z+1])/(h(Nz,zLen)*h(Nz,zLen));
        return (tmp1+tmp2+tmp3 -Ro(xCOORD(x),yCOORD(y),zCOORD(z+height*rank)) ) / ((double)2/ (h(Nx,xLen)*h(Nx,xLen)) +A+ (double)2/ (h(Ny,yLen)*h(Ny,yLen)) +(double)2/ (h(Nz,zLen)*h(Nz,zLen)));
    }

    if (z==0) {
        tmp1= (oldNode[x-1][y][z] + oldNode[x+1][y][z])/(h(Nx,xLen)*h(Nx,xLen));
        tmp2= (oldNode[x][y-1][z] + oldNode[x][y+1][z])/(h(Ny,yLen)*h(Ny,yLen));
        tmp3= (oldNode[x][y][z+1] + recvEdge)/(h(Nz,zLen)*h(Nz,zLen));
        return (tmp1+tmp2+tmp3-Ro(xCOORD(x),yCOORD(y),zCOORD(z+height*rank))) / ((double)2/ (h(Nx,xLen)*h(Nx,xLen)) +A+ (double)2/ (h(Ny,yLen)*h(Ny,yLen)) +(double)2/ (h(Nz,zLen)*h(Nz,zLen)));
    }
    if (z==height-1) {
        tmp1= (oldNode[x-1][y][z] + oldNode[x+1][y][z])/(h(Nx,xLen)*h(Nx,xLen));
        tmp2= (oldNode[x][y-1][z] + oldNode[x][y+1][z])/(h(Ny,yLen)*h(Ny,yLen));
        tmp3= (oldNode[x][y][z-1] + recvEdge)/(h(Nz,zLen)*h(Nz,zLen));
        return (tmp1+tmp2+tmp3-Ro(xCOORD(x),yCOORD(y),zCOORD(z+height*rank))) / ((double)2/ (h(Nx,xLen)*h(Nx,xLen)) +A+ (double)2/ (h(Ny,yLen)*h(Ny,yLen)) +(double)2/ (h(Nz,zLen)*h(Nz,zLen)));
    }
}


void fillEdges(double myLowEdges[Nx][Ny], double myUpEdges[Nx][Ny],double rcvLowEdges[Nx][Ny],double rcvUpEdges [Nx][Ny],int height, double nodes[Nx][Ny][Nz],int rank,int size) {

    for (int x = 0; x < Nx; ++x) {
        for (int y = 0; y < Ny; ++y) {
            if (rank)
                nodes[x][y][0] = calculateNode(x,y,0,height,nodes,rcvLowEdges[x][y],rank);
            if (rank!=size-1)
                nodes[x][y][height-1] = calculateNode(x,y,height-1,height,nodes,rcvUpEdges[x][y],rank);
            myLowEdges[x][y] = nodes[x][y][0];
            myUpEdges[x][y] = nodes[x][y][height-1];
        }
    }
}


void calcCenter(double nodes[Nx][Ny][Nz], int height, int rank) {
    for (int z = 1; z < height-1; ++z) {
        for (int x = 0; x < Nx; ++x) {
            for (int y = 0; y < Ny; ++y) {
                nodes[x][y][z] = calculateNode(x, y, z, height, nodes, 0,rank);
            }
        }
    }
}




void sendEdges (double myLowEdges[Nx][Ny], double myUpEdges[Nx][Ny],double rcvLowEdges[Nx][Ny],double rcvUpEdges[Nx][Ny],MPI_Request* requests,int rank,int size) {
    if (rank!=0) {
        MPI_Isend(myLowEdges, Nx * Ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(rcvLowEdges,Nx*Ny,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD,&requests[1]);
    }
    if (rank!=size-1) {
        MPI_Isend(myUpEdges, Nx * Ny, MPI_DOUBLE, rank +1, 0, MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(rcvUpEdges,Nx*Ny,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD,&requests[3]);
    }
}

void getEdges(MPI_Request* requests,int rank, int size) {
    if (rank) {
        MPI_Wait(requests,MPI_STATUS_IGNORE);
        MPI_Wait(&requests[1],MPI_STATUS_IGNORE);
    }
    if (rank!=size-1) {
        MPI_Wait(&requests[2],MPI_STATUS_IGNORE);
        MPI_Wait(&requests[3],MPI_STATUS_IGNORE);
    }
}

bool cond(double nodes[Nx][Ny][Nz], double oldNodes[Nx][Ny][Nz], int height) {

    double max = fabs(nodes[0][0][0] - oldNodes[0][0][0]);

    for (int x = 0; x < Nx; ++x) {
        for (int y = 0; y < Ny; ++y) {
            for (int z = 0; z < height; ++z) {
                if (fabs(nodes[x][y][z]-oldNodes[x][y][z]) > max)
                    max = fabs(nodes[x][y][z]-oldNodes[x][y][z]);
                oldNodes[x][y][z]=nodes[x][y][z];
            }
        }
    }
    double globalMax;
    MPI_Allreduce(&max,&globalMax,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

    return globalMax>EPSILON;
}


double F (double x , double y, double z) {
    return x*x+ y*y + z*z;
}

int main(int argc, char* argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int height = Nz / size;


    if (Nz % size ) {
        for (int i = 0; i < Nz%size; ++i) {
            if(rank==i) {
                height++;
            }
        }
    }
    double oldNodes[Nx][Ny][Nz] = {0};
    double Nodes[Nx][Ny][Nz];


    double myLowEdges [Nx][Ny] = {0};
    double myUpEdges[Nx][Ny] = {0};
    double rcvLowEdges [Nx][Ny] = {0};
    double rcvUpEdges [Nx][Ny] = {0};


    MPI_Request requests [4];


    for (auto &Node : Nodes) {
        for (auto &y : Node) {
            for (int z = 0; z < height; ++z) {
                y[z] = 0;
            }
        }
    }

    for (int z = 0; z < height; ++z) {

        for (int x = 0; x < Nx; ++x) {
            Nodes[x][0][z] = F(xCOORD(x),yCOORD(0),zCOORD(z+height*rank));
            oldNodes[x][0][z]= Nodes[x][0][z];
            Nodes[x][Ny-1][z]= F(xCOORD(x),yCOORD(Ny-1),zCOORD(z+height*rank));
            oldNodes[x][Ny-1][z]= Nodes[x][Ny-1][z];
        }

        for (int y = 0; y < Ny; ++y) {
            Nodes[0][y][z] = F(xCOORD(0),yCOORD(y),zCOORD(z+height*rank));
            oldNodes[0][y][z]= Nodes[0][y][z];
            Nodes[Nx-1][y][z] = F(xCOORD(Nx-1),yCOORD(y),zCOORD(z+height*rank));
            oldNodes[Nx-1][y][z]= Nodes[Nx-1][y][z];
        }
    }

    if(!rank) {
        for (int x = 0; x < Nx; ++x) {
            for (int y = 0; y < Ny; ++y) {
                Nodes[x][y][0] = F(xCOORD(x),yCOORD(y),zCOORD(0));
                oldNodes[x][y][0]= Nodes[x][y][0];
            }
        }
    }

    if (rank==size-1) {
        for (int x = 0; x < Nx; ++x) {
            for (int y = 0; y < Ny; ++y) {
                Nodes[x][y][height-1] = F(xCOORD(x),yCOORD(y),zCOORD((height-1)+rank*height));
                oldNodes[x][y][height-1]= Nodes[x][y][height-1];
            }
        }
    }
    size_t  i = 0;


    do {
        sendEdges(myLowEdges, myUpEdges, rcvLowEdges, rcvUpEdges, requests, rank, size);


        ;
        calcCenter(Nodes, height, rank);


        getEdges(requests, rank, size);

        fillEdges(myLowEdges, myUpEdges, rcvLowEdges, rcvUpEdges, height, Nodes, rank, size);

        i++;

    }while (cond(Nodes, oldNodes, height));


    if (rank==0) {
            std::cout<<i<<std::endl<<std::endl;
        for (int z = 0; z < height; ++z) {
            std::cout<<"NA VISOTE Z = "<<z<<std::endl;
            for (int y = 0; y < Ny; ++y) {
                for (auto &Node : Nodes) {
                    std::cout<< Node[y][z]<<" ";
                }
                std::cout<<std::endl;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}

