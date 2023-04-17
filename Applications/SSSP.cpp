#include "CombBLAS/CombBLAS.h"
#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>

double cblas_alltoalltime;
double cblas_allgathertime;
double cblas_mergeconttime;
double cblas_transvectime;
double cblas_localspmvtime;

using namespace std;
using namespace combblas;

// graph500 params
#define EDGEFACTOR 16
MTRand GlobalMT(123); // for reproducable result

// type definitions
// typedef SpParMat<int64_t, int64_t, SpDCCols<int64_t, int64_t>> PSpMat_Int64;

// SSSP configs
#define MAX_DIST std::numeric_limits<int>::max()
using Vec = FullyDistVec<int64_t, int64_t>;

void SSSP(SpParMat<int64_t, int64_t, SpDCCols<int64_t, int64_t>> &A, int64_t source, Vec &dist)
{
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // graph info
    int64_t n = A.getnrow();
    cout << "n = " << n << endl;

    // initialize distance vector
    dist = Vec(A.getcommgrid(), n, MAX_DIST); // init distances to MAX_DIST
    dist.SetElement(source, 0);               // set source distance to 0

    Vec ones = Vec(A.getcommgrid(), n, 1);

    cout << "Starting SSSP ..." << endl;

    // bellman-ford algorithm

    for (int64_t i = 0; i < n; i++)
    {
        // For each edge (u, v) with weight w, update dist[v] = min(dist[v], dist[u] + w)

        // dist = A * dist (relaxation step to update distances)
        // @manish: WIP -- can't find the right SpMV function!!!!
        SpMV(A, dist, dist, false);
    }

    // print shortest path distances
    for (int i = 0; i < n; i++)
    {
        cout << "Shortest distance from " << source << " to " << i << " is " << dist[i] << endl;
    }
}

int main(int argc, char *argv[])
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // if(argc < 2){
    //     if(myrank == 0){
    //         cout << "Usage: ./sssp <scale> <source> " << endl;
    //         cout << "Example: mpirun -np 4 ./sssp 3 5" << endl;
    //     }
    //     MPI_Finalize();
    //     return -1;
    // }
    {
        int n = 6;
        int source = 0;

        SpParMat<int64_t, int64_t, SpDCCols<int64_t, int64_t>> A; // adjacency matrix (undirected graph)
        Vec dist;                                                 // distance vector from source to all nodes

        unsigned scale = 3; // 2^scale vertices
        double initiator[4] = {.57, .19, .19, .05};

        cout << "[Graph500] generating random graph ..." << endl;

        double t01 = MPI_Wtime();
        double t02;

        DistEdgeList<int64_t> *DEL = new DistEdgeList<int64_t>();
        DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, true);
        MPI_Barrier(MPI_COMM_WORLD);

        t02 = MPI_Wtime();
        ostringstream tinfo;
        tinfo << "Generation took " << t02 - t01 << " seconds" << endl;

        SpParMat<int64_t, int64_t, SpDCCols<int64_t, int64_t>> *ABool = new SpParMat<int64_t, int64_t, SpDCCols<int64_t, int64_t>>(*DEL, false);
        delete DEL;
        int64_t removed = ABool->RemoveLoops();
        ABool->PrintInfo();

        SSSP(*ABool, source, dist);
    }

    MPI_Finalize();
    return 0;
}