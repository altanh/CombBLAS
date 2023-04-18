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

#define EDGEFACTOR 16
MTRand GlobalMT(123);

#define MAX_DIST std::numeric_limits<double>::max()
using Mat = SpParMat<int64_t, double, SpDCCols<int64_t, double>>;
using Vec = FullyDistVec<int64_t, double>;

// print the SpParMat
template <typename IT, typename NT>
void printSpParMat(SpParMat<IT, NT, SpDCCols<IT, NT>> &A)
{
    // get local matrix
    SpDCCols<IT, NT> localMat = A.seq();
    int count = 0;
    int total = 0;

    // temporary vector of vector to store the matrix
    vector<vector<NT>> temp(A.getnrow(), vector<NT>(A.getncol(), 0));

    // use SpColIter to iterate over cols of local matrix
    for (SpDCCols<int64_t, double>::SpColIter colit = A.seq().begcol(); colit != A.seq().endcol(); ++colit)
    {
        for (SpDCCols<int64_t, double>::SpColIter::NzIter nzit = A.seq().begnz(colit); nzit != A.seq().endnz(colit); ++nzit)
        {
            count++;
            temp[nzit.rowid()][colit.colid()] = nzit.value();
        }
    }

    // print the temp vector
    for (auto row : temp)
    {
        for (auto col : row)
        {
            cout << setw(5) << col << " ";
        }
        cout << endl;
    }
}



void SSSP(Mat &A, int64_t source, Vec &dist)
{
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int64_t n = A.getnrow();
    cout << "n = " << n << endl;

    dist = Vec(A.getcommgrid(), n, MAX_DIST); // init distances to MAX_DIST
    dist.SetElement(source, 0);               // set source distance to 0

    cout << "Starting SSSP ..." << endl;

    // bellman-ford
    for (int64_t i = 0; i < n - 1; i++)
    {
        // printSpParMat(A);
        // dist.DebugPrint();
        // @manish: A is sparse but dist is dense. So runs a dense SpMV.
        dist = SpMV<MinPlusSRing<double, double>>(A, dist);
    }

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
    // TODO: Add loaders for binary graph files
    {
        int source = 0;

        unsigned scale = 10; // 2^scale vertices
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

        // adjacency matrix
        Mat *A = new Mat(*DEL, false);
        delete DEL;
        int64_t removed = A->RemoveLoops();

        // distance vector from source to all nodes
        Vec dist;
        A->Transpose();

        // set diagonal to 0
        A->AddLoops(0, true);

        SSSP(*A, source, dist);
    }

    MPI_Finalize();
    return 0;
}