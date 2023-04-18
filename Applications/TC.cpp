

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

using Mat = SpParMat<int64_t, int64_t, SpDCCols<int64_t, int64_t>>;
using Vec = FullyDistVec<int64_t, int64_t>;

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
    for (SpDCCols<int64_t, int64_t>::SpColIter colit = A.seq().begcol(); colit != A.seq().endcol(); ++colit)
    {
        for (SpDCCols<int64_t, int64_t>::SpColIter::NzIter nzit = A.seq().begnz(colit); nzit != A.seq().endnz(colit); ++nzit)
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


void TC(Mat &A, Vec &deg, Vec &triangles)
{
    // @manish: assumes undirected graph
    // note: can be updated for directed by computing in and out degrees separately
    // computing results, and then adding them up.

    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int64_t n = A.getnrow();
    cout << "n = " << n << endl;

    // convert adjacency matrix to unweighted
    A.Apply([](int64_t x) { return 1; });

    triangles = Vec(A.getcommgrid(), n, 0);
    deg = Vec(A.getcommgrid(), n, 0);

    // deg = Reduce(A, plus)
    deg = A.Reduce(Column, plus<int64_t>(), static_cast<int64_t>(0));

    // twos = deg * deg
    Vec twos = deg;
    twos.Apply([](int64_t x) { return x * x; });

    // triangles = SpMV(A, twos)
    triangles = SpMV<PlusTimesSRing<int64_t, int64_t>>(A, twos);
    
    // result = Reduce(triangles, plus)
    int64_t result = triangles.Reduce(plus<int64_t>(), static_cast<int64_t>(0));

    // printSpParMat(A);
    // deg.DebugPrint();
    // triangles.DebugPrint();

    // divide by 6 because each triangle is counted 3 times
    if (myrank == 0)
    {
        cout << "triangles = " << result/6 << endl;
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

        Vec deg; // degree
        Vec triangles; // triangles per vertex

        TC(*A, deg, triangles);
    }

    MPI_Finalize();
    return 0;
}