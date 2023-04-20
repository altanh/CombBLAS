

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

// symmetricize the matrix
template <typename PARMAT>
void Symmetricize(PARMAT &A)
{
    PARMAT AT = A;
    AT.Transpose();
    A += AT;
}

// get lower triangular matrix
// @manish: is there a faster way to do this?
template <typename PARMAT>
PARMAT GetLowerTriangular(PARMAT &A)
{
    PARMAT L = A;
    for (SpDCCols<int64_t, int64_t>::SpColIter colit = L.seq().begcol(); colit != L.seq().endcol(); ++colit)
    {
        for (SpDCCols<int64_t, int64_t>::SpColIter::NzIter nzit = L.seq().begnz(colit); nzit != L.seq().endnz(colit); ++nzit)
        {
            if (nzit.rowid() < colit.colid())
            {
                nzit.value() = 0;
            }
        }
    }
    return L;
}

void TC(Mat &A)
{
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int64_t n = A.getnrow();
    cout << "n = " << n << endl;

    // convert adjacency matrix to unweighted and undirected
    A.Apply([](int64_t x)
            { return 1; });
    Symmetricize(A);
    // printSpParMat(A);

    // L = tril(A)
    Mat L = GetLowerTriangular(A);
    // printSpParMat(L);

    // C = (L * L) .* L
    Mat Ltemp = L;
    Mat C = Mult_AnXBn_Synch<PlusTimesSRing<int64_t, int64_t>, int64_t, SpDCCols<int64_t, int64_t>>(L, Ltemp);
    C.EWiseMult(L, false);
    // printSpParMat(C);

    // reduce C to get number of triangles
    Vec triangles = C.Reduce(Column, plus<int64_t>(), static_cast<int64_t>(0));
    int64_t result = triangles.Reduce(plus<int64_t>(), static_cast<int64_t>(0));

    if (myrank == 0)
    {
        cout << "triangles = " << result << endl;
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

        TC(*A);
    }

    MPI_Finalize();
    return 0;
}