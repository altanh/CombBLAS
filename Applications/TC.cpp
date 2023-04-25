

#include "CombBLAS/CombBLAS.h"
#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>

#include "DGB.h"

using namespace std;
using namespace combblas;

#define EDGEFACTOR 16
MTRand GlobalMT(123);

using Mat = SpParMat<int64_t, int64_t, SpDCCols<int64_t, int64_t>>;
using Vec = FullyDistVec<int64_t, int64_t>;

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
void to_tril(PARMAT *A)
{
    A->PruneI([](const std::tuple<int64_t, int64_t, int64_t> &t){ return std::get<0>(t) < std::get<1>(t); }, true);
}

void TC(Mat &L, Timer *timer)
{
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // C = (L * L) .* L
    timer->reset("copy_L");
    Mat Ltemp = L;
    timer->elapsed();

    timer->reset("spgemm");
    Mat C = Mult_AnXBn_DoubleBuff<PlusTimesSRing<int64_t, int64_t>, int64_t, SpDCCols<int64_t, int64_t>>(L, Ltemp, /*clearA=*/false, /*clearB=*/true);
    timer->elapsed();

    timer->reset("mask");
    C.EWiseMult(L, false);
    timer->elapsed();
    // printSpParMat(C);

    // reduce C to get number of triangles
    timer->reset("reduce");
    Vec triangles = C.Reduce(Column, plus<int64_t>(), static_cast<int64_t>(0));
    int64_t result = triangles.Reduce(plus<int64_t>(), static_cast<int64_t>(0));
    timer->elapsed();

    MAIN_COUT("triangles = " << result << endl);
}

int main(int argc, char *argv[])
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 2)
    {
        cout << "Usage: " << argv[0] << " <input_graph>" << endl;
        MPI_Finalize();
        return 1;
    }

    string input_graph = argv[1];

    // TODO: Add loaders for binary graph files
    {
        // unsigned scale = 18; // 2^scale vertices
        // double initiator[4] = {.57, .19, .19, .05};

        // cout << "[Graph500] generating random graph ..." << endl;

        // double t01 = MPI_Wtime();
        // double t02;

        // DistEdgeList<int64_t> *DEL = new DistEdgeList<int64_t>();
        // DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, true);
        // MPI_Barrier(MPI_COMM_WORLD);

        // t02 = MPI_Wtime();
        // ostringstream tinfo;
        // tinfo << "Generation took " << t02 - t01 << " seconds" << endl;

        // adjacency matrix
        // Mat *A = new Mat(*DEL, false);
        // delete DEL;
        // int64_t removed = A->RemoveLoops();

        Timer timer(myrank);

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));
        Mat A(fullWorld);

        timer.reset("load");
        load_mtx<int64_t, int64_t, decltype(A), float>(&A, input_graph, false);
        timer.elapsed();

        MAIN_COUT("n = " << A.getnrow() << ", nnz = " << A.getnnz() << std::endl);

        // convert adjacency matrix to unweighted and undirected
        timer.reset("symmetrize");
        Symmetricize(A);
        A.Apply([](int64_t x){ return 1; });
        timer.elapsed();

        // L = tril(A)
        timer.reset("tril");
        to_tril(&A);
        timer.elapsed();

        TC(A, &timer);

        std::string timing_output = input_graph;
        timing_output.replace(timing_output.find_last_of('.'), std::string::npos, ".tc_time.csv");
        timer.save(timing_output);
    }

    MPI_Finalize();
    return 0;
}