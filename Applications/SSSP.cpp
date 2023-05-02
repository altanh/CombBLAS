#include "CombBLAS/CombBLAS.h"
#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>

#include "DGB_CombBLAS.h"

using namespace std;
using namespace combblas;

#define EDGEFACTOR 16
MTRand GlobalMT(123);

#define MAX_DIST std::numeric_limits<double>::max()
using Mat = SpParMat<int64_t, double, SpDCCols<int64_t, double>>;
using Vec = FullyDistVec<int64_t, double>;

// TODO(@altanh): delta-stepping?
void SSSP(Mat &A, int64_t source, Vec &dist, dgb::Timer *timer)
{
    int64_t n = A.getnrow();
    // cout << "n = " << n << endl;

    timer->reset("init_vec");
    dist = Vec(A.getcommgrid(), n, MAX_DIST); // init distances to MAX_DIST
    dist.SetElement(source, 0);               // set source distance to 0
    timer->elapsed();

    // cout << "Starting SSSP ..." << endl;
    timer->reset("sssp");
    // bellman-ford
    for (int64_t i = 0; i < n - 1; i++)
    {
        // printSpParMat(A);
        // dist.DebugPrint();
        // @manish: A is sparse but dist is dense. So runs a dense SpMV.
        Vec new_dist = SpMV<MinPlusSRing<double, double>>(A, dist);

        // TODO(@altanh): it would be nicer if we could avoid this; VC frameworks can track #updated
        //                values as they are computed.
        dist.EWiseApply(new_dist, [](double a, double b) { return static_cast<double>(a != b); });
        double updates = dist.Reduce(plus<double>(), 0.0);
        MAIN_COUT("iteration " << i << " done, " << updates << " updates" << std::endl);
        if (updates == 0.0) {
            break;
        }

        dist = new_dist;
    }
    timer->elapsed();
    // for (int i = 0; i < n; i++)
    // {
    //     cout << "Shortest distance from " << source << " to " << i << " is " << dist[i] << endl;
    // }
}



int main(int argc, char *argv[])
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 3)
    {
        cout << "Usage: " << argv[0] << " <input_graph> <source>" << endl;
        MPI_Finalize();
        return 1;
    }

    dgb::print_process_grid();

    string input_graph = argv[1];
    int64_t source = atoll(argv[2]);

    // TODO: Add loaders for binary graph files
    {
        // int source = 0;

        // unsigned scale = 10; // 2^scale vertices
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

        // // adjacency matrix
        // Mat *A = new Mat(*DEL, false);
        // delete DEL;
        // int64_t removed = A->RemoveLoops();

        // // distance vector from source to all nodes
        // Vec dist;
        // A->Transpose();

        // // set diagonal to 0
        // A->AddLoops(0, true);

        // SSSP(*A, source, dist);
        dgb::Timer timer;

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));
        Mat A(fullWorld);

        timer.reset("load");
        dgb::load_mtx<int64_t, double, decltype(A), float>(&A, input_graph, /*transpose=*/true, /*pattern=*/false);
        timer.elapsed();

        // TODO(@altanh): this is slow! ideally, this should be fused into loading somehow
        timer.reset("set_diag");
        A.AddLoops(0, true);
        timer.elapsed();

        Vec dist(A.getcommgrid());
        SSSP(A, source, dist, &timer);

        timer.save(dgb::get_timer_output(input_graph, "COMBBLAS", "sssp"));
    }

    MPI_Finalize();
    return 0;
}