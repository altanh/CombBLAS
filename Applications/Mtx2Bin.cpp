#include "CombBLAS/CombBLAS.h"
#include "DGB.h"

using namespace combblas;

int main(int argc, char **argv)
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <input_mtx> [output_bin]" << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::string input_mtx = argv[1];
    std::string output_bin = input_mtx + ".bin64";

    if (argc > 2)
    {
        output_bin = argv[2];
    }

    MAIN_COUT("input MatrixMarket file: " << input_mtx << std::endl);
    MAIN_COUT("output binary file: " << output_bin << std::endl);

    using IT = int64_t;
    using NT = double;

    {
        std::shared_ptr<CommGrid> fullWorld;
        fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));
        // read the input matrix
        SpParMat<int64_t, double, SpDCCols<int64_t, double>> A(fullWorld);

        Timer T(myrank);

        MAIN_COUT("reading input..." << std::endl);
        T.reset("read");
        load_mtx<IT, NT, decltype(A)>(&A, input_mtx, false);
        T.elapsed();

        MAIN_COUT("writing output..." << std::endl);
        T.reset("write");
        A.ParallelBinaryWrite(output_bin, /*pattern=*/true);
        T.elapsed();
    }

    MPI_Finalize();

    return 0;
}
