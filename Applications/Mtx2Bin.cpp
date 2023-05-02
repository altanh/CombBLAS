#include "CombBLAS/CombBLAS.h"
#include "DGB_CombBLAS.h"

using namespace combblas;

int main(int argc, char **argv)
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <input_mtx> [output_bin] [pattern]" << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::string input_mtx = argv[1];
    std::string output_bin = input_mtx + ".bin64";

    if (argc > 2)
    {
        output_bin = argv[2];
    }

    bool pattern = false;
    if (argc > 3)
    {
        pattern = std::stoi(argv[3]);
    }

    dgb::print_process_grid();

    MAIN_COUT("input MatrixMarket file: " << input_mtx << std::endl);
    MAIN_COUT("output binary file: " << output_bin << std::endl);
    MAIN_COUT("output is pattern: " << pattern << std::endl);

    using IT = int64_t;
    using NT = float;

    const size_t nt_size = pattern ? 0 : sizeof(NT);
    size_t expected_filesize = 0;

    {
        std::shared_ptr<CommGrid> fullWorld;
        fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));
        // read the input matrix
        SpParMat<IT, NT, SpDCCols<IT, NT>> A(fullWorld);

        dgb::Timer T;

        MAIN_COUT("reading input..." << std::endl);
        T.reset("read");
        dgb::load_mtx<IT, NT, decltype(A)>(&A, input_mtx, false);
        T.elapsed();

        // print statistics of loaded matrix
        MAIN_COUT("input matrix: " << A.getnrow() << " x " << A.getncol() << ", nnz = " << A.getnnz() << std::endl);

        MAIN_COUT("writing output..." << std::endl);
        T.reset("write");
        A.ParallelBinaryWrite(output_bin, /*pattern=*/pattern);
        T.elapsed();

        expected_filesize = A.getnnz() * (2 * sizeof(IT) + nt_size) + 52;
    }

    // check the output file has the correct size
    std::ifstream f(output_bin, std::ios::binary | std::ios::ate);
    if (f.tellg() != expected_filesize)
    {
        MAIN_COUT("ERROR: output file has incorrect size" << std::endl);
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();

    return 0;
}