// We follow the pseudocode described in Algorithm 3 of [1].
//
// [1] Manoj Kumar, José E. Moreira, and Pratap Pattnaik. 2018.
//     GraphBLAS: handling performance concerns in large graph analytics.
//     In Proceedings of the 15th ACM International Conference on Computing Frontiers (CF '18).
//     Association for Computing Machinery, New York, NY, USA, 260–267. https://doi.org/10.1145/3203217.3205342

#include <chrono>
#include <iostream>
#include "CombBLAS/CombBLAS.h"

#include "WG.h"

using namespace combblas;

class Timer
{
public:
    Timer(int myrank) : start_(std::chrono::steady_clock::now()), myrank_(myrank) {}

    void reset()
    {
        start_ = std::chrono::steady_clock::now();
    }

    double elapsed(bool print = true) const
    {
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
        if (print && myrank_ == 0)
        {
            std::cout << "[rank " << myrank_ << "] elapsed time: " << s << "s" << std::endl;
        }
        return s;
    }

    template <typename T>
    const Timer &operator<<(T v) const
    {
        if (myrank_ == 0)
        {
            std::cout << v;
        }
        return *this;
    }

private:
    std::chrono::steady_clock::time_point start_;
    int myrank_;
};

const double alpha = 0.85;

int main(int argc, char **argv)
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int nthreads = 1;
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif

    if (argc < 2)
    {
        if (myrank == 0)
        {
            std::cout << "Usage: " << argv[0] << " <input_graph> [eps] [max_iter] [save]" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    double eps = 1e-6;
    int max_iter = 10;
    bool save = false;

    if (argc > 2)
    {
        eps = std::stod(argv[2]);
    }
    if (argc > 3)
    {
        max_iter = std::stoi(argv[3]);
    }
    if (argc > 4)
    {
        save = std::stoi(argv[4]);
    }

    if (myrank == 0)
    {
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "input = " << argv[1] << std::endl;
        std::cout << "eps = " << eps << std::endl;
        std::cout << "Process Grid (p x p x t): " << sqrt(nprocs) << " x " << sqrt(nprocs) << " x " << nthreads << endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    { // begin main scope

        std::string input_graph(argv[1]);
        std::shared_ptr<CommGrid> fullWorld;
        fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

        using ArithSR = PlusTimesSRing<double, double>;
        using Mat = SpParMat<int64_t, double, SpDCCols<int64_t, double>>;
        using Vec = FullyDistVec<int64_t, double>;
        using SpVec = FullyDistSpVec<int64_t, double>;

        Timer timer(myrank);
        timer << "reading matrix...\n";
        Mat A(fullWorld);
        // check if binary
        if (input_graph.find(".bin64") != std::string::npos)
        {
            BinHandler<int64_t, double> handler;
            A.ReadDistribute(input_graph, /*master=*/0, /*nonum=*/true, /*handler=*/handler, /*transpose=*/false, /*pario=*/true);
        }
        else
        {
            A.ParallelReadMM(input_graph, true, maximum<double>());
        }
        timer.elapsed();
        timer << "load imbalance = " << A.LoadImbalance() << "\n";

        int64_t n = A.getnrow();
        double inv_n = 1.0 / static_cast<double>(n);
        double err = inv_n;

        timer << "n = " << n << "\n";

        // p <- 1/n
        timer << "initializing dense vectors...\n";
        timer.reset();
        Vec p(A.getcommgrid(), n, inv_n);

        // NOTE(@altanh): using ArithSR::add instead of std::plus<double>() leads to a segfault
        //                when using multiple MPI ranks...
        Vec od_dense = A.Reduce(Dim::Row, std::plus<double>(), 0.0);

        // print number of dangling nodes
        int64_t num_dangling = od_dense.Count([](double x)
                                              { return x == 0.0; });
        timer << "found " << num_dangling << " dangling nodes\n";

        // TODO(@altanh): is there a better way? also, should I be using SpVec?
        // SpVec od = SpVec(od_dense, [](double x) { return x != 0.0; });
        // SpVec not_od = SpVec(od_dense, [](double x) { return x == 0.0; });

        // b<od> <- (1 - alpha)/n; b<!od> <- 1/n
        Vec b(od_dense);
        b.Apply([inv_n](double x)
                { return x == 0.0 ? inv_n : (1.0 - alpha) * inv_n; });

        // od_inv<od># <- 1/od
        Vec od_inv(od_dense);
        od_inv.Apply([](double x)
                     { return x == 0.0 ? 0.0 : 1.0 / x; });
        timer.elapsed();

        // SpVec od_inv(od);
        // od_inv.Apply([](double x) { return 1.0 / x; });

        // transpose A for the vector-matrix product later (we'll use SpMV)
        timer << "transposing A...\n";
        timer.reset();
        A.Transpose();
        timer.elapsed();

        timer.reset();
        timer << "running pagerank...\n";

        int iter = 0;
        while (err > eps && iter < max_iter)
        {
            Vec p_old = p; // copies the vector

            // temp <- b .* p_old
            Vec temp = b;
            temp.EWiseApply(p_old, std::multiplies<double>());

            double t = temp.Reduce(std::plus<double>(), 0.0);

            // p_new <- t
            p = t;

            // temp<od># <- p_old .* od_inv. (the "masking" is encoded in od_inv)
            p_old.EWiseOut(od_inv, std::multiplies<double>(), temp);

            // temp <- temp +.* A (equivalent to A^T +.* temp)
            temp = SpMV<ArithSR>(A, temp);

            // p_new <- p_new + (temp .* alpha)
            temp.Apply([](double x)
                       { return x * alpha; }); // TODO: ideally should vectorize
            p += temp;

            // check convergence
            // temp <- p_new - p_old
            p.EWiseOut(p_old, std::minus<double>(), temp);

            // temp <- temp .* temp
            temp.Apply([](double x)
                       { return x * x; });
            err = temp.Reduce(std::plus<double>(), 0.0);

            iter++;
        }

        double pr_time = timer.elapsed(false);
        if (myrank == 0)
        {
            std::cout << "PageRank stopped after " << iter << " iterations in " << pr_time << " seconds" << std::endl;
        }

        if (save)
        {
            // get a local copy of p
            std::vector<double> p_local(n);
            for (int64_t i = 0; i < n; i++)
            {
                p_local[i] = p[i];
            }
            // save to file
            if (myrank == 0)
            {
                // output filename is input filename with .pr.txt extension
                std::string output_filename = input_graph;
                output_filename.replace(output_filename.find_last_of('.'), std::string::npos, ".pr.txt");

                // write PageRank values to file
                std::cout << "Writing PageRank values to " << output_filename << "..." << std::endl;
                std::ofstream pr_file(output_filename);
                for (int64_t i = 0; i < n; i++)
                {
                    pr_file << p_local[i] << std::endl;
                    // print progress every 5%
                    if (i % (n / 20) == 0)
                    {
                        std::cout << (i * 100 / n) << "%..." << std::endl;
                    }
                }
                pr_file.close();
                std::cout << "Done." << std::endl;
            }
        }

    } // end of main scope, this should free MPI stuff

    MPI_Finalize();

    return 0;
}
