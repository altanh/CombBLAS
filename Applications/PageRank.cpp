// We follow the pseudocode described in Algorithm 3 of [1].
//
// [1] Manoj Kumar, José E. Moreira, and Pratap Pattnaik. 2018.
//     GraphBLAS: handling performance concerns in large graph analytics.
//     In Proceedings of the 15th ACM International Conference on Computing Frontiers (CF '18).
//     Association for Computing Machinery, New York, NY, USA, 260–267. https://doi.org/10.1145/3203217.3205342

#include <chrono>
#include <iostream>
#include <sstream>
#include "CombBLAS/CombBLAS.h"

#include "DGB.h"

using namespace combblas;

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
        MAIN_COUT("Usage: " << argv[0] << " <input_graph> [eps] [max_iter] [save]" << std::endl);
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

    MAIN_COUT("----------------------------------------" << std::endl);
    MAIN_COUT("input = " << argv[1] << std::endl);
    MAIN_COUT("eps = " << eps << std::endl);
    MAIN_COUT("max_iter = " << max_iter << std::endl);
    print_process_grid();
    MAIN_COUT("----------------------------------------" << std::endl);

    { // begin main scope

        std::string input_graph(argv[1]);
        std::shared_ptr<CommGrid> fullWorld;
        fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

        using ArithSR = PlusTimesSRing<double, double>;
        using Mat = SpParMat<int64_t, double, SpDCCols<int64_t, double>>;
        using Vec = FullyDistVec<int64_t, double>;
        using SpVec = FullyDistSpVec<int64_t, double>;

        Timer timer(myrank);

        Mat A(fullWorld);
        MAIN_COUT("reading matrix..." << std::endl);
        timer.reset("load");
        load_mtx<int64_t, double, Mat>(&A, input_graph, /*transpose=*/true, /*pattern=*/true);
        // set values to 1.0
        A.Apply([](double x)
                { return 1.0; });
        timer.elapsed();
        MAIN_COUT("load imbalance = " << A.LoadImbalance() << std::endl);

        int64_t n = A.getnrow();
        double inv_n = 1.0 / static_cast<double>(n);
        double err = inv_n;

        MAIN_COUT("  n = " << n << std::endl);
        MAIN_COUT("nnz = " << A.getnnz() << std::endl);

        // p <- 1/n
        MAIN_COUT("initializing dense vectors..." << std::endl);
        timer.reset("vec_init");
        Vec p(A.getcommgrid(), n, inv_n);

        // NOTE(@altanh): using ArithSR::add instead of std::plus<double>() leads to a segfault
        //                when using multiple MPI ranks...
        // TODO(@altanh): support promoted accumulator for reduce (e.g. bool -> double)
        Vec od_dense = A.Reduce(Dim::Column, std::plus<double>(), 0.0);

        // print number of dangling nodes
        int64_t num_dangling = od_dense.Count([](double x)
                                              { return x == 0.0; });
        MAIN_COUT("found " << num_dangling << " dangling nodes" << std::endl);

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

        MAIN_COUT("starting PageRank..." << std::endl);
        timer.reset("pagerank");

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
        MAIN_COUT("PageRank stopped after " << iter << " iterations in " << pr_time << " seconds" << std::endl);

        std::string timing_output = input_graph;
        timing_output.replace(timing_output.find_last_of('.'), std::string::npos, ".pr_time.csv");
        timer.save(timing_output);

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
