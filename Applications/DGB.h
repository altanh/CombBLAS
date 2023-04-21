#include <chrono>
#include <sstream>
#include <iostream>

#include "CombBLAS/CombBLAS.h"

#define MAIN_COUT(x)                                                \
    {                                                               \
        /* evaluate x even if myrank != 0, to avoid MPI deadlock */ \
        std::stringstream ss;                                       \
        ss << x;                                                    \
        if (myrank == 0)                                            \
        {                                                           \
            std::cout << ss.str();                                  \
        }                                                           \
    }

class Timer
{
public:
    Timer(int myrank, const std::string &segment = "") : myrank(myrank) {
        reset(segment);
    }

    void reset(const std::string &segment = "")
    {
        start_ = std::chrono::steady_clock::now();
        this->segment = segment.empty() ? "timer" : segment;
    }

    double elapsed(bool print = true) const
    {
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
        if (print)
        {
            MAIN_COUT("[" << segment << "] ");
            MAIN_COUT("elapsed time: " << s << " s" << std::endl);
        }
        return s;
    }

private:
    std::chrono::steady_clock::time_point start_;
    std::string segment;
    int myrank;
};

// utility stuff for loading wg2mtx binary graphs

template <typename IT, typename NT>
struct BinHandler
{
    constexpr static NT ONE = static_cast<NT>(1);

    void binaryfill(FILE *f, IT &r, IT &c, NT &v)
    {
        IT rc[2];

        fread(rc, sizeof(IT), 2, f);
        r = rc[0] - 1;
        c = rc[1] - 1;
        v = ONE;
    }

    NT getNoNum(IT r, IT c)
    {
        return ONE;
    }

    template <typename c, typename t>
    NT read(std::basic_istream<c, t> &is, IT row, IT col)
    {
        return ONE;
    }

    template <typename c, typename t>
    void save(std::basic_ostream<c, t> &os, const NT &v, IT row, IT col)
    {
        os << v;
    }

    size_t entrylength()
    {
        return 2 * sizeof(IT);
    }
};

template <typename IT, typename NT, typename Mat>
void load_mtx(Mat *A, const std::string &filename, bool transpose) {
    if (filename.find(".bin64") != std::string::npos) {
        BinHandler<IT, NT> handler;
        A->ReadDistribute(filename, 0, true, handler, transpose, true);
    } else {
        A->ParallelReadMM(filename, true, combblas::maximum<NT>(), transpose);
    }
}
