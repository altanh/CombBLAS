#include <chrono>
#include <sstream>
#include <iostream>

#include "CombBLAS/CombBLAS.h"

#define MAIN_COUT(x)                                                \
    {                                                               \
        /* evaluate x even if myrank != 0, to avoid MPI deadlock */ \
        std::ostringstream ss;                                      \
        ss << x;                                                    \
        if (myrank == 0)                                            \
        {                                                           \
            std::cout << ss.str();                                  \
        }                                                           \
    }

class Timer
{
public:
    Timer(int myrank, const std::string &segment = "") : myrank(myrank)
    {
        reset(segment);
    }

    void reset(const std::string &segment = "")
    {
        start_ = std::chrono::steady_clock::now();
        if (segment.empty())
        {
            std::ostringstream ss;
            ss << "segment" << entries.size();
            this->segment = ss.str();
        }
        else
        {
            this->segment = segment;
        }
    }

    double elapsed(bool print = true)
    {
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
        if (print)
        {
            MAIN_COUT("[" << segment << "] ");
            MAIN_COUT("elapsed time: " << s << " s" << std::endl);
        }
        Entry e;
        strncpy(e.segment, segment.c_str(), 64);
        e.myrank = myrank;
        e.elapsed = s;
        entries.push_back(e);
        return s;
    }

    void save(const std::string &filename)
    {
        // make MPI datatype for entries
        static MPI_Datatype MPI_Entry;
        static bool MPI_Entry_initialized = false;
        if (!MPI_Entry_initialized)
        {
            MPI_Type_contiguous(sizeof(Entry), MPI_BYTE, &MPI_Entry);
            MPI_Type_commit(&MPI_Entry);
            MPI_Entry_initialized = true;
        }

        // gather entries from all ranks
        int nprocs;
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        std::vector<Entry> all_entries;
        // TODO: assumes all ranks timed the same number of segments
        all_entries.resize(entries.size() * nprocs);

        // gather to root
        MPI_Gather(entries.data(), entries.size(), MPI_Entry,
                   all_entries.data(), entries.size(), MPI_Entry,
                   0, MPI_COMM_WORLD);

        // write to file
        if (myrank == 0)
        {
            std::ofstream f(filename);
            if (!f)
            {
                std::cerr << "could not save timing results: failed to open " << filename << std::endl;
                return;
            }
            f << "segment,rank,elapsed" << std::endl;
            for (const auto &e : all_entries)
            {
                f << e.segment << "," << e.myrank << "," << e.elapsed << std::endl;
            }
            f.close();
        }
    }

private:
    struct Entry
    {
        char segment[64];
        int myrank;
        double elapsed;
    } __attribute__((packed));

    std::chrono::steady_clock::time_point start_;
    std::string segment;
    int myrank;
    std::vector<Entry> entries;
};

// utility stuff for loading wg2mtx binary graphs

template <typename IT, typename NT, typename FNT = NT>
struct BinHandler
{
    constexpr static NT ONE = static_cast<NT>(1);
    bool pattern;

    BinHandler(bool pattern) : pattern(pattern) {}

    void binaryfill(FILE *f, IT &r, IT &c, NT &v)
    {
        IT rc[2];
        FNT vv = static_cast<FNT>(1);

        fread(rc, sizeof(IT), 2, f);
        if (!pattern)
        {
            fread(&vv, sizeof(FNT), 1, f);
        }

        r = rc[0] - 1;
        c = rc[1] - 1;
        v = static_cast<NT>(vv);
    }

    NT getNoNum(IT r, IT c)
    {
        return ONE;
    }

    template <typename c, typename t>
    NT read(std::basic_istream<c, t> &is, IT row, IT col)
    {
        NT v;
        is >> v;
        return v;
    }

    template <typename c, typename t>
    void save(std::basic_ostream<c, t> &os, const NT &v, IT row, IT col)
    {
        os << v;
    }

    size_t entrylength()
    {
        return 2 * sizeof(IT) + (pattern ? 0 : sizeof(FNT));
    }
};

template <typename IT, typename NT, typename Mat, typename FNT = NT>
void load_mtx(Mat *A, const std::string &filename, bool transpose, bool pattern = false)
{
    if (filename.find(".bin64") != std::string::npos)
    {
        BinHandler<IT, NT, FNT> handler(pattern);
        A->ReadDistribute(filename, 0, /*nonum=*/false, handler, transpose, true);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else
    {
        A->ParallelReadMM(filename, true, combblas::maximum<NT>(), transpose);
    }
}

int get_myrank()
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    return myrank;
}

// print the SpParMat
template <typename IT, typename NT>
void printSpParMat(combblas::SpParMat<IT, NT, combblas::SpDCCols<IT, NT>> &A)
{
    // get local matrix
    combblas::SpDCCols<IT, NT> localMat = A.seq();
    int count = 0;
    int total = 0;

    // temporary vector of vector to store the matrix
    vector<vector<NT>> temp(A.getnrow(), vector<NT>(A.getncol(), 0));

    // use SpColIter to iterate over cols of local matrix
    for (auto colit = A.seq().begcol(); colit != A.seq().endcol(); ++colit)
    {
        for (auto nzit = A.seq().begnz(colit); nzit != A.seq().endnz(colit); ++nzit)
        {
            count++;
            temp[nzit.rowid()][colit.colid()] = nzit.value();
        }
    }

    int myrank = get_myrank();

    // print the temp vector
    if (myrank == 0)
    {
        for (auto row : temp)
        {
            for (auto col : row)
            {
                std::cout << setw(5) << col << " ";
            }
            std::cout << endl;
        }
    }
}

void print_process_grid() {
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    int nthreads = 1;
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif
    MAIN_COUT("Process Grid (p x p x t): " << sqrt(nprocs) << " x " << sqrt(nprocs) << " x " << nthreads << endl);
}
