#include "CombBLAS/CombBLAS.h"

// utility stuff for loading wg2mtx binary graphs

template<typename IT, typename NT>
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
    NT read(std::basic_istream<c,t>& is, IT row, IT col)
    {
        return ONE;
    }

    template <typename c, typename t>
    void save(std::basic_ostream<c,t>& os, const NT &v, IT row, IT col)
    {
        os << v;
    }

    size_t entrylength() {
        return 2 * sizeof(IT);
    }
};
