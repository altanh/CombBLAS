
#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "CombBLAS/CombBLAS.h"

#include "DGB.h"

using namespace std;
using namespace combblas;

#ifdef TIMING
double cblas_alltoalltime;
double cblas_allgathertime;
#endif

#ifdef _OPENMP
int cblas_splits = omp_get_max_threads(); 
#else
int cblas_splits = 1;
#endif


// Simple helper class for declarations: Just the numerical type is templated 
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class PSpMat 
{ 
public:
	typedef SpDCCols < int64_t, NT > DCCols;
	typedef SpParMat < int64_t, NT, DCCols > MPI_DCCols;
};

int main(int argc, char* argv[])
{
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	if(argc < 4)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./MultTest <MatrixA> <MatrixB> <MatrixC>" << endl;
			cout << "<MatrixA>,<MatrixB>,<MatrixC> are absolute addresses, and files should be in triples format" << endl;
		}
		MPI_Finalize(); 
		return -1;
	}				
	{
		string Aname(argv[1]);		
		string Bname(argv[2]);
		string Cname(argv[3]);
        
		MPI_Barrier(MPI_COMM_WORLD);	
		typedef PlusTimesSRing<double, double> PTDOUBLEDOUBLE;	
		typedef SelectMaxSRing<bool, int64_t> SR;	

		shared_ptr<CommGrid> fullWorld;
		fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

        // construct objects
        PSpMat<double>::MPI_DCCols A(fullWorld);
        PSpMat<double>::MPI_DCCols B(fullWorld);
        PSpMat<double>::MPI_DCCols C(fullWorld);
        PSpMat<double>::MPI_DCCols CControl(fullWorld);

		cout << Aname << endl;

		A.ReadGeneralizedTuples(Aname, maximum<double>());
		B.ReadGeneralizedTuples(Bname, maximum<double>());
		CControl.ReadGeneralizedTuples(Cname, maximum<double>());
		C = Masked_Mult_AnXBn_Synch<PTDOUBLEDOUBLE, double, PSpMat<double>::DCCols >(A,B,B);

		if (CControl == C)
		{
			SpParHelper::Print("Synchronous Multiplication working correctly\n");	
			// C.SaveGathered("CControl.txt");
		}
		else
		{
			SpParHelper::Print("ERROR in Synchronous Multiplication, go fix it!\n");	
		}
	}
	MPI_Finalize();
	return 0;
}

