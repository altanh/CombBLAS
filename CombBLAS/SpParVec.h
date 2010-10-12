#ifndef _SP_PAR_VEC_H_
#define _SP_PAR_VEC_H_

#include <iostream>
#include <vector>
#include <utility>

#ifdef NOTR1
	#include <boost/tr1/memory.hpp>
#else
	#include <tr1/memory>
#endif
#include "CommGrid.h"

using namespace std;
using namespace std::tr1;


/** 
  * A sparse vector of length n (with nnz <= n of them being nonzeros) is distributed to diagonal processors 
  * This is done without any reference to ordering of the nonzero indices
  * Example: x = [5,1,6,2,9] and we use 4 processors P_00, P_01, P_10, P_11
  * Then P_00 owns [5,1,6] and P_11 owns [2,9]
  * 	In the case of A(v,w) type sparse matrix indexing, this doesn't matter because n = nnz
  * 	After all, A(v,w) will have dimensions length(v) x length (w) 
  * 	v and w will be of numerical type (NT) "int" and their indices (IT) will be consecutive integers 
 **/
  
template <class IT, class NT>
class SpParVec
{
public:
	SpParVec ( shared_ptr<CommGrid> grid);
	SpParVec<IT,NT> & operator+=(const SpParVec<IT,NT> & rhs);
	ifstream& ReadDistribute (ifstream& infile, int master);	
	
	IT getnnz() const
	{
		IT totnnz = 0;
		IT locnnz = arr.length();
		(commGrid->GetDiagWorld()).Allreduce( &locnnz, & totnnz, 1, MPIType<IT>(), MPI::SUM); 
	}

private:
	shared_ptr<CommGrid> commGrid;
	vector< pair<IT, NT> > arr;	// arr.length() give the number of nonzeros
	IT length;			// actual length of the vector (including zeros)
	bool diagonal;
	const static IT zero = static_cast<IT>(0);

	template <class IU, class NU>
	friend class DenseParVec;
	
	template <class IU, class NU, class UDER>
	friend class SpParmat;
};

#include "SpParVec.cpp"
#endif

