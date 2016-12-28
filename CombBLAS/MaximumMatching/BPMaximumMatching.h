#include "../CombBLAS.h"
#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>


using namespace std;

// algorithmic options




struct VertexType
{
public:
    VertexType(int64_t p=-1, int64_t r=-1, int16_t pr=0){parent=p; root = r; prob = pr;};
    
    friend bool operator<(const VertexType & vtx1, const VertexType & vtx2 )
    {
        if(vtx1.prob==vtx2.prob) return vtx1.parent<vtx2.parent;
        else return vtx1.prob<vtx2.prob;
    };
    friend bool operator==(const VertexType & vtx1, const VertexType & vtx2 ){return vtx1.parent==vtx2.parent;};
    friend ostream& operator<<(ostream& os, const VertexType & vertex ){os << "(" << vertex.parent << "," << vertex.root << ")"; return os;};
    //private:
    int64_t parent;
    int64_t root;
    int16_t prob; // probability of selecting an edge
    
};






/**
 * Create a boolean matrix A (not necessarily a permutation matrix)
 * Input: ri: a dense vector (actual values in FullyDistVec should be IT)
 *        ncol: number of columns in the output matrix A
 * Output: a boolean matrix A with m=size(ri) and n=ncol (input)
 and  A[k,ri[k]]=1
 * This can be done by Matlab like constructor, no?
 */
template <class IT, class DER>
SpParMat<IT, bool, DER> PermMat (const FullyDistVec<IT,IT> & ri, const IT ncol)
{
    
    IT procsPerRow = ri.commGrid->GetGridCols();	// the number of processor in a row of processor grid
    IT procsPerCol = ri.commGrid->GetGridRows();	// the number of processor in a column of processor grid
    
    
    IT global_nrow = ri.TotalLength();
    IT global_ncol = ncol;
    IT m_perprocrow = global_nrow / procsPerRow;
    IT n_perproccol = global_ncol / procsPerCol;
    
    
    // The indices for FullyDistVec are offset'd to 1/p pieces
    // The matrix indices are offset'd to 1/sqrt(p) pieces
    // Add the corresponding offset before sending the data
    
    vector< vector<IT> > rowid(procsPerRow); // rowid in the local matrix of each vector entry
    vector< vector<IT> > colid(procsPerRow); // colid in the local matrix of each vector entry
    
    IT locvec = ri.arr.size();	// nnz in local vector
    IT roffset = ri.RowLenUntil(); // the number of vector elements in this processor row before the current processor
    for(typename vector<IT>::size_type i=0; i< (unsigned)locvec; ++i)
    {
        if(ri.arr[i]>=0 && ri.arr[i]<ncol) // this specialized for matching. TODO: make it general purpose by passing a function
        {
            IT rowrec = (n_perproccol!=0) ? std::min(ri.arr[i] / n_perproccol, procsPerRow-1) : (procsPerRow-1);
            // ri's numerical values give the colids and its local indices give rowids
            rowid[rowrec].push_back( i + roffset);
            colid[rowrec].push_back(ri.arr[i] - (rowrec * n_perproccol));
        }
        
    }
    
    
    
    int * sendcnt = new int[procsPerRow];
    int * recvcnt = new int[procsPerRow];
    for(IT i=0; i<procsPerRow; ++i)
    {
        sendcnt[i] = rowid[i].size();
    }
    
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, ri.commGrid->GetRowWorld()); // share the counts
    
    int * sdispls = new int[procsPerRow]();
    int * rdispls = new int[procsPerRow]();
    partial_sum(sendcnt, sendcnt+procsPerRow-1, sdispls+1);
    partial_sum(recvcnt, recvcnt+procsPerRow-1, rdispls+1);
    IT p_nnz = accumulate(recvcnt,recvcnt+procsPerRow, static_cast<IT>(0));
    
    
    IT * p_rows = new IT[p_nnz];
    IT * p_cols = new IT[p_nnz];
    IT * senddata = new IT[locvec];
    for(int i=0; i<procsPerRow; ++i)
    {
        copy(rowid[i].begin(), rowid[i].end(), senddata+sdispls[i]);
        vector<IT>().swap(rowid[i]);	// clear memory of rowid
    }
    MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), p_rows, recvcnt, rdispls, MPIType<IT>(), ri.commGrid->GetRowWorld());
    
    for(int i=0; i<procsPerRow; ++i)
    {
        copy(colid[i].begin(), colid[i].end(), senddata+sdispls[i]);
        vector<IT>().swap(colid[i]);	// clear memory of colid
    }
    MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), p_cols, recvcnt, rdispls, MPIType<IT>(), ri.commGrid->GetRowWorld());
    delete [] senddata;
    
    tuple<IT,IT,bool> * p_tuples = new tuple<IT,IT,bool>[p_nnz];
    for(IT i=0; i< p_nnz; ++i)
    {
        p_tuples[i] = make_tuple(p_rows[i], p_cols[i], 1);
    }
    DeleteAll(p_rows, p_cols);
    
    
    // Now create the local matrix
    IT local_nrow = ri.MyRowLength();
    int my_proccol = ri.commGrid->GetRankInProcRow();
    IT local_ncol = (my_proccol<(procsPerCol-1))? (n_perproccol) : (global_ncol - (n_perproccol*(procsPerCol-1)));
    
    // infer the concrete type SpMat<IT,IT>
    typedef typename create_trait<DER, IT, bool>::T_inferred DER_IT;
    DER_IT * PSeq = new DER_IT();
    PSeq->Create( p_nnz, local_nrow, local_ncol, p_tuples);		// deletion of tuples[] is handled by SpMat::Create
    
    SpParMat<IT,bool,DER_IT> P (PSeq, ri.commGrid);
    //Par_DCSC_Bool P (PSeq, ri.commGrid);
    return P;
}




/**
 * Create a boolean matrix A (not necessarily a permutation matrix)
 * Input: ri: a sparse vector (actual values in FullyDistVec should be IT)
 *        ncol: number of columns in the output matrix A
 * Output: a boolean matrix A with m=size(ri) and n=ncol (input)
 and  A[k,ri[k]]=1
 * not used anymore. Candidate for deletion
 */

template <class IT, class NT, class DER, typename _UnaryOperation>
SpParMat<IT, bool, DER> PermMat1 (const FullyDistSpVec<IT,NT> & ri, const IT ncol, _UnaryOperation __unop)
{
    
    IT procsPerRow = ri.commGrid->GetGridCols();	// the number of processor in a row of processor grid
    IT procsPerCol = ri.commGrid->GetGridRows();	// the number of processor in a column of processor grid
    
    
    IT global_nrow = ri.TotalLength();
    IT global_ncol = ncol;
    IT m_perprocrow = global_nrow / procsPerRow;
    IT n_perproccol = global_ncol / procsPerCol;
    
    
    // The indices for FullyDistVec are offset'd to 1/p pieces
    // The matrix indices are offset'd to 1/sqrt(p) pieces
    // Add the corresponding offset before sending the data
    
    vector< vector<IT> > rowid(procsPerRow); // rowid in the local matrix of each vector entry
    vector< vector<IT> > colid(procsPerRow); // colid in the local matrix of each vector entry
    
    IT locvec = ri.num.size();	// nnz in local vector
    IT roffset = ri.RowLenUntil(); // the number of vector elements in this processor row before the current processor
    for(typename vector<IT>::size_type i=0; i< (unsigned)locvec; ++i)
    {
        IT val = __unop(ri.num[i]);
        if(val>=0 && val<ncol)
        {
            IT rowrec = (n_perproccol!=0) ? std::min(val / n_perproccol, procsPerRow-1) : (procsPerRow-1);
            // ri's numerical values give the colids and its local indices give rowids
            //rowid[rowrec].push_back( i + roffset);
            rowid[rowrec].push_back( ri.ind[i] + roffset);
            colid[rowrec].push_back(val - (rowrec * n_perproccol));
        }
    }
    
    
    
    int * sendcnt = new int[procsPerRow];
    int * recvcnt = new int[procsPerRow];
    for(IT i=0; i<procsPerRow; ++i)
    {
        sendcnt[i] = rowid[i].size();
    }
    
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, ri.commGrid->GetRowWorld()); // share the counts
    
    int * sdispls = new int[procsPerRow]();
    int * rdispls = new int[procsPerRow]();
    partial_sum(sendcnt, sendcnt+procsPerRow-1, sdispls+1);
    partial_sum(recvcnt, recvcnt+procsPerRow-1, rdispls+1);
    IT p_nnz = accumulate(recvcnt,recvcnt+procsPerRow, static_cast<IT>(0));
    
    
    IT * p_rows = new IT[p_nnz];
    IT * p_cols = new IT[p_nnz];
    IT * senddata = new IT[locvec];
    for(int i=0; i<procsPerRow; ++i)
    {
        copy(rowid[i].begin(), rowid[i].end(), senddata+sdispls[i]);
        vector<IT>().swap(rowid[i]);	// clear memory of rowid
    }
    MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), p_rows, recvcnt, rdispls, MPIType<IT>(), ri.commGrid->GetRowWorld());
    
    for(int i=0; i<procsPerRow; ++i)
    {
        copy(colid[i].begin(), colid[i].end(), senddata+sdispls[i]);
        vector<IT>().swap(colid[i]);	// clear memory of colid
    }
    MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), p_cols, recvcnt, rdispls, MPIType<IT>(), ri.commGrid->GetRowWorld());
    delete [] senddata;
    
    tuple<IT,IT,bool> * p_tuples = new tuple<IT,IT,bool>[p_nnz];
    for(IT i=0; i< p_nnz; ++i)
    {
        p_tuples[i] = make_tuple(p_rows[i], p_cols[i], 1);
    }
    DeleteAll(p_rows, p_cols);
    
    
    // Now create the local matrix
    IT local_nrow = ri.MyRowLength();
    int my_proccol = ri.commGrid->GetRankInProcRow();
    IT local_ncol = (my_proccol<(procsPerCol-1))? (n_perproccol) : (global_ncol - (n_perproccol*(procsPerCol-1)));
    
    // infer the concrete type SpMat<IT,IT>
    typedef typename create_trait<DER, IT, bool>::T_inferred DER_IT;
    DER_IT * PSeq = new DER_IT();
    PSeq->Create( p_nnz, local_nrow, local_ncol, p_tuples);		// deletion of tuples[] is handled by SpMat::Create
    
    SpParMat<IT,bool,DER_IT> P (PSeq, ri.commGrid);
    //Par_DCSC_Bool P (PSeq, ri.commGrid);
    return P;
}





/***************************************************************************
// Augment a matching by a set of vertex-disjoint augmenting paths.
// The paths are explored level-by-level similar to the level-synchronous BFS
// This approach is more effecient when we have many short augmenting paths
***************************************************************************/


void AugmentLevel(FullyDistVec<int64_t, int64_t>& mateRow2Col, FullyDistVec<int64_t, int64_t>& mateCol2Row, FullyDistVec<int64_t, int64_t>& parentsRow, FullyDistVec<int64_t, int64_t>& leaves)
{
    
    int64_t nrow = mateRow2Col.TotalLength();
    int64_t ncol = mateCol2Row.TotalLength();
    FullyDistSpVec<int64_t, int64_t> col(leaves, [](int64_t leaf){return leaf!=-1;});
    FullyDistSpVec<int64_t, int64_t> row(mateRow2Col.getcommgrid(), nrow);
    FullyDistSpVec<int64_t, int64_t> nextcol(col.getcommgrid(), ncol);
    
    while(col.getnnz()!=0)
    {
        
        row = col.Invert(nrow);
        row = EWiseApply<int64_t>(row, parentsRow,
                                  [](int64_t root, int64_t parent){return parent;},
                                  [](int64_t root, int64_t parent){return true;},
                                  false, (int64_t)-1);
        
        col = row.Invert(ncol); // children array
        nextcol = EWiseApply<int64_t>(col, mateCol2Row,
                                      [](int64_t child, int64_t mate){return mate;},
                                      [](int64_t child, int64_t mate){return mate!=-1;},
                                      false, (int64_t)-1);
        mateRow2Col.Set(row);
        mateCol2Row.Set(col);
        col = nextcol;
    }
}


/***************************************************************************
// Augment a matching by a set of vertex-disjoint augmenting paths.
// An MPI processor is responsible for a complete path.
// This approach is more effecient when we have few long augmenting paths
// We used one-sided MPI. Any PGAS language should be fine as well.
// This function is not thread safe, hence multithreading is not used here
 ***************************************************************************/

template <typename IT, typename NT>
void AugmentPath(FullyDistVec<int64_t, int64_t>& mateRow2Col, FullyDistVec<int64_t, int64_t>& mateCol2Row,FullyDistVec<int64_t, int64_t>& parentsRow, FullyDistVec<int64_t, int64_t>& leaves)
{
    MPI_Win win_mateRow2Col, win_mateCol2Row, win_parentsRow;
    MPI_Win_create(mateRow2Col.GetLocArr(), mateRow2Col.LocArrSize() * sizeof(int64_t), sizeof(int64_t), MPI_INFO_NULL, mateRow2Col.commGrid->GetWorld(), &win_mateRow2Col);
    MPI_Win_create(mateCol2Row.GetLocArr(), mateCol2Row.LocArrSize() * sizeof(int64_t), sizeof(int64_t), MPI_INFO_NULL, mateCol2Row.commGrid->GetWorld(), &win_mateCol2Row);
    MPI_Win_create(parentsRow.GetLocArr(), parentsRow.LocArrSize() * sizeof(int64_t), sizeof(int64_t), MPI_INFO_NULL, parentsRow.commGrid->GetWorld(), &win_parentsRow);
    
    
    int64_t* leaves_ptr = leaves.GetLocArr();
    //MPI_Win_fence(0, win_mateRow2Col);
    //MPI_Win_fence(0, win_mateCol2Row);
    //MPI_Win_fence(0, win_parentsRow);
    
    int64_t row, col=100, nextrow;
    int owner_row, owner_col;
    IT locind_row, locind_col;
    int myrank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    
    for(IT i=0; i<leaves.LocArrSize(); i++)
    {
        int depth=0;
        row = *(leaves_ptr+i);
        while(row != - 1)
        {
            
            owner_row = mateRow2Col.Owner(row, locind_row);
            MPI_Win_lock(MPI_LOCK_SHARED, owner_row, 0, win_parentsRow);
            MPI_Get(&col, 1, MPIType<NT>(), owner_row, locind_row, 1, MPIType<NT>(), win_parentsRow);
            MPI_Win_unlock(owner_row, win_parentsRow);
            
            owner_col = mateCol2Row.Owner(col, locind_col);
            MPI_Win_lock(MPI_LOCK_SHARED, owner_col, 0, win_mateCol2Row);
            MPI_Fetch_and_op(&row, &nextrow, MPIType<NT>(), owner_col, locind_col, MPI_REPLACE, win_mateCol2Row);
            MPI_Win_unlock(owner_col, win_mateCol2Row);
            
            MPI_Win_lock(MPI_LOCK_SHARED, owner_row, 0, win_mateRow2Col);
            MPI_Put(&col, 1, MPIType<NT>(), owner_row, locind_row, 1, MPIType<NT>(), win_mateRow2Col);
            MPI_Win_unlock(owner_row, win_mateRow2Col); // we need this otherwise col might get overwritten before communication!
            row = nextrow;
            
        }
    }
    
    //MPI_Win_fence(0, win_mateRow2Col);
    //MPI_Win_fence(0, win_mateCol2Row);
    //MPI_Win_fence(0, win_parentsRow);
    
    MPI_Win_free(&win_mateRow2Col);
    MPI_Win_free(&win_mateCol2Row);
    MPI_Win_free(&win_parentsRow);
}





// Maximum cardinality matching
// Output: mateRow2Col and mateRow2Col
template <typename Par_DCSC_Bool>
void maximumMatching(Par_DCSC_Bool & A, FullyDistVec<int64_t, int64_t>& mateRow2Col,
                     FullyDistVec<int64_t, int64_t>& mateCol2Row, bool prune=true, bool mvInvertMate = false, bool randMM = false)
{
    
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    int64_t nrow = A.getnrow();
    int64_t ncol = A.getncol();
    
    FullyDistSpVec<int64_t, VertexType> fringeRow(A.getcommgrid(), nrow);
    FullyDistSpVec<int64_t, int64_t> umFringeRow(A.getcommgrid(), nrow);
    FullyDistVec<int64_t, int64_t> leaves ( A.getcommgrid(), ncol, (int64_t) -1);
    
    vector<vector<double> > timing;
    vector<int> layers;
    vector<int64_t> phaseMatched;
    double t1, time_search, time_augment, time_phase;
    
    bool matched = true;
    int phase = 0;
    int totalLayer = 0;
    int64_t numUnmatchedCol;
    
    
    MPI_Win winLeaves;
    MPI_Win_create(leaves.GetLocArr(), leaves.LocArrSize() * sizeof(int64_t), sizeof(int64_t), MPI_INFO_NULL, A.getcommgrid()->GetWorld(), &winLeaves);
    
    
    while(matched)
    {
        time_phase = MPI_Wtime();
        
        Par_DCSC_Bool Mbool = PermMat<int64_t, SpDCCols<int64_t,bool>>(mateCol2Row, nrow);
        
        //#ifdef _OPENMP
        //if(Mbool.getnnz()>cblas_splits)
        //   Mbool.ActivateThreading(cblas_splits);
        //#endif
        
        
        vector<double> phase_timing(8,0);
        leaves.Apply ( [](int64_t val){return (int64_t) -1;});
        FullyDistVec<int64_t, int64_t> parentsRow ( A.getcommgrid(), nrow, (int64_t) -1);
        FullyDistSpVec<int64_t, VertexType> fringeCol(A.getcommgrid(), ncol);
        fringeCol  = EWiseApply<VertexType>(fringeCol, mateCol2Row,
                                            [](VertexType vtx, int64_t mate){return vtx;},
                                            [](VertexType vtx, int64_t mate){return mate==-1;},
                                            true, VertexType());
        
        
        if(randMM) //select rand
        {
            fringeCol.ApplyInd([](VertexType vtx, int64_t idx){return VertexType(idx,idx,static_cast<int16_t>((GlobalMT.rand() * 9999999)+1));});
        }
        else
        {
            fringeCol.ApplyInd([](VertexType vtx, int64_t idx){return VertexType(idx,idx);});
        }
        
        ++phase;
        numUnmatchedCol = fringeCol.getnnz();
        int layer = 0;
        
        
        time_search = MPI_Wtime();
        while(fringeCol.getnnz() > 0)
        {
            layer++;
            t1 = MPI_Wtime();
            SpMV<Select2ndMinSR<bool, VertexType>>(A, fringeCol, fringeRow, false);
            phase_timing[0] += MPI_Wtime()-t1;
            
            
            // remove vertices already having parents
            
            t1 = MPI_Wtime();
            fringeRow = EWiseApply<VertexType>(fringeRow, parentsRow,
                                               [](VertexType vtx, int64_t parent){return vtx;},
                                               [](VertexType vtx, int64_t parent){return parent==-1;},
                                               false, VertexType());
            
            // Set parent pointer
            parentsRow.EWiseApply(fringeRow,
                                  [](int64_t dval, VertexType svtx){return svtx.parent;},
                                  [](int64_t dval, VertexType svtx){return true;},
                                  false, VertexType());
            
            
            umFringeRow = EWiseApply<int64_t>(fringeRow, mateRow2Col,
                                              [](VertexType vtx, int64_t mate){return vtx.root;},
                                              [](VertexType vtx, int64_t mate){return mate==-1;},
                                              false, VertexType());
            
            phase_timing[1] += MPI_Wtime()-t1;
            
            
            int64_t nnz_umFringeRow = umFringeRow.getnnz(); // careful about this timing
            
            t1 = MPI_Wtime();
            if(nnz_umFringeRow >0)
            {
                if(nnz_umFringeRow < 25*nprocs)
                {
                    leaves.GSet(umFringeRow,
                                [](int64_t valRoot, int64_t idxLeaf){return valRoot;},
                                [](int64_t valRoot, int64_t idxLeaf){return idxLeaf;},
                                winLeaves);
                }
                else
                {
                    FullyDistSpVec<int64_t, int64_t> temp1(A.getcommgrid(), ncol);
                    temp1 = umFringeRow.Invert(ncol);
                    leaves.Set(temp1);
                }
            }
            
            phase_timing[2] += MPI_Wtime()-t1;
            
            
            
            
            // matched row vertices in the the fringe
            fringeRow = EWiseApply<VertexType>(fringeRow, mateRow2Col,
                                               [](VertexType vtx, int64_t mate){return VertexType(mate, vtx.root);},
                                               [](VertexType vtx, int64_t mate){return mate!=-1;},
                                               false, VertexType());
            
            t1 = MPI_Wtime();
            if(nnz_umFringeRow>0 && prune)
            {
                fringeRow.FilterByVal (umFringeRow,[](VertexType vtx){return vtx.root;}, false);
            }
            double tprune = MPI_Wtime()-t1;
            phase_timing[3] += tprune;
            
            
            // Go to matched column from matched row in the fringe. parent is automatically set to itself.
            t1 = MPI_Wtime();
            if(mvInvertMate)
                SpMV<Select2ndMinSR<bool, VertexType>>(Mbool, fringeRow, fringeCol, false);
            else
                fringeCol = fringeRow.Invert(ncol,
                                             [](VertexType& vtx, const int64_t & index){return vtx.parent;},
                                             [](VertexType& vtx, const int64_t & index){return vtx;},
                                             [](VertexType& vtx1, VertexType& vtx2){return vtx1;});
            phase_timing[4] += MPI_Wtime()-t1;
            
            
            
            
        }
        time_search = MPI_Wtime() - time_search;
        phase_timing[5] += time_search;
        
        int64_t numMatchedCol = leaves.Count([](int64_t leaf){return leaf!=-1;});
        phaseMatched.push_back(numMatchedCol);
        time_augment = MPI_Wtime();
        if (numMatchedCol== 0) matched = false;
        else
        {
            if(numMatchedCol < (2* nprocs * nprocs))
                AugmentPath<int64_t,int64_t>(mateRow2Col, mateCol2Row,parentsRow, leaves);
            else
                AugmentLevel(mateRow2Col, mateCol2Row,parentsRow, leaves);
        }
        time_augment = MPI_Wtime() - time_augment;
        phase_timing[6] += time_augment;
        
        time_phase = MPI_Wtime() - time_phase;
        phase_timing[7] += time_phase;
        timing.push_back(phase_timing);
        totalLayer += layer;
        layers.push_back(layer);
        
    }
    
    
    MPI_Win_free(&winLeaves);
    
    //isMaximalmatching(A, mateRow2Col, mateCol2Row, unmatchedRow, unmatchedCol);
    //isMatching(mateCol2Row, mateRow2Col); //todo there is a better way to check this
    
    
    // print statistics
    double combTime;
    if(myrank == 0)
    {
        cout << "****** maximum matching runtime ********\n";
        cout << endl;
        cout << "========================================================================\n";
        cout << "                                     BFS Search                       \n";
        cout << "===================== ==================================================\n";
        cout  << "Phase Layer    Match   SpMV EWOpp CmUqL  Prun CmMC   BFS   Aug   Total\n";
        cout << "===================== ===================================================\n";
        
        vector<double> totalTimes(timing[0].size(),0);
        int nphases = timing.size();
        for(int i=0; i<timing.size(); i++)
        {
            printf(" %3d  %3d  %8lld   ", i+1, layers[i], phaseMatched[i]);
            for(int j=0; j<timing[i].size(); j++)
            {
                totalTimes[j] += timing[i][j];
                //timing[i][j] /= timing[i].back();
                printf("%.2lf  ", timing[i][j]);
            }
            
            printf("\n");
        }
        
        cout << "-----------------------------------------------------------------------\n";
        cout  << "Phase Layer   UnMat   SpMV EWOpp CmUqL  Prun CmMC   BFS   Aug   Total \n";
        cout << "-----------------------------------------------------------------------\n";
        
        combTime = totalTimes.back();
        printf(" %3d  %3d  %8lld   ", nphases, totalLayer/nphases, numUnmatchedCol);
        for(int j=0; j<totalTimes.size()-1; j++)
        {
            printf("%.2lf  ", totalTimes[j]);
        }
        printf("%.2lf\n", combTime);
    }
    
    int64_t nrows=A.getnrow();
    int64_t matchedRow = mateRow2Col.Count([](int64_t mate){return mate!=-1;});
    if(myrank==0)
    {
        cout << "***Final Maximum Matching***\n";
        cout << "***Total-Rows Matched-Rows  Total Time***\n";
        printf("%lld %lld %lf \n",nrows, matchedRow, combTime);
        printf("matched rows: %lld , which is: %lf percent \n",matchedRow, 100*(double)matchedRow/(nrows));
        cout << "-------------------------------------------------------\n\n";
    }
    
}





