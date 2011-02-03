import time
import scipy as sc

import sys
sys.path.append('/home/spr/kdt/trunk/Python/pyCombBLAS')
sys.path.append('/home/spr/kdt/trunk/Python/distribCombBLAS')

import pyCombBLAS as pcb
import DiGraph as kdtdg
#from DiGraph import DiGraph

def k2Validate(G, start, parents):
	good = True;
	
	ret = G.isBfsTree(start, parents);
	#	isBfsTree implements Graph500 tests 1 and 2 
	if type(ret) != tuple:
		print "isBfsTree detected failure of Graph500 test %d" % abs(ret);
		good = False;
		return;
	(valid, levels) = ret;

	# Spec test #3:
	# every input edge has vertices whose levels differ by no more than 1
	# Note:  don't actually have input edges, will use the edges in
	#    the resulting graph as a proxy
	[origI, origJ, ign] = G.toParVec();
	li = levels[origI]; 
	lj = levels[origJ];
	if not ((abs(li-lj) <= 1) | ((li==-1) & (lj==-1))).all():
		print "At least one graph edge has endpoints whose levels differ by more than one and is in the BFS tree"
		good = False;

	# Spec test #4:
	# the BFS tree spans a connected component's vertices (== all edges 
	# either have both endpoints in the tree or not in the tree, or 
	# source is not in tree and destination is the root)
	neither_in = (li == -1) & (lj == -1);
	both_in = (li > -1) & (lj > -1);
	out2root = (li == -1) & (origJ == start);
	if not (neither_in | both_in | out2root).all():
		print "The tree does not span exactly the connected component, root=%d" % start
		good = False;

	# Spec test #5:
	# a vertex and its parent are joined by an edge of the original graph
	respects = abs(li-lj) <= 1
	if not (neither_in | respects).all():
		print "At least one vertex and its parent are not joined by an original edge"
		good = False;

	return good;



scale = 8;
nstarts = 64;

GRAPH500 = 2;
if GRAPH500 == 1:
	print 'Using Graph500 graph generator'
	G = kdtdg.DiGraph();
	degrees = kdtdg.ParVec.zeros(4);
	K1elapsed = G.genGraph500Edges(scale, degrees);

	#	indices of vertices with degree > 2
	deg3verts = (G.degree() > 2).findInds();
	starts = sc.random.randint(0, high=len(deg3verts), size=(nstarts,));
	# deg3verts stays distributed; indices to it (starts) are scalars
elif GRAPH500 == 2:
	print 'Using fully connected graph generator'
	G = kdtdg.DiGraph.fullyConnected(2**scale,2**scale)
	K1elapsed = 0.00005;
	starts = sc.random.randint(0, high=2**scale, size=(nstarts,));
elif GRAPH500 == 3:
	print 'Loading small_nonsym_int.mtx'
	G = kdtdg.DiGraph.load('small_nonsym_int.mtx')
	K1elapsed = 0.00005;
	starts = sc.random.randint(0, 9, size=(nstarts,));

G.onesWeight();		# set all values to 1

K2elapsed = 1e-12;
K2edges = 0;
for start in starts:
	start = int(start);
	before = time.clock();
	parents = G.bfsTree(start);
	K2elapsed += time.clock() - before;
	if not k2Validate(G, start, parents):
		print "Invalid BFS tree generated by bfsTree";
		break;
	[origI, origJ, ign] = G.toParVec();
	K2edges += len((parents[origI] != -1).find());

if kdtdg.master():
	print 'Graph500 benchmark run for scale = %2i' % scale
	print 'Kernel 1 time = %8.4f seconds' % K1elapsed
	print 'Kernel 2 time = %8.4f seconds' % K2elapsed
	print '                    %8.4f seconds for each of %i starts' % (K2elapsed/nstarts, nstarts)
	print 'Kernel 2 TEPS = %7.4e' % (K2edges/K2elapsed)
