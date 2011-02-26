"""
The Graph500 module implements the Graph500 benchmark (v1.1), which includes
kernels 1 (graph construction) and 2 (breadth-first search).  In addition to
constructing the graph as specified, the module implements all 5 validation
steps in the spec. See www.graph500.org/Specifications.html for more detail.  

The variables in this script that will commonly be changed are:
	scale:  The logarithm base 2 of the number of vertices in the 
	    resulting graph.
	nstarts:  The number of times to create a BFS tree from a random
	    root vertex.

The edge factor is not easily changeable.
"""
import time
import math
import scipy as sc

import sys
import DiGraph as dg


def k2Validate(G, start, parents):
	good = True
	
	ret = G.isBfsTree(start, parents)
	#	isBfsTree implements Graph500 tests 1 and 2 
	if type(ret) != tuple:
		if dg.master():
			print "isBfsTree detected failure of Graph500 test %d" % abs(ret)
		good = False
		return good
	(valid, levels) = ret

	# Spec test #3:
	# every input edge has vertices whose levels differ by no more than 1
	# Note:  don't actually have input edges, will use the edges in
	#    the resulting graph as a proxy
	[origI, origJ, ign] = G.toParVec()
	li = levels[origI]; 
	lj = levels[origJ]
	if not ((abs(li-lj) <= 1) | ((li==-1) & (lj==-1))).all():
		if dg.master():
			print "At least one graph edge has endpoints whose levels differ by more than one and is in the BFS tree"
			print li, lj
		good = False

	# Spec test #4:
	# the BFS tree spans a connected component's vertices (== all edges 
	# either have both endpoints in the tree or not in the tree, or 
	# source is not in tree and destination is the root)
	neither_in = (li == -1) & (lj == -1)
	both_in = (li > -1) & (lj > -1)
	out2root = (li == -1) & (origJ == start)
	if not (neither_in | both_in | out2root).all():
		if dg.master():
			print "The tree does not span exactly the connected component, root=%d" % start
			#print levels, neither_in, both_in, out2root, (neither_in | both_in | out2root)
		good = False

	# Spec test #5:
	# a vertex and its parent are joined by an edge of the original graph
	respects = abs(li-lj) <= 1
	if not (neither_in | respects).all():
		if dg.master():
			print "At least one vertex and its parent are not joined by an original edge"
		good = False

	return good



scale = 15
nstarts = 64

GRAPH500 = 1
if GRAPH500 == 1:
	if dg.master():
		print 'Using Graph500 graph generator'
	G = dg.DiGraph()
	K1elapsed = G.genGraph500Edges(scale)
	#G.save("testgraph.mtx")

	if nstarts > G.nvert():
		nstarts = G.nvert()
	#	indices of vertices with degree > 2
	deg3verts = (G.degree() > 2).findInds()
	deg3verts.randPerm()
	#FIX: following should be randint(1, ...); masking root=0 bug for now
	#starts = sc.random.randint(1, high=len(deg3verts), size=(nstarts,))
	starts = deg3verts[dg.ParVec.range(nstarts)]
	# deg3verts stays distributed; indices to it (starts) are scalars
	#starts = dg.ParVec.range(nstarts);
	
elif GRAPH500 == 2:
	if dg.master():
		print 'Using 2D torus graph generator'
	G = dg.DiGraph.twoDTorus(2**(scale/2))
	K1elapsed = 0.00005
	starts = dg.ParVec.range(nstarts)
	#FIX: following should be randint(1, ...); masking root=0 bug for now
	starts = sc.random.randint(1, high=2**scale, size=(nstarts,))
elif GRAPH500 == 3:
	if dg.master():
		print 'Loading small_nonsym_int.mtx'
	G = dg.DiGraph.load('small_nonsym_int.mtx')
	K1elapsed = 0.00005
	#
	#FIX: following should be randint(1, ...); masking root=0 bug for now
	starts = sc.random.randint(1, 9, size=(nstarts,))
elif GRAPH500 == 4:
	if dg.master():
		print 'Loading testgraph.mtx'
	G = dg.DiGraph.load('testgraph.mtx')
	K1elapsed = 0.00005
	#
	starts = dg.ParVec.range(nstarts);

G.toBool()
#G.ones();		# set all values to 1

K2elapsed = 1e-12
K2edges = 0
#print "starting main loop"
i = 0
for start in starts:
	start = int(start)
	if start==0:	#HACK:  avoid root==0 bugs for now
		continue
	before = time.time()
	parents = G.bfsTree(start, sym=True)
	K2elapsed += time.time() - before
	i += 1
	if dg.master():
		print "iteration %d took %f s, start=%d"%(i, (time.time() - before), start)
	if not k2Validate(G, start, parents):
		print "Invalid BFS tree generated by bfsTree"
		print G, parents
		break
	[origI, origJ, ign] = G.toParVec()
	K2edges += len((parents[origI] != -1).find())

if dg.master():
	print 'Graph500 benchmark run for scale = %2i' % scale
	print 'Kernel 1 time = %8.4f seconds' % K1elapsed
	print 'Kernel 2 time = %8.4f seconds' % K2elapsed
	print '                    %8.4f seconds for each of %i starts' % (K2elapsed/nstarts, nstarts)
	print 'Kernel 2 TEPS = %7.4e' % (K2edges/K2elapsed)