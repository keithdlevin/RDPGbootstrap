
'''
Implementation of Fangzheng and Yanxun's one-step estimator.
https://arxiv.org/pdf/1910.04333.pdf


'''

import numpy as np
import scipy.linalg as spla
import copy

import networks as net

def xx_onestep( A, d ):
    '''
    Given
    A : n-by-n adjacency matrix
    d : positive integer <= n, encoding the embedding dimension

    Compute the one-step ASE embedding, as given in Alg 1 of Xie and Xu
    https://arxiv.org/pdf/1910.04333.pdf
    '''

    # TODO: error check on A

    d = int(d)
    if d <= 0:
        raise ValueError('Embedding dimension d must be positive.')
    n = A.shape[0]
    if d > n:
        errmsg = 'Embedding dimension cannot exceed number of vertices.'
        raise ValueError( errmsg )

    # Steps 1 and 2: compute "vanilla" ASE
    Xtilde = net.ase( A, d )

    '''
    Step 3: update each of the rows based on a one-step update
    xhat_i = xtilde_i + Zi^{-1} Yi, where
    Zi \in d-by-d
    Zi = n^{-1} \sum_j xtildej xtildej^T/xtildei^Txtildej(1-xtildei^Txtildej)
    Yi is a d-vector
    Yi = n^{-1} \sum_j (Aij - xtildei^Txtildej)xtildej
      			 /xtildei^Txtildej(1-xtildei^Txtildej)
    '''
    # Ptilde[i,j] is xtildei^T xtildej 
    Ptilde = Xtilde @ Xtilde.T
    # Use to construct mx with i,j entry xtildei^Txtildej(1-xtildei^Txtildej)
    denom_mx = Ptilde * (1-Ptilde) 
    # Create an d-by-d-by-n tensor Z with Z[:,:,i] = Zi
    # TODO: figure out how to do this directly in numpy.
    # np.tensordot and np.outer both don't quite get it.
    outer_prods = np.zeros( (d,d,n) )
    for i in range(n):
        outer_prods[:,:,i] = np.outer( Xtilde[i,:], Xtilde[i,:] )
    # Other option:
    # Doing this in a for-loop to save memory, avoiding
    # having to build out the n-by-n-by-n tensor, which is expensive.
    # On the other hand, this would have us building the same
    # collection of n n-by-n outer products n times.
    # So i don't know; time-space tradeoff.
    Xhat = copy.deepcopy( Xtilde )
    # Now we'll go row by row and update.
    for i in range(n):
        denomrow = denom_mx[i,:]
        #Zi = np.inner( outer_prods, 1/denomrow )/n
        Zi = np.zeros( (d,d) )
        for j in range(n):
            Zi = Zi + outer_prods[:,:,j]/denom_mx[i,j]
        Zi_inv = spla.inv( Zi )
        # Now compute Yi
        numerrow = A[i,:]-Ptilde[i,:]
        multiprow = numerrow/denomrow
        #Yi = np.inner(  Xtilde.T, multiprow )/n
        Yi = np.zeros( d )
        for j in range(n):
            scalar = ((A[i,j]-Ptilde[i,j])/denom_mx[i,j])
            # Have to do this annoying np.array business to keep Yi shape.
            Yi = Yi + scalar*( np.array( Xtilde[j,:])[0] )
        Xhat[i,:] = Xhat[i,:] + Zi_inv@Yi

    return Xhat

