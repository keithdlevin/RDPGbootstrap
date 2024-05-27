
import itertools
import numpy as np
import numpy.linalg as npla
import scipy as sp
from scipy.linalg import fractional_matrix_power as fmp
#import scipy.special as spsp
from scipy.special import binom

def edgedensity( A ):
    '''
    Compute the edge density of A
    '''

    n = A.shape[0]
    np.fill_diagonal( A, 0 ) # Just to be safe 
    
    return np.sum(A)/( n*(n-1) )

def triden( A ):
    '''
    Compute the triangle density of A
    '''

    if len(A.shape) != 2:
        raise ValueError('Input should be a matrix.')
    if A.shape[0] != A.shape[1]:
        raise ValueError('Input should be square.')
    if not np.allclose(A,A.T,1e-12):
        raise ValueError('Input should be symmetric.')
    n = A.shape[0]

    # Make sure we don't get tricked by self-loops.
    np.fill_diagonal( A, 0 )

    tricount = np.trace( fmp( A, 3 ) )
    return tricount/(6*binom(n,3)) # 6 = 3!, accounts for automorphisms

def compute_edgedensity_from_X( X ):
    '''
    Compute the edge density for a given n-by-d X,
    nC2^{-1} \sum_{i < j} X_i^T X_j .
    '''

    if len( X.shape ) != 2:
        raise ValueError('X should be an n-by-d array.')

    n = X.shape[0]
    P = X @ X.T
    np.fill_diagonal( P, 0.0 )
    # now just compute sum_i,j P_ij, and adjust for
    # the fact that each i<j gets counted twice instead of just once.
    return np.sum(P)/(2*binom(n,2))

def compute_triden_from_X( X ):
    '''
    Compute the triangle density for a given n-by-d X,
    nC3^{-1} \sum_{i < j < k} X_i^T X_j X_j^T X_k X_k^T X_i
    '''

    if len( X.shape ) != 2:
        raise ValueError('X should be an n-by-d array.')

    n = X.shape[0]
    P = X @ X.T
    np.fill_diagonal( P, 0.0 )
    #Pcubed = spla.factional_matrix_power(P, 3) #TODO: time this against npla?
    Pcubed = npla.matrix_power(P, 3)
    # trace P^3 = sum_i,j,k X_i^T X_j X_j^T X_k X_k^T X_i
    # so each i<j<k triple gets counted six times instead of just once.
    return Pcubed.trace()/(6*binom(n,3))

def triden_fast( A ):
    '''
    Compute the triangle density of A
    in the setting where the network is sparse

    This should be faster,
    except that right now it relies on lots of for-loops
    '''

    if len(A.shape) != 2:
        raise ValueError('Input should be a matrix.')
    if A.shape[0] != A.shape[1]:
        raise ValueError('Input should be square.')
    if not np.allclose(A,A.T,1e-12):
        raise ValueError('Input should be symmetric.')
    n = A.shape[0]

    # Make sure we don't get tricked by self-loops.
    np.fill_diagonal( A, 0 )

    tricount = 0 # We'll increment this as we go.
    # For each vertex, retrieve its neighbors and count how many 
    # pairs of them form an edge.
    rowvec = None # Keep Python from using lots of memory
    for i in range(n):
        rowvec = A[i,:]
        neighbors = list( np.nonzero(rowvec)[0] )
        for (u,v) in itertools.combinations(neighbors, 2 ):
            tricount += A[u,v]
    # We have counted each triangle three times, so correct for that.
    tricount = tricount/3
    # Now compute the density
    return tricount/binom(n,3)

def CI_contains( CI, target ):
    if len( CI ) !=2:
        raise ValueError('CI should be a length-2 array-like object.')
    target = float(target)
    return ( (CI[0] <= target) and (target <= CI[1]) )

def randic( A ):
    '''
    Compute the (renormalized) Randic index of A
    https://en.wikipedia.org/wiki/Randi%C4%87%27s_molecular_connectivity_index
    nC2^{-1} sum_{i<j} 1/sqrt( d_i d_j )
    '''
    n = A.shape[0]
    deg = np.sum( A, axis=0 )
    if np.any( np.less_equal( deg, 0.1 ) ):
        deg = deg+1/n
    degprods = np.outer( deg, deg )

    return (np.sum( 1/np.sqrt(degprods) ) - np.sum( 1/deg ))/(2*binom(n,2))

