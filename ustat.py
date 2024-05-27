
import numpy as np
from scipy.special import binom
import itertools

def triE( x, y, z ):
    '''
    Compute the probability that three vertices form a triangle
    when they have latent positions x,y,z.
    '''

    # Error checking
    if len(x.shape) != 1 or len(y.shape) != 1 or len(z.shape) != 1:
        raise ValueError('Inputs should be vectors.')
    assert( x.shape[0]==y.shape[0] and y.shape[0]==z.shape[0] )

    xTy = np.clip( np.inner( x, y ), 0.0, 1.0 )
    xTz = np.clip( np.inner( x, z ), 0.0, 1.0 )
    zTy = np.clip( np.inner( z, y ), 0.0, 1.0 )

    return xTy * xTz * zTy

def triEmx( X ):
    if X.shape[0] != 3:
        raise ValueError('Input to triEmx should have three rows.')
    return triE( X[0,:], X[1,:], X[2,:] )

def edgeE( x, y ):
    '''
    Compute the probability that two vertices form an edge
    when they have latent positions x,y.
    '''

    # Error checking
    if len(x.shape) != 1 or len(y.shape) != 1:
        raise ValueError('Inputs should be vectors.')
    assert( x.shape[0]==y.shape[0] )

    return np.clip( np.inner( x, y ), 0.0, 1.0 )

def edgeEmx( X ):
    '''
    Transform a 2-by-d submatrix of the latent positions into
    two separate rows, and pass it into edge E.
    '''
    if len( X.shape ) != 2:
        raise ValueError('Input to edgeEmx should be a matrix.')
    if X.shape[0] != 2:
        raise ValueError('Input to edgeEmx should have two rows.')
    return edgeE( X[0,:], X[1,:] )

def weighted_bootstrap( X, h, arity, B ):
    '''
    Perform the generalized bootstrap as described in Bose and Chatterjee,
    with variables given by the rows of X (n times d),
    and kernel h.
    arity specifies the number of arguments passed to h,
    where it is assumed that h takes the form h(X'),
    where X' is a submatrix of m rows of X.
    Perform B bootstrap iterations.
    '''

    if type(arity) != int:
        raise ValueError('Arity must be an integer.')
    if arity <= 0:
        raise ValueError('Arity must be positive.')

    if len(X.shape) != 2:
        raise ValueError('X must be a matrix.')

    # Compute Utilde, per page 112 of Bose and Chatterjee.
    Utilde = construct_Utilde_exact( X, h, arity )

    # Now do bootstrap iterates.
    return weighted_Utilde_bootstrap( Utilde, B )

def construct_Utilde_exact2( X, h, arity ):
    '''
    Construct the vector Utilde, as on page 112 of Bose and Chatterjee,
    Utilde_i = binom(n-1,m-1)^{-1} \sum_{j_1<j_2<...<j_{m-1}; j_k != i}
				h(X_i,X_{j1},X_{j2},...,X_{jm-1})
    '''
    n = X.shape[0]
    Utilde = np.zeros(n)
    for s in itertools.combinations( range(n), arity ):
        hs = h( X[s,:] )
        for i in s:
            Utilde[ i ] = Utilde[ i ] + hs
    return Utilde/binom(n-1,arity-1)

def construct_Utilde_exact( X, h, arity ):
    '''
    Construct the vector Utilde, as on page 112 of Bose and Chatterjee,
    Utilde_i = binom(n-1,m-1)^{-1} \sum_{j_1<j_2<...<j_{m-1}; j_k != i}
				h(X_i,X_{j1},X_{j2},...,X_{jm-1})
    '''
    n = X.shape[0]
    Utilde = np.zeros(n)
    for s in itertools.combinations( range(n), arity ):
        hs = h( X[s,:] )
        idx = np.array(s)
        Utilde[ idx ] = Utilde[ idx ] + hs
    return Utilde/binom(n-1,arity-1)

def approx_weighted_bootstrap( X, h, arity, B, nsamp ):
    '''
    Perform the generalized bootstrap per Bose and Chatterjee,
    as above, but replace the full sum in Utilde[i] with a Monte Carlo
    approximation based on nsamp samples.
    '''
    if type(arity) != int:
        raise ValueError('Arity is not an integer, somehow?')
    if arity <= 0:
        raise ValueError('Arity must be positive.')
    if len(X.shape) != 2:
        raise ValueError('X must be a matrix.')

    # Compute Utilde, per page 112 of Bose and Chatterjee.
    n = X.shape[0]
    Utilde = np.zeros(n)
    available_inds = np.arange( n )
    for i in range(n):
        # Samples should not include i.
        available_inds = np.delete( available_inds, i)
        for _ in range(nsamp):
            inds = np.random.choice(available_inds, arity-1, replace=False)
            s = (i,)+tuple(inds)
            Utilde[i] += h( X[s,:] )
        # Put the index back.
        available_inds = np.insert(available_inds, i, i)
    Utilde = Utilde/nsamp

    return weighted_Utilde_bootstrap( Utilde, B )

def approx_weighted_bootstrap_alt( X, h, arity, B, nsamp ):
    '''
    Perform the generalized bootstrap per Bose and Chatterjee,
    as above, but replace the full sum in Utilde[i] with a Monte Carlo
    approximation based on nsamp samples.

    Do the summation in a slightly different way.
    '''
    if type(arity) != int:
        raise ValueError('Arity is not an integer, somehow?')
    if arity <= 0:
        raise ValueError('Arity must be positive.')

    if len(X.shape) != 2:
        raise ValueError('X must be a matrix.')

    # Compute Utilde, per page 112 of Bose and Chatterjee.
    n = X.shape[0]
    Utilde = np.zeros(n)
    sampcounts = np.zeros(n) #sampcounts[i] is # of times ind i was picked.
    available_inds = np.arange( n )
    for _ in range(nsamp):
        s = np.random.choice(available_inds, arity, replace=False)
        hs = h( X[s,:] )
        for i in s:
            Utilde[i] += hs
            sampcounts[i] += 1
    # Fix the sampcounts to not be zero.
    sampcounts[ sampcounts==0 ] = 1
    Utilde = Utilde/sampcounts

    return weighted_Utilde_bootstrap( Utilde, B )

def error_check_Utilde( X, arity ):
    if type(arity) != int:
        raise ValueError('Arity is not an integer, somehow?')
    if arity <= 0:
        raise ValueError('Arity must be positive.')
    if len(X.shape) != 2:
        raise ValueError('X must be a matrix.')

def construct_Utilde( X, h, arity, nsamp ):
    '''
    Use MC to approximate Utilde by replacing the sum for Utilde[i]
    with a Monte Carlo approximation based on nsamp samples.
    '''
    error_check_Utilde( X, arity )

    # Compute Utilde, per page 112 of Bose and Chatterjee.
    n = X.shape[0]
    Utilde = np.zeros(n)
    for i in range(n):
        available_inds = list(range(n))
        available_inds.remove(i) # Samples should not include i.
        for _ in range(nsamp):
            inds = np.random.choice(available_inds, arity-1, replace=False)
            s = (i,)+tuple(inds)
            hs = h( X[s,:] )
            Utilde[i] += hs
    return Utilde/nsamp

def construct_Utilde2( X, h, arity, nsamp ):
    '''
    Use MC to approximate Utilde.
    '''
    error_check_Utilde( X, arity )

    # Compute Utilde, per page 112 of Bose and Chatterjee.
    n = X.shape[0]
    Utilde = np.zeros(n)
    sampcounts = np.zeros(n) #sampcounts[i] is # of times ind i was picked.
    available_inds = np.arange( n )
    for _ in range(nsamp):
        s = np.random.choice(available_inds, arity, replace=False)
        Z = X[s,:]
        hs = h( Z )
        for i in s:
            Utilde[i] += hs
            sampcounts[i] += 1
    # Fix the sampcounts to not be zero.
    sampcounts[ sampcounts==0 ] = 1
    return Utilde/sampcounts

def weighted_Utilde_bootstrap( Utilde, B ):
    '''
    Do the weighted bootstrap for U-statistics from B&C pg 112ish.

    Utilde is an array of n sums. The i-th such sum is a sum over all the
    kernel evaluations that involve element i.
    '''

    n = Utilde.shape[0]

    bs_iters = np.zeros(B)
    for bb in range(B):
        # Generate the weights.
        M = np.random.multinomial(n=n, pvals=np.ones(n)/n)
        # Weights should sum to n, under their notation, so no normalize.
        bs_iters[bb] = np.dot( M, Utilde )/n # B&C eq 4.18, bottom page 112

    return bs_iters

def compute_ustat( X, h, arity ):
    '''
    Compute the U-statistic with kernel h on given data X,
    where X is a n-by-d matrix.
    h takes matrices of dimension arity-by-d
    '''
    if len( X.shape ) != 2:
        raise ValueError('X should be an n-by-d matrix of data.')

    (n, _) = X.shape

    # Compute this the slow way for now.
    U = 0.0
    for s in itertools.combinations( range(n), arity ):
        U += h( X[s,:] )
    return U/binom(n,arity)


