import numpy as np
import scipy.spatial.distance as spsd
import scipy.linalg as spla

def gen_er(n,p):
    '''
    Generate a random n-by-n ER(p) graph.
    '''
    n = int(n)
    Ncoins = n*(n-1)//2
    coinflips = np.random.binomial(1,p, size=Ncoins)
    return spsd.squareform(coinflips)

def gen_sbm( n, B, pivec ):
    '''
    Generate a random n-by-n SBM with given B matrix and pivec.
    '''
    
    # Verify that pivec is a probability vector.
    if len(pivec.shape) != 1:
        raise ValueError('pivec should be a vector.')
    # Verify that B is square symmetric.
    if len(B.shape) != 2:
        raise ValueError('B should be a matrix.')
    if B.shape[0] != B.shape[1]:
        raise ValueError('B should be square.')
    if not np.allclose(B,B.T,atol=1e-12):
        print( B )
        raise ValueError('B should be symmetric.')
    
    # Verify that B matrix and pivec agree in dimensions.
    K = pivec.shape[0]
    if B.shape[0] != K:
        raise ValueError('B and pivec should agree in lengths.')
    # Generate community assignments 
    Z = np.random.multinomial(n=1, size=n, pvals=pivec)
    A = gen_adj_from_P( Z @ B @ Z.T )
    return (A,Z)

def gen_beta_rdpg( n, a, b ):
    '''
    Generate a random n-by-n graph according to an RDPG with Beta(a,b)
    latent positions, with parameters as given.

    Return adjacency matrix A.
    '''

    n = int(n)
    if n < 1:
        raise ValueError('Number of vertices n should be positive.')
    (a, b) = (float(a), float(b))
    if a <=0 or b <=0:
        raise ValueError('Beta parameters a and b must be positive.') 

    def F(n):
        # Generate n independent draws from Beta(a,b)
        from scipy.stats import beta
        return beta.rvs( a, b, size=(n,1) )

    return gen_rdpg( n, F )

def gen_dirichlet_rdpg( n, alphavec ):
    '''
    Generate a random n-by-n graph according to an RDPG with Dirichlet
    latent positions, with parameter as given.

    Return adjacency matrix A.
    '''

    n = int(n)
    if n < 1:
        raise ValueError('Number of vertices n should be positive.')

    def F(n):
        # Generate n independent draws from Dir( alphavec )
        from scipy.stats import dirichlet
        return dirichlet.rvs( alphavec, size= n )

    return gen_rdpg( n, F )

def gen_dirimix( n, K, alpha1, alpha0 ):
    '''
    Generate a network from latent positions drawn accordingly:
    To generate a vector,
    draw k ~ unif( 1,2,...,K ).
    Draw X ~ Diri( alphavec )
	where alphavec[k] = alpha1 and alphavec[i] = alpha0 for i != k. 
    Draw Z ~ Beta( K-1, 1 )
    Latent position is then Z X.
    '''

    n = int(n)
    if n < 1:
        raise ValueError('Number of vertices n should be positive.')

    def F(n):
        # Generate n indenepdnent draws where we choose one of K
        # Dirichlet mixture components and draw from it.
        from scipy.stats import dirichlet

        counts = np.random.multinomial( n, np.ones(K)/K )
        # For each component, draw from Diri( alpha )
        # where alpha[k] = alpha1 and rest are alpha0
        X = np.zeros( (n, K) )
        idx=0
        for k in range(K):
            c = counts[k]
            if c==0:
                continue
            alphavec = alpha0*np.ones( K )
            alphavec[k] = alpha1
            X[idx:(idx+c), :] = dirichlet.rvs( alphavec, size=c )
            idx += c
        assert( idx==n )
        # Now scale the rows of X by Z ~ Beta.
        from scipy.stats import beta
        Z= beta.rvs( K-1, 1, size=(n,1) )
        return Z*X # Scale rows of X.

    return gen_rdpg( n, F )

def gen_rdpg( n, F ):
    '''
    Generate a random n-by-n graph according to an RDPG with latent position
    distribution F.
    F must be such that F(n) returns a matrix of n rows, each row being a
    draw from some distribution.
    ''' 

    n = int(n)
    if n < 1:
        raise ValueError('Number of vertices n should be positive.')

    X = F(n)
    A = gen_adj_from_posns(X)
    return (A,X)

def estimate_p(A):
    if len(A.shape) != 2:
        raise ValueError('Input should be a matrix.')
    if A.shape[0] != A.shape[1]:
        raise ValueError('Input should be square.')
    if not np.allclose(A,A.T,atol=1e-12):
        raise ValueError('Input should be symmetric.')
    np.fill_diagonal(A,0) # Make sure we can call squareform without incident.
    return np.mean( spsd.squareform(A) )

def check_valid_adjmx( A ):
    if len(A.shape) != 2:
        raise ValueError('Input should be a matrix.')
    if A.shape[0] != A.shape[1]:
        raise ValueError('Input should be square.')
    if not np.allclose(A,A.T,atol=1e-12):
        raise ValueError('Input should be symmetric.')

def ase(A,d):
    check_valid_adjmx( A )

    n = A.shape[0]
    (S,U) = spla.eigh(A,eigvals=(n-d,n-1))
    # Scale the eigenvectors by square roots of eigenvalues.
    # This is the ASE, Xhat = U S^{1/2}
    return np.matrix(U)*np.matrix(np.diag(np.sqrt(S)))

def gen_adj_from_posns(X, clipval=0.0):
    '''
    Generate an adjacency matrix with expectation X X^T.
    '''
    n = X.shape[0]
    # Avoid issue where Xhats are out of range
    #P = np.maximum( clipval, np.minimum(X * X.T,1-clipval))
    P = np.clip( X @ X.T, clipval, 1-clipval )
    #P = np.maximum(0, np.minimum(X * X.T,1))
    return gen_adj_from_P( P )

def gen_adj_from_P( P ):
    '''
    Given P = \E A, generate a hollow binary matrix with independent edge.
    Assume that P is as-given, etc.
    '''
    # Verify square symmetric P
    if len(P.shape) != 2:
        raise ValueError('Input should be a matrix.')
    if P.shape[0] != P.shape[1]:
        raise ValueError('Input should be square.')
    # Ensure that P is symmetric; np.allclose causes errors
    P = (P + P.T)/2
    # Hollow out P, generate edges, and re-form a matrix.
    np.fill_diagonal(P,0)
    # If entries of P are really small, symmetry check will still fail.
    # We just ensured that P is as symmetric as possible and has zero diag,
    # so there's nothing else for squareform to check.
    # Therefore, we suppress the error checks.
    probs = spsd.squareform(P, checks=False)
    probs = np.clip( probs, 0.0, 1.0 )
    coinflips = np.random.binomial(n=1,p=probs)
    return spsd.squareform(coinflips)

def laplacian( A, reg=True ):
    '''
    Compute the Laplacian of A. If reg=True, regularize by the avg degree.

    A : adj matrix.
    reg : Boolean. If True, regularize the degrees in computing the Laplacian

    Returns
    L : same dimension as A; symmetric normalized graph Laplacian.
    '''

    check_valid_adjmx( A )
    
    degrees = np.sum( A, axis=0 ) 
    if reg:
        avgdeg = np.mean( degrees )
        degrees = degrees + avgdeg
    Dsqrtinv = np.diag( 1/np.sqrt(degrees) )
    return Dsqrtinv @ A @ Dsqrtinv
