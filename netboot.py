import numpy as np
import scipy.stats as spstats
import networks as net
import netstat
import xx_onestep as xxose
import ustat

def bootstrap_interval( bootreps, alpha ):
    repmean = np.mean( bootreps )
    SDhat = np.std( bootreps )
    CI = centered_interval( repmean, SDhat, alpha )
    return ( CI, SDhat )

def centered_interval( center, sd, alpha ):
    Zscore = -spstats.norm.ppf( alpha/2, loc=0, scale=1 )
    CILB = center - Zscore*sd
    CIUB = center + Zscore*sd
    return (CILB, CIUB)

class ASENetResampler():
    def __init__( self, A, d, diagaug=None ):
        d = int(d)
        if d < 1:
            raise ValueError('embedding dimension d should be positive')
        self.d = d

        net.check_valid_adjmx( A )
        self.n = A.shape[0] 

        if diagaug:
            np.fill_diagonal( A, np.sum(A)/self.n )

        Xhat = net.ase( A, self.d )
        self.X = Xhat

    def edgeden_CI_classicboot( self, density_obsd, nboot=30, alpha=0.05 ):
        bootreps = nboot * [0.0]
        for i in range(nboot):
            Ahatstar = self.resample()
            bootreps[i] = netstat.edgedensity( Ahatstar )
        # Now estimate the SD from those resamples
        # and use it to construct upper- and lower-bounds for CI.
        SDhat = np.std( bootreps )
        CI = centered_interval( density_obsd, SDhat, alpha )
        return ( CI, SDhat )

    def triden_CI_classicboot( self, triden_obsd, nboot=30, alpha=0.05 ):
        bootreps = nboot * [0.0]
        for i in range(nboot):
            Ahatstar = self.resample()
            bootreps[i] = netstat.triden( Ahatstar )
        # Now estimate the SD from those resamples
        # and use it to construct upper- and lower-bounds for CI.
        SDhat = np.std( bootreps )
        CI = centered_interval( triden_obsd, SDhat, alpha )
        return ( CI, SDhat )

    def bootstrap( self, fn, nboot ):
        '''
        Produce nboot bootstrap replicates; compute fn on each one.
        return a numpy array storing the results.

        fn : a function that takes a single network (i.e., adj mx) as its
		only argument and outputs a numeric.
	nboot : non-negative integer; number of bootstrap replicates to prodce
        '''
        nboot = int(nboot)
        if nboot <= 0:
            msg='Number of bootstrap replicates should be positive.'
            raise ValueError( msg )

        bootreps = np.zeros( nboot )
        for i in range(nboot):
            bootreps[i] = fn( self.resample() )
            
        return bootreps

    def bootstrap_conditional( self, checkfn, netfn, nboot ):
        '''
        Generate nboot replicates Ahat, evaluating netfn on each,
        but ensure that every replicate has checkfn( Ahat )==True.
        '''

        bootreps = np.zeros( nboot )
        for i in range(nboot):
            while True:
                Ahat = self.resample()
                if checkfn( Ahat ): 
                    bootreps[i] = netfn( Ahat )
                    break

        return bootreps

    def CI_classic( self, fn, nboot, alpha, center ):
        bootreps = self.bootstrap( fn, nboot )
        return centered_interval( center, np.std(bootreps), alpha )

    def CI_classic2( self, fn, nboot, alpha, center ):
        bootreps = self.bootstrap( fn, nboot )
        CI = centered_interval( center, np.std(bootreps), alpha )
        bootSD = np.std(bootreps)
        return ( CI, bootSD )

    def CI_emp( self, fn, nboot, alpha ):
        '''
        Generate nboot replicate networks; compute fn(Ahat) on each.
        Use these to construct a CI centered at mean of the boots.
        '''
        bootreps = self.bootstrap( fn, nboot )

        return bootstrap_interval( bootreps, alpha ) 

    def CI_emp_conditional( self, checkfn, netfn, nboot, alpha ):
        '''
        Same as CI_emp, but make sure that all samples satisfy checkfn(Ahat).
        '''
        bootreps = self.bootstrap_conditional( checkfn, netfn, nboot )
        return bootstrap_interval( bootreps, alpha )

    def CI_classic_conditional( self, checkfn, netfn, nboot, alpha, center ):
        '''
        Same as CI_emp, but make sure that all samples satisfy checkfn(Ahat).
        '''
        bootreps = self.bootstrap_conditional( checkfn, netfn, nboot )
        sd = np.std( bootreps )
        CI = centered_interval( center, sd, alpha )
        return ( CI, sd )

class ASEMarginalResampler( ASENetResampler ):

    def resample( self, m=None ):
        '''
        Resample a network on m nodes. m Should default to self.n if unspec'd
        '''
        if m is None:
            m=self.n

        # Resample the latent positions
        idxs = np.random.choice( self.n, size=m, replace=True )
        Xresamp = self.X[idxs,:]
        # Resample a network, truncating the out-of-bounds entries.
        return net.gen_adj_from_posns( Xresamp )

class OSEMarginalResampler( ASEMarginalResampler ):
    def __init__( self, A, d, diagaug=False ):
        d = int(d)
        if d < 1:
            raise ValueError('embedding dimension d should be positive')
        self.d = d

        net.check_valid_adjmx( A )
        self.n = A.shape[0]

        if diagaug:
            np.fill_diagonal( A,np.sum(A)/self.n )

        Xhat = xxose.xx_onestep( A, self.d )
        self.X = Xhat 

class ASEConditionalResampler( ASENetResampler ):

    def resample( self ):
        '''
        Resample a network conditional on self.X
        '''

        return net.gen_adj_from_posns( self.X )

class BetaResampler( ASENetResampler ):
    def __init__(self, A, method='moments', embed='ase' ):
        net.check_valid_adjmx( A )

        import betaparam
        (ahat,bhat) = betaparam.estimate_beta_params_from_network( A, method='moments', embed='ase' )
        self.ahat=ahat
        self.bhat=bhat
        self.n=A.shape[0]

    def resample( self ):
        (A,_) = net.gen_beta_rdpg( self.n, self.ahat, self.bhat ) 
        return A

class EmpiricalGraphonResampler( ASEMarginalResampler ):
    def __init__( self, A ):
        net.check_valid_adjmx( A )
        self.n = A.shape[0]

        self.A = A # Also make extra sure we have zeros on diagonal.
        np.fill_diagonal( self.A, 0 )

    def resample( self, nstar=None ):
        '''
        Resample a network according to the empirical graphon.
        '''

        if nstar is not None:
            nstar = int(nstar)
        else:
            nstar = self.n

        # Drawnstar samples from {0,1,2,\dots,n-1}
        vxs = np.random.randint(self.n, size=nstar)
        # And use those indices to graph entries of original adj matrix.
        return self.A[np.ix_(vxs,vxs)]

class ASEUstatResampler( ASEMarginalResampler ):

    def __init__( self, A, d, h, arity, nMC=None, diagaug=None ):
        super().__init__(A, d, diagaug)

        assert( self.n==self.X.shape[0] ) 
        self.X = np.array(self.X) # Else grabbing rows doesn't work.

        self.h = h

        if type(arity) != int:
            raise ValueError('Arity is not an integer, somehow?')
        if arity <= 0:
            raise ValueError('Arity must be positive.')
        self.arity=arity
        if nMC is None:
            nMC=self.n # Default chosen based on 'calib' expts.
        self.nMC = nMC

        # Now, built the Utilde matrix and all that.
        #self.Utilde = ustat.construct_Utilde_exact( self.X, self.h, self.arity)
        self.Utilde = ustat.construct_Utilde( self.X, self.h,
						self.arity, self.nMC )

    def edgeden_CI_classicboot( self, density_obsd, nboot=30, alpha=0.05 ):
        # Obtain nboot resamples. multiply by arity to fix bug.
        bootreps = 2*self.resample( nboot )
        # Now estimate the SD from those resamples
        # and use it to construct upper- and lower-bounds for CI.
        SDhat = np.std( bootreps )
        #Zscore = -spstats.norm.ppf( alpha/2, loc=0, scale=1 )
        #CILB = density_obsd - Zscore*SDhat
        #CIUB = density_obsd + Zscore*SDhat
        CI = (CILB, CIUB)
        CI = centered_interval( density_obsd, SDhat, alpha )
        return ( CI, SDhat )

    def triden_CI_classicboot( self, triden_obsd, nboot=30, alpha=0.05 ):
        # Obtain nboot resamples. multiply by arity to fix bug.
        bootreps = 3*self.resample( nboot )
        # Now estimate the SD from those resamples
        # and use it to construct upper- and lower-bounds for CI.
        SDhat = np.std( bootreps )
        Zscore = -spstats.norm.ppf( alpha/2, loc=0, scale=1 )
        CILB = triden_obsd - Zscore*SDhat
        CIUB = triden_obsd + Zscore*SDhat
        CI = (CILB, CIUB)
        return ( CI, SDhat )

    def triden_CI_empboot( self, triden_obsd, nboot=30, alpha=0.05 ):
        # Obtain nboot resamples. multiply by arity to fix bug.
        bootreps = 3*self.resample( nboot )
        # Now estimate the SD from those resamples
        # and use it to construct upper- and lower-bounds for CI.
        SDhat = np.std( bootreps )
        Zscore = -spstats.norm.ppf( alpha/2, loc=0, scale=1 )
        # Center at the mean of the boostraps.
        bootmean = np.mean( bootreps )
        CILB = bootmean - Zscore*SDhat
        CIUB = bootmean + Zscore*SDhat
        CI = (CILB, CIUB)
        return ( CI, SDhat )

    def resample( self, B ):
        '''
        Resampling using the "alternate" Ustatistic bootstrap
	adapting Bose and Chatterjee, and approximating the Utilde_i
	entries via MC estimation.
        '''
        return ustat.weighted_Utilde_bootstrap( self.Utilde, B )

class OSEUstatResampler( OSEMarginalResampler ):

    def __init__( self, A, d, h, arity, nMC=None, diagaug=None ):
        super().__init__(A, d, diagaug)

        assert( self.n==self.X.shape[0] ) 
        self.X = np.array(self.X) # Else grabbing rows doesn't work.

        self.h = h

        if type(arity) != int:
            raise ValueError('Arity is not an integer, somehow?')
        if arity <= 0:
            raise ValueError('Arity must be positive.')

        self.arity=arity
        # Now, built the Utilde matrix and all that.
        if nMC is None:
            self.nMC=self.n # Default based on 'calib' expts
        self.Utilde = ustat.construct_Utilde( self.X, self.h,
						self.arity, self.nMC )

    def edgeden_CI_classicboot( self, density_obsd, nboot=30, alpha=0.05 ):
        # Obtain nboot resamples. Multiply by 2 to fix bug.
        bootreps = 2*self.resample( nboot )
        # Now estimate the SD from those resamples
        # and use it to construct upper- and lower-bounds for CI.
        SDhat = np.std( bootreps )
        #Zscore = -spstats.norm.ppf( alpha/2, loc=0, scale=1 )
        #CILB = density_obsd - Zscore*SDhat
        #CIUB = density_obsd + Zscore*SDhat
        #CI = (CILB, CIUB)
        CI = centered_interval( density_obsd, SDhat, alpha )
        return ( CI, SDhat )

    def triden_CI_classicboot( self, triden_obsd, nboot=30, alpha=0.05 ):
        # Obtain nboot resamples. Multiply by 3 to fix bug.
        bootreps = 3*self.resample( nboot )
        # Now estimate the SD from those resamples
        # and use it to construct upper- and lower-bounds for CI.
        SDhat = np.std( bootreps )
        Zscore = -spstats.norm.ppf( alpha/2, loc=0, scale=1 )
        CILB = triden_obsd - Zscore*SDhat
        CIUB = triden_obsd + Zscore*SDhat
        CI = (CILB, CIUB)
        return ( CI, SDhat )

    def resample( self, B ):
        '''
        Resampling using the "alternate" Ustatistic bootstrap
	adapting Bose and Chatterjee, and approximating the Utilde_i
	entries via MC estimation.
        '''
        return ustat.weighted_Utilde_bootstrap( self.Utilde, B )
