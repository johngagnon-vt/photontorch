"""
wg_factory=wg_factory,

The reck module implements a unitary matrix network based on the Reck network


Reference:
    https://journals.aps.org/prl/abstract/10.1103/NhysRevLett.73.58

"""

#############
## Imports ##
#############

# other
import numpy as np

# relative
from .network import Network

from ..components.mzm import Mzm
from ..components.mmis import Mmi
from ..components.terms import Source, Detector, Term
from ..components.waveguides import Waveguide

def _mzm_factory(theta = 2 * np.pi * np.random.rand(),normalize = False):
    return Mzm(
        theta=theta,
        phi=0,
        trainable=True,
        normalize=normalize
    )
    
def _wg_factory(phase = 0):
    return Waveguide(phase=phase,
                     length=0,
                     trainable=True)
    
def splitter(n=2):
    weights = np.sqrt(np.ones((1,n)))
    return Mmi(weights=weights)

def combiner(n=2):
    weights = np.sqrt(np.ones((n, 1)))
    return Mmi(weights=weights)
    
#############
## Classes ##
#############


class MVMNxN(Network):
    """A helper network for ReckNxN"""
    
    def __init__(
        self,
        N=2,
        mzm_factory=_mzm_factory,
        splitter_factory = splitter,
        combiner_factory = combiner,
        wg_factory = _wg_factory,
        phase_weights = None,
        name=None,
        weights = None,
        normalize = True,
    ):
        """
        Args:
            N (int): number of input / output ports (the network represents an NxN matrix)
            mzm_factory (callable): function without arguments which creates the MZIs or
                any other general 2-port component with  ports defined anti-clockwise.
            name (optional, str): name of the component

        """
        self.N = N
        num_mzms = N**2
        self.name=name
        # define components
        components = {}
        if ((weights is None) or (phase_weights is None)):
            for i in range(num_mzms):
                components["mzm%i" % i] = mzm_factory()
                components["phase_shift%i" % i] = wg_factory() #these have 0 length as we are treating them as a phase shift and not like real waveguides
        else:
            assert np.shape(weights) == (N,N)
            assert np.shape(phase_weights) == (N,N)
            
            for i in range(num_mzms):
                components["mzm%i" % i] = mzm_factory(theta= weights[i%N][i//N],
                                                      normalize = normalize)
                components["phase_shift%i" % i] = wg_factory(phase_weights[i//N][i%N]/2)
        
        for i in range(N):
            components["split%i" % i] = splitter_factory(N)
            components["comb%i" % i] = combiner_factory(N)
        
        # connections between mzms:
        connections = []
        for i in range(num_mzms):
            connections += ["mzm%i:0:split%i:%i" % (i, i//N,1+i%N)]
            connections += ["mzm%i:1:phase_shift%i:0" % (i, i)]
            connections += ["phase_shift%i:1:comb%i:%i" % (i, i%N ,i//N)]
            
        
        # input ports:
        for i in range(N):
            connections += ["split%i:0:%i" % (i, i)]
            
        # output ports
        for i in range(N):
            connections += ["comb%i:%i:%i" % (i,N, N+i)]
        #print("connections: ",connections)
        super(MVMNxN, self).__init__(components, connections, name=name)

        
        
    def terminate(self, term=None):
        """Terminate open conections with the term of your choice

        Args:
            term: (Term|list|dict): Which term to use. Defaults to Term. If a
                dictionary or list is specified, then one needs to specify as
                many terms as there are open connections.

        Returns:
            terminated network with sources on the bottom and detectors on the right.
        """
        if term is None:
            #print("creating source and detector")
            term = [Source(name="s%i" % i) for i in range(self.N)]
            term += [Detector(name="d%i" % i) for i in range(self.N)]
        #print("entering term init")
        ret = super(MVMNxN, self).terminate(term)
        ret.to(self.device)
        return ret
        